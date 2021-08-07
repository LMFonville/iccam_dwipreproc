#!/usr/bin/env python
import sys
if len(sys.argv)>1:
    ppNo = str(sys.argv[1])
    subject_id=ppNo
    if subject_id in ['sub-3008', 'sub-3038']:
        print("Participant %s is excluded from DTI analyses" % subject_id)
        exit()
else:
    print ("No input argument")
    exit()

import os

#Add FSL
os.environ["FSLDIR"] = "/apps/fsl/6.0.1/fsl"
os.environ["PATH"] += ":/apps/fsl/6.0.1/fsl/bin"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
# Add Cuda libraries
os.environ["CUDA_HOME"] = '/apps/cuda/8.0.44/'
os.environ["LD_LIBRARY_PATH"] = '/apps/cuda/8.0.44/lib64:/apps/cuda/8.0.44/openclprof/bin:/apps/cuda/8.0.44/computeprof/bin:/apps/cuda/8.0.44/sdk/lib:/apps/cuda/8.0.44/sdk/common/lib/linux:/apps/cuda/8.0.44/sdk/bin/linux/release:/apps/fsl/6.0.1/fsl//lib'

#Cluster stuff
os.environ["TMPDIR"] = "/rds/general/ephemeral/user/lfonvill/ephemeral/"
os.environ["TMP"] = "/rds/general/ephemeral/user/lfonvill/ephemeral/"
os.environ["LOGNAME"] = "lfonvill"

from nipype import config
cfg = dict(execution={'stop_on_first_crash': True,
                      'remove_unnecessary_outputs' : False,
                      'local_hash_check': False,
                      'crashfile_format': 'txt',
                      'keep_inputs': True,
                      'hash_method': 'timestamp'})
config.update_config(cfg)

# Load modules
import numpy as np
import nibabel as nb
from os.path import join as opj

from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.base import TraitedSpec, File, traits, isdefined
from nipype.interfaces.fsl.base import FSLCommandInputSpec, FSLCommand
from nipype.interfaces.fsl import Eddy, ExtractROI

# Paths and parameters
preproc_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/re_dwipreproc/'
scratch_dir = opj('/rds/general/project/adobe_addiction_and_obesity_cohort/ephemeral', subject_id)

acqp_file = opj(preproc_dir, 'dwi_acqp_AP.txt') # switched from PA for some reason I had it the other way round before
index_file = opj(preproc_dir, 'dwi_index.txt')

# Custom function to use dipy median_otsu for masking
def mask_img(in_file):
    import nibabel as nib
    import numpy as np
    import os
    from dipy.segment.mask import median_otsu
    img = nib.load(in_file)
    data = img.get_data()
    header = img.header.copy()
    # Get mask and autocrop data
    maskdata, mask = median_otsu(data, vol_idx=(0,), median_radius=2, numpass=1, autocrop=False, dilate=4) #changed from default to improve temporal coverage
    b0_img = nib.Nifti1Image(maskdata[:,:,:,0].astype(np.float32), img.affine, header=header)
    b0Out = os.path.abspath('b0.nii')
    nib.save(b0_img, b0Out)
    #save mask output
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine, header=header)
    maskOut = os.path.abspath('dwi_mask.nii')
    nib.save(mask_img, maskOut)
    return (b0Out, maskOut)

# Set up a custom Eddy function to work with Cuda on our HPC
# Severely limited in use and not intended as a long-term substitute!
class customEddyInputSpec(FSLCommandInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='--imain=%s')
    in_mask = File(exists=True, mandatory=True, argstr='--mask=%s')
    in_index = File(exists=True, mandatory=True, argstr='--index=%s')
    in_acqp = File(exists=True, mandatory=True, argstr='--acqp=%s')
    in_bvec = File(exists=True, mandatory=True, argstr='--bvecs=%s')
    in_bval = File(exists=True, mandatory=True, argstr='--bvals=%s')
    out_base = traits.Str('eddy_corrected', argstr='--out=%s', usedefault=True)
    residuals = traits.Bool(False, argstr='--residuals')
class customEddyOutputSpec(TraitedSpec):
    out_parameter = File(exists=True)
    out_corrected = File(exists=True)
    out_rotated_bvecs = File(exists=True)
    out_outlier_report = File(exists=True)
    out_residuals = File(exists=True)
class customEddy(FSLCommand):
    _cmd = 'eddy_cuda8.0'
    input_spec = customEddyInputSpec
    output_spec = customEddyOutputSpec
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected'] = os.path.abspath('%s.nii.gz' % self.inputs.out_base)
        outputs['out_parameter'] = os.path.abspath('%s.eddy_parameters' % self.inputs.out_base)
        outputs['out_rotated_bvecs'] = os.path.abspath('%s.eddy_rotated_bvecs' % self.inputs.out_base)
        outputs['out_outlier_report'] = os.path.abspath('%s.eddy_outlier_report' % self.inputs.out_base)
        if isdefined(self.inputs.residuals) and self.inputs.residuals:
            outputs['out_residuals'] = os.path.abspath('%s.eddy_residuals.nii.gz' % self.inputs.out_base)
        return outputs

# Specify nodes

# Get input nifti files for dwi and t1
getdata = Node(DataGrabber(outfields=['dwi', 'bvec', 'bval'], sort_filelist=True), name = 'getdata')
getdata.inputs.base_directory = preproc_dir
getdata.inputs.template = '*'
getdata.inputs.field_template = dict(dwi='%s/preproc1_denoise/*dwi_den.nii.gz',bvec='%s/preproc1_denoise/*.bvec',bval='%s/preproc1_denoise/*.bval')
getdata.inputs.template_args = dict(dwi=[[subject_id]], bvec=[[subject_id]], bval=[[subject_id]])

# Get B0 again because dimensions can get slightly altered after MRtrix
getB0 = Node(ExtractROI(t_min=0, t_size=1, roi_file='B0.nii.gz',output_type='NIFTI_GZ'), name = 'getB0')

# Recreate brain mask
mask = Node(Function(input_names=['in_file'],
                      output_names=['b0_file', 'mask_file'],
                      function=mask_img), name = 'mask')

# Eddy correction, motion correction, and slice timing correcion
eddy = Node(customEddy(in_acqp=acqp_file, in_index=index_file, residuals=True, out_base=subject_id + '_eddy'),
                name = 'eddy')

# Creates output folder for important outputs
datasink = Node(DataSink(base_directory=preproc_dir,
                         container=subject_id),
                name="datasink")
substitutions=[
    ('.eddy', ''),
    ('_parameters', '_movement_parameters.txt')
]
datasink.inputs.substitutions = substitutions

preproc2_wf = Workflow(name='preproc2_wf', base_dir=scratch_dir)
preproc2_wf.connect([
    (getdata, mask, [('dwi', 'in_file')]),
    (getdata, eddy, [('dwi', 'in_file'),
                    ('bvec', 'in_bvec'),
                    ('bval', 'in_bval')]),
    (mask, eddy, [('mask_file', 'in_mask')]),
    (eddy, datasink, [('out_corrected', 'preproc2_eddy.@dwi_eddy'),
                     ('out_rotated_bvecs', 'preproc2_eddy.@bvec_eddy'),
                     ('out_residuals', 'preproc2_eddy.@residuals'),
                     ('out_parameter', 'preproc2_eddy.@params')]),
    (getdata, datasink, [('bval', 'preproc2_eddy.@bval')])
])

preproc2_wf.run()
