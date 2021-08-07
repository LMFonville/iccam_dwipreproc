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
#Cluster stuff
os.environ["TMPDIR"] = "/rds/general/ephemeral/user/lfonvill/ephemeral/"
os.environ["TMP"] = "/rds/general/ephemeral/user/lfonvill/ephemeral/"
os.environ["LOGNAME"] = "lfonvill"

from nipype import config
cfg = dict(execution={'stop_on_first_crash': True,
                      'remove_unnecessary_outputs' : False,
                      'local_hash_check': True,
                      'crashfile_format': 'txt', 
                      'keep_inputs': True,
                      'hash_method': 'content'})
config.update_config(cfg)

# Load modules
import numpy as np
import nibabel as nb
from os.path import join as opj

from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.fsl import ExtractROI, ImageStats, ImageMaths, DTIFit, MultiImageMaths

# Paths and parameters
experiment_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/'
preproc_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/re_dwipreproc/'
scratch_dir = opj('/rds/general/project/adobe_addiction_and_obesity_cohort/ephemeral', subject_id)

# Custom functions
def medianWM(stat):
    stat=1000/stat
    formula='-mul {:.4f}'.format(stat)
    return formula

def reslice(in_file):
    import os
    import nibabel as nib
    from dipy.align.reslice import reslice
    img = nib.load(in_file)
    data = img.get_data()
    vox_res = img.header.get_zooms()[:3]
    new_vox_res = (1.5, 1.5, 1.5)
    new_data, new_affine = reslice(data, img.affine, vox_res, new_vox_res)
    reslicedOut = os.path.abspath('DWI_upsampled.nii.gz')
    nib.save(nib.Nifti1Image(new_data, new_affine), reslicedOut)
    return reslicedOut

def mask_img(in_file):
    import nibabel as nib
    import os
    import numpy as np
    from dipy.segment.mask import median_otsu
    img = nib.load(in_file)
    data = img.get_data()
    maskdata, mask = median_otsu(data, median_radius=2, numpass=1, autocrop=False, dilate=1) 
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine)
    maskOut = os.path.abspath('B0_mask.nii')
    nib.save(mask_img, maskOut)
    return maskOut

# Specify nodes
# Get input nifti files for dwi and t1
# add raw_dwi and dwi_eddy to produce different FA maps


getdata = Node(DataGrabber(outfields=['dwi','bvec', 'bval', 'b0', 'wm'], sort_filelist=True), name = 'getdata')
getdata.inputs.base_directory = preproc_dir
getdata.inputs.template = '*'
getdata.inputs.field_template = dict(dwi='%s/preproc3_distortion/*_dwi_synwarp.nii.gz',
                                     bvec='%s/preproc2_eddy/*_rotated_bvecs',
                                     bval='%s/preproc2_eddy/*.bval',
                                     b0='%s/preproc3_distortion/*_B0_synwarp.nii.gz',
                                     wm='%s/preproc3_distortion/*_WM_to_DWI.nii.gz')
getdata.inputs.template_args = dict(dwi=[[subject_id]], bvec=[[subject_id]], bval=[[subject_id]], b0=[[subject_id]], wm=[[subject_id]])

# Get median WM intensity using the warped WM mask
getmedian = Node(ImageStats(op_string='-k %s -P 50'), name='getmedian')

# Create formula to normalise WM intensity as 1000/medianWM
wmformula = Node(Function(input_names=['stat'],
                    output_names=['formula'],
                    function=medianWM), name = 'wmformula')

# Apply the WM intensity normalisation within the mask
normalisewm = Node(ImageMaths(out_file=subject_id + '_dwi_norm.nii.gz'), name = 'normalisewm')

# Upsample the DWI data to 1.5mm isotropic 
upsample = Node(Function(input_names=['in_file'],
                        output_names=['out_file'],
                        function=reslice), name = 'upsample')

#Get B0 one last time to make a mask for tensor fitting
getB0 = Node(ExtractROI(t_min=0, t_size=1, roi_file=subject_id + '_B0.nii.gz',output_type='NIFTI_GZ'), name = 'getB0')

# Make a mask from B0 using the upsampled data
mask = Node(Function(input_names=['in_file'],
                      output_names=['mask_file'],
                      function=mask_img), name = 'mask')

# If any holes still occur
correctmask = Node(ImageMaths(op_string='-fillh', out_file=subject_id + '_mask.nii.gz'), name='correctmask')

# Estimate the tensor
dtifit = Node(DTIFit(base_name=subject_id + '_dtifit'), name = 'dtifit')

# Calculate radial diffusivity
calcRD = Node(MultiImageMaths(op_string='-add %s -div 2'), name = 'calcRD')

# Create output folder for important outputs
datasink = Node(DataSink(base_directory=preproc_dir,
                         container=subject_id),
                name="datasink")
substitutions=[
    ('dtifit_', ''),
    ('L2_maths', 'RD'),
    ('L1', 'AD')
]
datasink.inputs.substitutions = substitutions

# Set up workflow
preproc4_wf = Workflow(name='preproc4_wf', base_dir=scratch_dir)
preproc4_wf.connect([
    #normalise white matter signal intensity
    (getdata, getmedian, [('b0', 'in_file'),
                         ('wm', 'mask_file')]),
    (getmedian, wmformula, [('out_stat', 'stat')]),
    (wmformula, normalisewm, [('formula', 'op_string')]),
    (getdata, normalisewm, [('dwi', 'in_file')]),
    (normalisewm, upsample, [('out_file', 'in_file')]),
    (upsample, getB0, [('out_file', 'in_file')]),
    (getB0, mask, [('roi_file', 'in_file')]),
    (mask, correctmask, [('mask_file', 'in_file')]),
    # use dtifit
    (upsample, dtifit, [('out_file', 'dwi')]),
    (correctmask, dtifit, [('out_file', 'mask')]),
    (getdata, dtifit, [('bval', 'bvals'),
                       ('bvec', 'bvecs')]),
    (dtifit, calcRD, [('L2', 'in_file'),
                     ('L3', 'operand_files')]),
    #datasink that shit
    (correctmask, datasink, [('out_file', 'preproc4_tensor.@mask')]),
    (dtifit, datasink, [('FA', 'preproc4_tensor.@FA'),
                        ('MD', 'preproc4_tensor.@MD'),
                        ('L1', 'preproc4_tensor.@AD')]),
    (calcRD, datasink, [('out_file', 'preproc4_tensor.@RD')])
])

preproc4_wf.run()