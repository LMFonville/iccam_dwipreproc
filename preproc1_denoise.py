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

from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataGrabber, DataSink

# Paths and parameters
data_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/'
experiment_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/'
scratch_dir = opj('/rds/general/project/adobe_addiction_and_obesity_cohort/ephemeral', subject_id)
out_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/re_dwipreproc/'

# Custom function for local PCA denoising
def mppca_dwi(in_file):
    import nibabel as nib
    import numpy as np
    import os
    from dipy.denoise.localpca import mppca
    img = nib.load(in_file)
    data = img.get_data()
    header = img.header.copy()
    [denoised_data, sigma] = mppca(data, patch_radius=2, return_sigma=True)
    denoisedOut = os.path.abspath('denoised_mppca.nii.gz')
    nib.save(nib.Nifti1Image(denoised_data.astype(np.float32), img.affine, header=header), denoisedOut)
    sigmaOut = os.path.abspath('sigma_mppca.nii.gz')
    nib.save(nib.Nifti1Image(sigma.astype(np.float32), img.affine, header=header), sigmaOut)
    residualsOut = os.path.abspath('rms_residuals_mppca.nii.gz')
    rms_diff = np.sqrt((data - denoised_data) ** 2)
    nib.save(nib.Nifti1Image(rms_diff.astype(np.float32), img.affine, header=header), residualsOut)
    return (denoisedOut, residualsOut, sigmaOut)
    # add calculation of SNR at b0 using mask with sigma output

# Remove Gibbs unringing
def gibbs_remove(in_file):
    import nibabel as nib
    import numpy as np
    import os
    from dipy.denoise.gibbs import gibbs_removal
    img = nib.load(in_file)
    data = img.get_data()
    header = img.header.copy()
    data_corrected = gibbs_removal(data, slice_axis=2)
    correctedOut = os.path.abspath('dwi_den_dg.nii.gz')
    nib.save(nib.Nifti1Image(data_corrected.astype(np.float32), img.affine, header=header), correctedOut) # make sure to retain the header
    return correctedOut

# Specify nodes

# Get input nifti files for dwi and t1
getdata = Node(DataGrabber(outfields=['dwi','bvec','bval','anat'], sort_filelist=True), name = 'getdata')
getdata.inputs.base_directory = experiment_dir
getdata.inputs.template = '*'
getdata.inputs.field_template = dict(dwi='nifti/%s/dwi/*.nii.gz',bvec='nifti/%s/dwi/*.bvec',bval='nifti/%s/dwi/*.bval',anat='nifti/%s/anat/*.nii.gz')
getdata.inputs.template_args = dict(dwi=[[subject_id]],bvec=[[subject_id]],bval=[[subject_id]], anat=[[subject_id]])

# NEW APPROACH USING DIPY
# Apply mppca denoising
denoise_mppca = Node(Function(input_names=['in_file'],
                      output_names=['denoised_file', 'residuals', 'sigma'],
                      function=mppca_dwi), name = 'denoise_mppca')

# Remove Gibbs ringing artifacts
degibbs_dipy = Node(Function(input_names=['in_file'],
                      output_names=['corrected_file'],
                      function=gibbs_remove), name = 'degibbs_dipy')

# Creates output folder for important outputs
datasink = Node(DataSink(base_directory=out_dir,
                         container=subject_id),
                name="datasink")
substitutions=[
    ('dwi_den_dg', subject_id + '_dwi_den'),
    ('rms_residuals_localpca', subject_id + '_rms_lpca_defaults'),
    ('rms_residuals_mppca', subject_id + '_rms_mppca'),
    ('sigma_mppca', subject_id + '_sigma_mppca')
]
datasink.inputs.substitutions = substitutions

preproc1_wf = Workflow('preproc1_wf', base_dir=scratch_dir)
preproc1_wf.connect([
    (getdata, denoise_mppca, [('dwi', 'in_file')]),
    (denoise_mppca, degibbs_dipy, [('denoised_file', 'in_file')]),
    (degibbs_dipy, datasink, [('corrected_file', 'preproc1_denoise.@dwi_den')]),
    (getdata, datasink, [('bvec', 'preproc1_denoise.@bvec'),
                        ('bval', 'preproc1_denoise.@bval')]),
    (denoise_mppca, datasink, [('residuals', 'preproc1_denoise.@res_mppca'),
                              ('sigma', 'preproc1_denoise.@sigma_mppca')])
])


preproc1_wf.run(plugin="MultiProc", plugin_args={'n_procs' : 8, 'memory_gb' : 8})
