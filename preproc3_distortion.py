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
#Add ANTs
os.environ["PATH"] += ":/apps/ants/2.2.0/bin"

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
from nipype.interfaces.io import DataGrabber, DataSink, FreeSurferSource
from nipype.interfaces.fsl import ImageStats, ImageMaths
from nipype.interfaces.ants import N4BiasFieldCorrection, Registration, ApplyTransforms
from nipype.interfaces.freesurfer import FSCommand, Binarize

# Paths and parameters
preproc_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/dwipreproc/'
scratch_dir = opj('/rds/general/project/adobe_addiction_and_obesity_cohort/ephemeral', subject_id)
freesurfer_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/freesurfer/'

#custom functions
def mask_img(in_file):
    import nibabel as nib
    import numpy as np
    import os
    from dipy.segment.mask import median_otsu
    img = nib.load(in_file)
    data = img.get_data()
    b0 = data[:,:,:,0]
    header = img.header.copy()
    # Get mask
    maskdata, mask = median_otsu(data,vol_idx=(0,) ,median_radius=4, numpass=4, autocrop=False, dilate=2)
    b0_img = nib.Nifti1Image(maskdata[:,:,:,0].astype(np.float32), img.affine, header=header)
    b0Out = os.path.abspath('b0.nii')
    nib.save(b0_img, b0Out)
    #save mask output
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.affine, header=header)
    maskOut = os.path.abspath('dwi_mask.nii')
    nib.save(mask_img, maskOut)
    return (b0Out, maskOut)

def calcT1(statT1, ratio):
    maxStat=statT1[1]
    formula='-mul -1 -add {:.2f} -mul {:.2f}'.format(maxStat, ratio)
    return formula

def calcratio(statT1, statB0):
    minT1=statT1[0]
    maxT1=statT1[1]
    minB0=statB0[0]
    maxB0=statB0[1]
    ratio=(maxT1-minT1)/(maxB0-minB0)
    return ratio

def get_warp(transforms): # grab the warp from the forward transforms
    """Return the transform1Warp.nii.gz file"""
    for name in transforms:
        if 'transform1Warp.nii.gz' in name:
            return name
    raise ValueError('transform1Warp.nii.gz not found')

def extract_warp(warp):# fix the warp file to grab the flow field in the y plane
    import os
    import numpy as np
    import nibabel as nb
    img = nb.load(warp)
    data = img.get_data()
    new_data = data.squeeze() #change from 5D file to a 4D file
    warp_data = new_data[:, :, :, 1] #the fourth index now indicates the warp along the x, y, z planes
    warp_img = nb.Nifti1Image(warp_data.astype(np.float32), img.affine)
    warpOut = os.path.abspath('transform1Warp_Y.nii.gz')
    nb.save(warp_img, warpOut)
    return warpOut

def get_aparc_aseg(files):
    for name in files:
        if 'aparc+aseg' in name:
            return name
    raise ValueError('aparc+aseg not found')

# Specify nodes
# Get input nifti files for dwi and t1
getdata = Node(DataGrabber(outfields=['dwi','bvec','bval','anat', 'mask', 'wm'], sort_filelist=True), name = 'getdata')
getdata.inputs.base_directory = preproc_dir
getdata.inputs.template = '*'
getdata.inputs.field_template = dict(dwi='%s/preproc2_eddy/*_eddy.nii.gz',
                                     bvec='%s/preproc2_eddy/*_rotated_bvecs',
                                     bval='%s/preproc2_eddy/*.bval',
                                     anat='%s/preproc0_segment/*_Brain.nii.gz',
                                     mask='%s/preproc0_segment/*_BrainMask.nii.gz',
                                     wm = '%s/preproc0_segment/*_WM.nii.gz')
getdata.inputs.template_args = dict(dwi=[[subject_id]], bvec=[[subject_id]], bval=[[subject_id]], anat=[[subject_id]], mask=[[subject_id]], wm=[[subject_id]])

# Data grabber specific for FreeSurfer data
fssource = Node(FreeSurferSource(subjects_dir=freesurfer_dir, hemi = 'both', subject_id=subject_id),
                run_without_submitting=True,
                name='fssource')

# Get B0 again after eddy and create mask
maskdwi = Node(Function(input_names=['in_file'],
                      output_names=['b0_file', 'mask_file'],
                      function=mask_img), name = 'maskdwi')

# Bias correction, first of B0 and then apply to the whole DWI series
biasB0 = Node(N4BiasFieldCorrection(num_threads=4, save_bias=True, dimension=3, output_image=subject_id + '_B0_n4.nii.gz'), name = 'biasB0')

# Apply bias correction to the whole DWI series
bias = Node(ImageMaths(op_string='-div', out_file=subject_id + '_dwi_n4.nii.gz'), name = 'bias')

# Get the minimum and maximum values within the B0 for T1 inversion
minmaxB0 = Node(ImageStats(op_string='-k %s -R'), name = 'minmaxB0')

# Get the minimum and maximum values within the T1
minmaxT1 = Node(ImageStats(op_string='-k %s -R'), name = 'minmaxT1')

# Calculate the ratio of min-max for the B0/T1
getratio = Node(Function(input_files=['statB0', 'statFS'],
                        output_files=['out'],
                        function=calcratio), name = 'getratio')

# Take the max T1 value and reverse the sign on the bias-corrected T1
getformula = Node(Function(input_files=['statFS', 'ratio'],
                             output_files=['out'],
                             function=calcT1), name = 'getformula')

# Apply the formula to the T1 image
invertT1 = Node(ImageMaths(out_file=subject_id + '_invertT1.nii.gz'), name = 'invertT1')

# Register extracted B0 to inverted T1 and apply inverse warp to T1
reg_rigid = Node(Registration(), name = 'reg_rigid')
reg_rigid.inputs.num_threads=16
reg_rigid.inputs.dimension=3
reg_rigid.inputs.interpolation = 'Linear'
reg_rigid.inputs.winsorize_lower_quantile=0.005
reg_rigid.inputs.winsorize_upper_quantile=0.995
reg_rigid.inputs.use_histogram_matching=False
reg_rigid.inputs.initial_moving_transform_com=1
reg_rigid.inputs.transforms=['Rigid']
reg_rigid.inputs.metric=['MI']
reg_rigid.inputs.metric_weight=[1.0]
reg_rigid.inputs.transform_parameters=[(0.1,)]
reg_rigid.inputs.radius_or_number_of_bins=[32]
reg_rigid.inputs.sampling_percentage=[0.25]
reg_rigid.inputs.sampling_strategy=['Regular']
reg_rigid.inputs.number_of_iterations=[[1000, 500, 250, 100]]
reg_rigid.inputs.convergence_threshold=[1e-6]
reg_rigid.inputs.convergence_window_size=[10]
reg_rigid.inputs.shrink_factors=[[8,4,2,1]]
reg_rigid.inputs.smoothing_sigmas=[[3,2,1,0]]
reg_rigid.inputs.output_warped_image=True
reg_rigid.write_composite_transform = False
reg_rigid.inputs.output_inverse_warped_image = True
reg_rigid.inputs.output_inverse_warped_image = subject_id + 'invT1_to_DWI.nii.gz'

# Warp the WM to DWI space as well
warpWMrigid = Node(ApplyTransforms(invert_transform_flags=True), name = 'warpWMrigid')
warpWMrigid.inputs.output_image = subject_id + '_WM_to_DWI.nii.gz'
warpWMrigid.inputs.interpolation = 'NearestNeighbor'

# Warp aparc+aseg.mgz as well
warpFS = Node(ApplyTransforms(invert_transform_flags=True), name = 'warpFS')
warpFS.inputs.output_image = subject_id + '_aparc+aseg_to_DWI.nii.gz'
warpFS.inputs.interpolation = 'NearestNeighbor'

# Get the non-linear deformation along the phase-encoding direction
synreg = Node(Registration(), name = 'synreg')
synreg.inputs.num_threads=8
synreg.inputs.dimension=3
synreg.inputs.interpolation = 'Linear'
synreg.inputs.winsorize_lower_quantile=0.005
synreg.inputs.winsorize_upper_quantile=0.995
synreg.inputs.use_histogram_matching=False
synreg.inputs.initial_moving_transform_com=0 # Align the images using the geometric center of the images since threy're already aligned
synreg.inputs.transforms=['SyN']
synreg.inputs.metric=['CC']
synreg.inputs.metric_weight=[1.0]
synreg.inputs.transform_parameters=[(0.1, 3.0, 0.0)]
synreg.inputs.radius_or_number_of_bins=[4]
synreg.inputs.sampling_percentage=[None]
synreg.inputs.sampling_strategy=[None]
synreg.inputs.number_of_iterations=[[100, 70, 50, 20]]
synreg.inputs.convergence_threshold=[1e-6]
synreg.inputs.convergence_window_size=[10]
synreg.inputs.shrink_factors=[[8,4,2,1]]
synreg.inputs.smoothing_sigmas=[[3,2,1,0]]
synreg.inputs.restrict_deformation=[[0,1,0]] # restrict along y-plane
synreg.inputs.output_warped_image= subject_id + '_B0_synwarp.nii.gz'
synreg.inputs.output_inverse_warped_image = subject_id + '_invT1_synwarp_rev.nii.gz'
synreg.inputs.write_composite_transform = False
synreg.inputs.output_inverse_warped_image = True

# Apply non-linear warp to correct for EPI distortions
synwarp = Node(ApplyTransforms(), name = 'synwarp')
synwarp.inputs.input_image_type=3
synwarp.inputs.dimension=3
synwarp.inputs.output_image = subject_id + '_dwi_synwarp.nii.gz'
synwarp.inputs.interpolation = 'Linear'
synwarp.inputs.num_threads=8

# Grab the 3D warp that was restricted to the y plane from the ANTs warp file (which is now a 5D file)
grabwarp = Node(Function(input_names=['warp'], output_names='yWarp', function=extract_warp), name = 'grabwarp')

# Create edge maps for further QA
dwi_edge = Node(ImageMaths(op_string='-bin -edge -bin', out_file=subject_id + '_dwi_edge_orig.nii.gz'), name = 'dwi_edge')
dwi_edge_syn = Node(ImageMaths(op_string='-bin -edge -bin', out_file=subject_id + '_dwi_edge_syn.nii.gz'), name = 'dwi_edge_syn')
wm_edge = Node(ImageMaths(op_string='-bin -edge -bin', out_file=subject_id + '_wm_rigid_edge.nii.gz'), name = 'wm_edge')
t1_edge = Node(ImageMaths(op_string='-bin -edge -bin', out_file=subject_id + '_t1_rigid_edge.nii.gz'), name = 't1_edge')

# Create output folder for important outputs
datasink = Node(DataSink(base_directory=preproc_dir,
                         container=subject_id,
                         substitutions=[
                         ('transform_InverseWarped', subject_id + '_invT1_synwarp'),
                         ('b0_bias', subject_id + '_bias_field'),
                         ('dwi_mask', subject_id + '_dwi_mask'),
                         ('transform1Warp_Y', subject_id + '_SyNwarp_Y')]),
                name="datasink")

# Set up workflow
preproc3_wf = Workflow(name = 'preproc3_wf', base_dir=scratch_dir)
preproc3_wf.connect([
    # grab B0 and create a new mask
    (getdata, maskdwi, [('dwi', 'in_file')]),
    # bias correction for B0 and whole dwi series
    (maskdwi, biasB0, [('b0_file', 'input_image'),
                      ('mask_file', 'mask_image')]),
    (getdata, bias, [('dwi', 'in_file')]),
    (biasB0, bias, [('bias_image', 'in_file2')]),
    (maskdwi, bias, [('mask_file', 'mask_file')]),
    # produce inverted T1
    (biasB0, minmaxB0, [('output_image', 'in_file')]),
    (maskdwi, minmaxB0, [('mask_file', 'mask_file')]),
    (getdata, minmaxT1, [('anat', 'in_file'),
                         ('mask', 'mask_file')]),
    (minmaxT1, getratio, [('out_stat', 'statT1')]),
    (minmaxB0, getratio, [('out_stat', 'statB0')]),
    (minmaxT1, getformula, [('out_stat', 'statT1')]),
    (getratio, getformula, [('out', 'ratio')]),
    (getformula, invertT1, [('out', 'op_string')]),
    (getdata, invertT1, [('anat', 'in_file'),
                          ('mask', 'mask_file')]),
    # register B0 to (inverted T1) and apply inverse warp to T1
    (invertT1, reg_rigid, [('out_file', 'fixed_image')]),
    (biasB0, reg_rigid, [('output_image', 'moving_image')]),
    # apply warp to WM segmentation as well
    (getdata, warpWMrigid, [('wm', 'input_image')]),
    (reg_rigid, warpWMrigid, [('reverse_transforms', 'transforms')]),
    (biasB0, warpWMrigid, [('output_image', 'reference_image')]),
    # also apply to freesurfer parcellations
    (fssource, warpFS, [(('aparc_aseg', get_aparc_aseg), 'input_image')]),
    (reg_rigid, warpFS, [('reverse_transforms', 'transforms')]),
    (biasB0, warpFS, [('output_image', 'reference_image')]),
    # non-linear registration of B0 to rigidly aligned inverted T1
    (biasB0, synreg, [('output_image', 'moving_image')]),
    (reg_rigid, synreg, [('inverse_warped_image', 'fixed_image')]),
    # apply warp to dwi series
    (synreg, grabwarp, [(('forward_transforms', get_warp), 'warp')]),
    (synreg, synwarp, [('forward_transforms', 'transforms')]),
    (bias, synwarp, [('out_file','input_image')]),
    (reg_rigid, synwarp, [('inverse_warped_image', 'reference_image')]),
    # produce edge maps for QC
    (biasB0, dwi_edge, [('output_image', 'in_file'),
                        ('output_image', 'mask_file')]),
    (synreg, dwi_edge_syn, [('warped_image', 'in_file'),
                            ('warped_image', 'mask_file')]),
    (warpWMrigid, wm_edge, [('output_image', 'in_file'),
                            ('output_image', 'mask_file')]),
    (reg_rigid, t1_edge, [('inverse_warped_image', 'in_file'),
                          ('inverse_warped_image', 'mask_file')]),
    # store relevant outputs
    (maskdwi, datasink, [('mask_file', 'preproc3_distortion.native.@mask')]),
    (biasB0, datasink, [('output_image', 'preproc3_distortion.native.@biasB0'),
                        ('bias_image', 'preproc3_distortion.native.@bias_field')]),
    (bias, datasink, [('out_file', 'preproc3_distortion.native.@bias_dwi')]),
    (invertT1, datasink, [('out_file', 'preproc3_distortion.native.@invT1')]),
    (dwi_edge, datasink, [('out_file', 'preproc3_distortion.native.@dwi_edge')]),
    (reg_rigid, datasink, [('inverse_warped_image', 'preproc3_distortion.@invT1')]),
    (reg_rigid, datasink, [('forward_transforms', 'preproc3_distortion.@t1warp')]),
    (synreg, datasink, [('warped_image', 'preproc3_distortion.@normB0')]),
    (synreg, datasink, [('inverse_warped_image', 'preproc3_distortion.@synT1')]),
    (synwarp, datasink, [('output_image', 'preproc3_distortion.@normdwi')]),
    (grabwarp, datasink, [('yWarp', 'preproc3_distortion.@warp')]),
    (warpWMrigid, datasink, [('output_image', 'preproc3_distortion.@WM')]),
    (warpFS, datasink, [('output_image', 'preproc3_distortion.@aparc')]),
    (wm_edge, datasink, [('out_file', 'preproc3_distortion.@WMedge')]),
    (t1_edge, datasink, [('out_file', 'preproc3_distortion.@t1_edge')]),
    (dwi_edge_syn, datasink, [('out_file', 'preproc3_distortion.@dwi_edge_syn')]),
])

preproc3_wf.run()
