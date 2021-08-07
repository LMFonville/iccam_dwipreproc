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
from os.path import join as opj
#Add FSL
os.environ["FSLDIR"] = "/apps/fsl/6.0.1/fsl"
os.environ["PATH"] += ":/apps/fsl/6.0.1/fsl/bin"
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
#Add Freesurfer
os.environ["PATH"] += ":/apps/freesurfer/6.0.0/freesurfer/bin"
os.environ["FREESURFER_HOME"] = "/apps/freesurfer/6.0.0/freesurfer"
os.environ["SUBJECTS_DIR"] = "/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/freesurfer"
os.environ["MNI_DIR"] = "/apps/freesurfer/6.0.0/freesurfer/mni"
os.environ["MINC_BIN_DIR"] = "/apps/freesurfer/6.0.0/freesurfer/mni/bin"
os.environ["MINC_LIB_DIR"] = "/apps/freesurfer/6.0.0/freesurfer/mni/lib"
os.environ["MNI_DATAPATH"] = "/apps/freesurfer/6.0.0/freesurfer/mni/data"
os.environ["PERL5LIB"] = "/apps/freesurfer/6.0.0/freesurfer/mni/lib/perl5/5.8.5"
os.environ["MNI_PERL5LIB"] = "/apps/freesurfer/6.0.0/freesurfer/mni/lib/perl5/5.8.5"
os.environ["PATH"] += ":/apps/freesurfer/6.0.0/freesurfer/mni/bin"

from nipype import config
from nipype import config
cfg = dict(logging=dict(workflow_level = 'DEBUG'),
           execution={'stop_on_first_crash': True,
                      'local_hash_check': True,
                      'crashfile_format': 'txt',
                      'keep_inputs': True,
                      'hash_method': 'timestamp',
                     'job_finished_timeout': 15.0})
config.update_config(cfg)

#Load modules
from nipype.pipeline.engine import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.io import SelectFiles, FreeSurferSource
from nipype.interfaces.freesurfer import MRIConvert, FSCommand, Binarize
from nipype.interfaces.fsl import Reorient2Std
from nipype.interfaces.freesurfer.utils import ImageInfo
from nipype.interfaces.utility import Merge, Select
from nipype.interfaces.io import DataSink

# Specify path and iterables
experiment_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/'
scratch_dir = opj('/rds/general/project/adobe_addiction_and_obesity_cohort/ephemeral', subject_id)
freesurfer_dir = opj(experiment_dir, 'output/freesurfer/')
output_dir = '/rds/general/project/adobe_addiction_and_obesity_cohort/live/output/re_dwipreproc/'
seg_dir = opj(experiment_dir, 'output/segmentations/')

# Set environmental variable SUBJECTS_DIR
FSCommand.set_default_subjects_dir(freesurfer_dir)

# Data grabber specific for FreeSurfer data
fssource = Node(FreeSurferSource(subjects_dir=freesurfer_dir, hemi = 'both', subject_id=subject_id),
                run_without_submitting=True,
                name='fssource')

def get_aparc_aseg(files):
    for name in files:
        if 'aparc+aseg' in name:
            return name
    raise ValueError('aparc+aseg not found')

# Combine output into 1 list to use
merge = Node(Merge(4), name = 'merge')

# Make sure we have the right in orientation before reorienting the file
getorientation = MapNode(ImageInfo(), iterfield=['in_file'], name = 'getorientation')

# Convert mgz files to nifti
mriconvert = MapNode(MRIConvert(out_type = 'niigz'), iterfield=['in_file', 'in_orientation'], name = 'mriconvert')

# change orientation to match dwi data
reorientFS = MapNode(Reorient2Std(), iterfield=['in_file'], name = 'reorientFS')

# Select all but the aparc+aseg file to store
select = Node(Select(index=[0, 1, 3]), name='select')

# Create masks of whole brain and tissues
maskbrain = Node(Binarize(min=0, max=1, bin_val=0, bin_val_not=1, erode=1, dilate=1, binary_file=subject_id + '_BrainMask.mgh', out_type='mgz'), name = 'maskbrain')
binGM = Node(Binarize(args='--gm', binary_file=subject_id + '_GM.nii.gz', out_type = 'nii.gz'), name = 'binGM')
binWM = Node(Binarize(args='--all-wm', binary_file=subject_id + '_WM.nii.gz', out_type = 'nii.gz'), name = 'binWM')

# Creates output folder for important outputs
preprocsink = Node(DataSink(base_directory=output_dir,
                         container=subject_id),
                name="preprocsink")
substitutions=[
    ('_reorientFS*',''),
    ('_reoriented', ''),
    ('_out', ''),
    ('T1', subject_id + '_T1w'),
    ('brain', subject_id + '_Brain'),
]
substitutions += [("_reorientFS%d" % i, "") for i in range(4)]
preprocsink.inputs.substitutions = substitutions

segsink = Node(DataSink(base_directory=seg_dir),
                name="segsink")

preproc0_wf = Workflow(name = 'preproc0_wf', base_dir = scratch_dir)
preproc0_wf.connect([
    (fssource, merge, [('T1', 'in1'),
                       ('brain', 'in2'),
                       (('aparc_aseg', get_aparc_aseg), 'in3')]),
    (fssource, maskbrain, [('brain', 'in_file')]),
    (maskbrain, merge, [('binary_file', 'in4')]),
    (merge, getorientation, [('out', 'in_file')]),
    (merge, mriconvert, [('out', 'in_file')]),
    (getorientation, mriconvert, [('orientation', 'in_orientation')]),
    (mriconvert, reorientFS, [('out_file', 'in_file')]),
    (reorientFS, select, [('out_file', 'inlist')]),
    (reorientFS, binGM, [(('out_file', get_aparc_aseg), 'in_file')]),
    (reorientFS, binWM, [(('out_file', get_aparc_aseg), 'in_file')]),
    (select, preprocsink, [('out', 'preproc0_segment.@outfiles')]),
    (binGM, preprocsink, [('binary_file', 'preproc0_segment.@GM')]),
    (binWM, preprocsink, [('binary_file', 'preproc0_segment.@WM')]),
    #(binGM, segsink, [('binary_file', 'GM.@GM')]),
    #(binWM, segsink, [('binary_file', 'WM.@WM')]),
])

preproc0_wf.run()
