# -*- coding: utf-8 -*-

"""
Created on Fri May 17 7:11:32 2019

@author: mdclarke

mnefun processing script for PreK Project

Notes:
1) run preprocessing (up to gen_covs)
2) run prek_setup_source.py
3) coregistration (mne coreg)
4) run fwd + inv (this script)

1 = words (N=30)
2 = faces (N=30)
3 = cars (N=30)
4 = aliens (N=10) + 10 button responses (64)

"""
import os
import yaml
import numpy as np
import mnefun
from prek_score import prek_score

pre_or_post = 'post' # convenience variable for rerunning 
target_dir = '/mnt/scratch/prek/%s_camp/twa_hp/erp/' % pre_or_post

params = mnefun.Params(tmin=-0.1, tmax=1, t_adjust=-0.067, n_jobs='cuda',
                       proj_sfreq=200, n_jobs_fir='cuda',
                       filter_length='5s', lp_cut=80.,
                       n_jobs_resample='cuda',
                       bmin=-0.1, bem_type='5120')
# load subjects
with open(os.path.join('..', '..', 'params', 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f, Loader=yaml.FullLoader)

print(subjects)
structurals = [x.upper() for x in subjects]

params.subjects = subjects
params.work_dir = target_dir
params.subjects_dir = '/mnt/scratch/prek/anat'
params.score = prek_score
params.structurals = structurals
params.dates = [(2013, 0, 00)] * len(params.subjects)
# define which subjects to run
# params.subject_indices = [0]
params.subject_indices = np.arange(len(params.subjects))
# params.subject_indices = np.arange(0,46)
# params.subject_indices = [47]
# Aquistion params
params.acq_ssh = 'nordme@kasga.ilabs.uw.edu'
params.acq_dir = '/brainstudio/prek/'
params.sws_ssh = 'nordme@kasga.ilabs.uw.edu'
params.sws_dir = '/data07/nordme/prek/'
# SSS options
params.sss_type = 'python'
params.sss_regularize = 'in'
params.tsss_dur = 4. # tSSS duration
params.int_order = 8
params.st_correlation = .98
params.trans_to = 'twa'  # time-weighted avg (use fixed head pos for sensor-level analyses)
params.coil_t_window = 'auto'
params.movecomp = 'inter'
# remove segments with < 3 good coils for at least 100 ms
params.coil_bad_count_duration_limit = 0.1
# Trial rejection criteria
params.reject = dict()
params.auto_bad_reject = None
params.ssp_ecg_reject = None
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_flat = None
params.auto_bad_meg_thresh = 10
# naming
params.run_names = ['%s_erp_'+ pre_or_post]
params.subject_run_indices = None
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = np.arange(1)
params.runs_empty = ['%s_erm']
# proj
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 0],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)
params.cov_method = 'shrunk'
params.bem_type = '5120'
params.compute_rank = True
params.cov_rank = None
params.force_erm_cov_rank_full=False
# Epoching
params.reject_epochs_by_annot = False   # new param due to EOG annots
params.in_names = ['words', 'faces', 'cars', 'aliens']
params.in_numbers = [10, 20, 30, 40]
params.analyses = ['All',
                   'Conditions']
params.out_names = [['All'],
                    ['words', 'faces', 'cars', 'aliens']]
params.out_numbers = [[10, 10, 10, 10],  # Combine all trials
                      [10, 20, 30, 40],  # Seperate trials
                      ]
params.must_match = [[],  # trials to match
                     [],
                     ]

params.report_params.update(  # add plots
    bem=True,
    sensor=[
        dict(analysis='Conditions', name='words', times='peaks'),
        dict(analysis='Conditions', name='faces', times='peaks'),
        dict(analysis='Conditions', name='cars', times='peaks'),
        dict(analysis='Conditions', name='aliens', times='peaks'),
    ],
    source=[
        dict(analysis='Conditions', name='words',
             inv='%s-80-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='faces',
             inv='%s-80-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='cars',
             inv='%s-80-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='aliens',
             inv='%s-80-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
    ],
    snr=[
        dict(analysis='Conditions', name='words',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='faces',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='cars',
             inv='%s-80-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='aliens',
             inv='%s-80-sss-meg-free-inv.fif')
    ],
    whitening=[
        dict(analysis='Conditions', name='words',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='Conditions', name='faces',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='Conditions', name='cars',
             cov='%s-80-sss-cov.fif'),
        dict(analysis='Conditions', name='aliens',
             cov='%s-80-sss-cov.fif')
    ],
    psd=False,
)

mnefun.do_processing(
    params,
    fetch_raw=False,
    do_sss=True,        # do tSSS
    do_score=True,       # do scoring
    gen_ssp=True,        # generate ssps
    apply_ssp=True,      # apply ssps
    write_epochs=True,   # epoching & filtering
    gen_covs=True,       # make covariance
    gen_fwd=True,        # generate fwd model
    gen_inv=True,        # general inverse
    gen_report=True,    # print report
    print_status=True    # show status
)
