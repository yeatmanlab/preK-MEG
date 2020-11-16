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
import numpy as np
import mnefun
from prek_score import prek_score
from analysis.aux_functions import load_paths, load_params

lp_cut = 30
pre_or_post = 'pre'  # str: 'pre' or 'post' convenience variable for rerunning

# load subjects
*_, subjects, cohort = load_params()
print(subjects)

if cohort == 'replication':
    target_dir = '/mnt/scratch/prek/r_cohort/%s_camp/twa_hp/erp/' % pre_or_post
else:
    target_dir = '/mnt/scratch/prek/%s_camp/twa_hp/erp/' % pre_or_post

params = mnefun.Params(tmin=-0.1, tmax=1, t_adjust=-0.067, n_jobs=8,
                       proj_sfreq=200, n_jobs_fir='cuda',
                       filter_length='5s', lp_cut=lp_cut,
                       n_jobs_resample='cuda',
                       bmin=-0.1, bem_type='5120')
# load paths
_, subjects_dir, _ = load_paths()

structurals = [x.upper() for x in subjects]
params.subjects = subjects
params.work_dir = target_dir
params.subjects_dir = subjects_dir
params.score = prek_score
params.structurals = structurals
params.dates = [(2013, 0, 00)] * len(params.subjects)
# define which subjects to run
params.subject_indices = np.arange(len(params.subjects))
# Aquistion params
params.acq_ssh = 'nordme@kasga.ilabs.uw.edu'
params.acq_dir = '/brainstudio/prek/'
params.sws_ssh = 'nordme@kasga.ilabs.uw.edu'
params.sws_dir = '/data07/nordme/prek/'
# SSS options
params.sss_type = 'python'
params.sss_regularize = 'in'
params.tsss_dur = 4.  # tSSS duration
params.int_order = 8
params.st_correlation = 0.98
params.trans_to = 'twa'  # "twa" (time-weighted avg) or "fixed" head pos; use "fixed" for within-subj sensor-level analyses
params.coil_t_window = 'auto'
params.movecomp = 'inter'
params.hp_type = 'python'
params.mf_autobad = True
params.mf_autobad_type = 'python'
# remove segments with < 3 good coils for at least 100 ms
params.coil_bad_count_duration_limit = 0.1
# epoch rejection params
params.reject = dict()
params.auto_bad_reject = None
params.ssp_ecg_reject = None
params.flat = dict(grad=1e-13, mag=1e-15)
params.auto_bad_flat = None
params.auto_bad_meg_thresh = 10
# SSP params
params.ssp_ecg_reject = dict(grad=1500e-13, mag=4500e-15)
params.ecg_channel = 'ECG063'
params.ssp_eog_reject = dict(grad=2000e-13, mag=6000e-15)
params.veog_channel = 'EOG062'
params.veog_f_lims = (0.5, 2)
params.veog_t_lims = (-0.22, 0.22)
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [0, 0, 0],  # EOG  (combined saccade and blink events)
                    [0, 0, 0],  # Continuous (from ERM)
                    [0, 0, 0],  # HEOG (focus on saccades)
                    [1, 1, 0]]  # VEOG  (focus on blinks)
# naming
params.run_names = ['%s_erp_' + pre_or_post]
params.subject_run_indices = None
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = np.arange(1)
params.runs_empty = ['%s_erm']
# covariance
params.cov_method = 'shrunk'
params.bem_type = '5120'
params.compute_rank = True
params.cov_rank = None
params.force_erm_cov_rank_full = False
# Epoching
params.reject_epochs_by_annot = False   # new param due to EOG annots
params.in_names = ['words', 'faces', 'cars', 'aliens']
params.in_numbers = [10, 20, 30, 40]
params.analyses = ['All',
                   'Conditions']
params.out_names = [['All'],
                    ['words', 'faces', 'cars', 'aliens']]
params.out_numbers = [[10, 10, 10, 10],  # Combine all trials
                      [10, 20, 30, 40],  # Separate trials
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
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='faces',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='cars',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
        dict(analysis='Conditions', name='aliens',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif',
             views=['lat', 'caudal'], size=(800, 800)),
    ],
    snr=[
        dict(analysis='Conditions', name='words',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='faces',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='cars',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif'),
        dict(analysis='Conditions', name='aliens',
             inv='%s-' + str(lp_cut) + '-sss-meg-free-inv.fif')
    ],
    whitening=[
        dict(analysis='Conditions', name='words',
             cov='%s-' + str(lp_cut) + '-sss-cov.fif'),
        dict(analysis='Conditions', name='faces',
             cov='%s-' + str(lp_cut) + '-sss-cov.fif'),
        dict(analysis='Conditions', name='cars',
             cov='%s-' + str(lp_cut) + '-sss-cov.fif'),
        dict(analysis='Conditions', name='aliens',
             cov='%s-' + str(lp_cut) + '-sss-cov.fif')
    ],
    psd=False,
)

mnefun.do_processing(
    params,
    fetch_raw=False,
    do_sss=True,        # do tSSS
    do_score=True,      # do scoring
    gen_ssp=True,        # generate ssps
    apply_ssp=True,      # apply ssps
    write_epochs=True,   # epoching & filtering
    gen_covs=True,       # make covariance
    gen_fwd=True,        # generate fwd model
    gen_inv=True,        # general inverse
    gen_report=True,     # print report
    print_status=True    # show status
)
