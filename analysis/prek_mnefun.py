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
import mnefun
import numpy as np
from score import score

params = mnefun.Params(tmin=-0.1, tmax=1, n_jobs=18,
                       proj_sfreq=200, n_jobs_fir='cuda',
                       filter_length='5s', lp_cut=80., 
                       n_jobs_resample='cuda',
                       bmin=-0.1, bem_type='5120')
#1451 rename
#1505 no erp file
params.subjects = ['prek_1112', 'prek_1208', 'prek_1271', 'prek_1382', 
                   'prek_1673', 'prek_1676', 'prek_1691', 'prek_1715',
                   'prek_1762', 'prek_1887', 'prek_1901', 'prek_1916',
                   'prek_1921', 'prek_1936', 'prek_1951', 'prek_1964', 
                   'prek_1184', 'prek_1103', 'prek_1505', 'prek_1113',
                   'prek_1868', 'prek_1302', 'prek_1210', 'prek_1714',
                   'prek_1401', 'prek_1706', 'prek_1878', 'prek_1818',
                   'prek_1490', 'prek_1768', 'prek_1939', 'prek_1293',
                   'prek_1751', 'prek_1869', 'prek_1443', 'prek_1372']

params.score = score
params.structurals = params.subjects
params.dates = [(2013, 0, 00)] * len(params.subjects)
# define which subjects to run
params.subject_indices = [0]
#params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)), [0])
# Aquistion params 
params.acq_ssh = 'maggie@kasga.ilabs.uw.edu'
params.acq_dir = '/brainstudio/prek/'
params.sws_ssh = 'maggie@kasga.ilabs.uw.edu'
params.sws_dir = '/data07/maggie/prek/'
# SSS options
params.sss_type = 'python'
params.sss_regularize = 'in'
params.tsss_dur = 4. # tSSS duration
params.int_order = 8
params.st_correlation = .98
params.trans_to='twa' # time weighted average head position (change this to fixed pos for group analysis)
params.coil_t_window = 'auto'
params.movecomp='inter'
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
params.run_names = ['%s_erp_pre']
params.get_projs_from = np.arange(1)
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = []
# proj
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [1, 1, 0],  # EOG
                    [0, 0, 0]]  # Continuous (from ERM)
params.cov_method = 'empirical'
params.bem_type = '5120'
params.compute_rank = True
# Epoching
params.in_names = ['words', 'faces', 'cars', 'aliens']
params.in_numbers = [10, 20, 30, 40]
params.analyses = ['All',
                   'Conditions']
params.out_names = [['All'],
                    ['words', 'faces', 'cars', 'aliens']]
params.out_numbers = [[1, 1, 1, 1],  # Combine all trials
                      [1, 2, 3, 4],  # Seperate trials
    ]
params.must_match = [
    [], # trials to match
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
    psd=False,
)

mnefun.do_processing(
    params,
    fetch_raw=False,
    do_sss=False, # do tSSS
    do_score=True,  # do scoring
    gen_ssp=False, # generate ssps
    apply_ssp=False, # apply ssps
    write_epochs=False, # epoching & filtering
    gen_covs=False, # make covariance 
    gen_fwd=False, # generate fwd model
    gen_inv=False, # general inverse
    gen_report=False, # print report
    print_status=True # show status
)
