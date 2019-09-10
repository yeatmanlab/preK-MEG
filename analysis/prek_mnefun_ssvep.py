# -*- coding: utf-8 -*-

"""
Created on Fri May 17 7:11:32 2019

@author: mdclarke

mnefun processing script for PreK SSVEP

Notes:
1) run preprocessing (tSSS and SSP only)
2) no scoring / epoching
3) will still need twa and fixed hp

"""


# pre_camp exceptions to run_names:
# prek 1714: use pskt_02, pskt_03
# prek_1936: use pskt_01, pskt_03
# prek_1964: use pskt_01, pskt_03

import mnefun
import numpy as np


dir = '/home/nordme/data/prek/post_camp/fixed_hp/'
skip = ['prek_1259', 'prek_1451', 'prek_1714', 'prek_1936', 'prek_1964']
#subjects = [x for x in os.listdir(dir) if op.isdir(op.join(dir, x)) and 'prek' in x and not np.in1d(x, skip)]
subjects = ['prek_1936', 'prek_1964']
# subjects = ['prek_1714']

subjects.sort()
print(subjects)

params = mnefun.Params(tmin=-0.1, tmax=1, n_jobs=18,
                       proj_sfreq=200, n_jobs_fir=18,
                       filter_length='5s', lp_cut=80., 
                       n_jobs_resample=18,
                       bmin=-0.1, bem_type='5120', )
#1451 rename

params.subjects = subjects
params.work_dir = dir
params.structurals = params.subjects
params.dates = [(2013, 0, 00)] * len(params.subjects)
# define which subjects to run
params.subject_indices = np.arange(len(params.subjects))
# params.subject_indices = np.setdiff1d(np.arange(len(params.subjects)), np.arange(11))
# params.subject_indices = [7]
# Acquisition params
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
params.trans_to = (0., 0., 0.04) # time weighted average or fixed head position (change this to fixed pos for sensor analysis)
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
# params.run_names = ['%s_pskt_02_pre', '%s_pskt_03_pre']
params.run_names = ['%s_pskt_01_pre', '%s_pskt_03_pre']
# params.run_names = ['%s_pskt_01_pre', '%s_pskt_02_pre']
params.get_projs_from = np.arange(2)
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

# No epoching for ssvep

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
    do_sss=True, # do tSSS
    do_score=False,  # do scoring
    gen_ssp=True, # generate ssps
    apply_ssp=True, # apply ssps
    write_epochs=False, # epoching & filtering
    gen_covs=False, # make covariance 
    gen_fwd=False, # generate fwd model
    gen_inv=False, # general inverse
    gen_report=False, #print report
    print_status=True # show status
)
