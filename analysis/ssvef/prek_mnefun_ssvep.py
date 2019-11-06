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

import os
import os.path as op
import mnefun
import numpy as np

print(mnefun)

fixed_or_twa = 'twa' # convenience variable for doing different runs
pre_or_post = 'post'

if fixed_or_twa == 'fixed':
    trans_to = (0., 0., 0.04)
else:
    trans_to = 'twa'

pskt_dir = '/mnt/scratch/prek/%s_camp/%s_hp/pskt/' % (pre_or_post, fixed_or_twa)

# load subjects
with open(os.path.join('..', '..', 'params', 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f, Loader=yaml.FullLoader

structurals = ['PREK%s' %x[4:] for x in subjects]

params = mnefun.Params(tmin=-0.1, tmax=1, n_jobs=2,
                       proj_sfreq=200, n_jobs_fir=2,
                       filter_length='5s', lp_cut=80., 
                       n_jobs_resample=2,
                       bmin=-0.1, bem_type='5120', )

params.subjects = subjects
params.work_dir = pskt_dir
params.structurals = structurals
params.subjects_dir = '/mnt/scratch/prek/anat'
params.dates = [(2013, 0, 00)] * len(params.subjects)
# define which subjects to run
params.subject_indices = np.arange(len(params.subjects))
# params.subject_indices = np.arange(28,45)
# params.subject_indices = [47]
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
params.trans_to = trans_to # time weighted average or fixed head position (change this to fixed pos for sensor analysis)
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
params.run_names = ['%s_pskt_01_' + pre_or_post, '%s_pskt_02_' + pre_or_post]
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
params.cov_rank = None

# No epoching for ssvep
# No conditions for report

params.report_params.update(  # add plots
    bem=True, 
    psd=False
)   

mnefun.do_processing(
    params,
    fetch_raw=False,
    do_sss=False, # do tSSS
    do_score=False,  # do scoring
    gen_ssp=False, # generate ssps
    apply_ssp=False, # apply ssps
    write_epochs=False, # epoching & filtering
    gen_covs=False, # make covariance 
    gen_fwd=False, # generate fwd model
    gen_inv=False, # general inverse
    gen_report=True, #print report
    print_status=True # show status
)
