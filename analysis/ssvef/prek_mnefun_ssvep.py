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

import numpy as np
import mnefun
from analysis.aux_functions import load_params, load_paths

print(mnefun)

fixed_or_twa = 'twa'  # convenience variable for doing different runs
pre_or_post = 'post'

if fixed_or_twa == 'fixed':
    trans_to = (0., 0., 0.04)
else:
    trans_to = 'twa'

_, subjects_dir, _ = load_paths()

# load subjects
*_, subjects, cohort = load_params()

if cohort == 'replication':
    pskt_dir = f'/mnt/scratch/prek/r_cohort/{pre_or_post}_camp/{fixed_or_twa}_hp/pskt/'  # noqa E501
else:
    pskt_dir = f'/mnt/scratch/prek/{pre_or_post}_camp/{fixed_or_twa}_hp/pskt/'

structurals = [x.upper() for x in subjects]

params = mnefun.Params(tmin=-0.1, tmax=1, n_jobs='cuda',
                       proj_sfreq=200, n_jobs_fir='cuda',
                       filter_length='5s', lp_cut=80.,
                       n_jobs_resample='cuda',
                       bmin=-0.1, bem_type='5120', )

params.subjects = subjects
params.work_dir = pskt_dir
params.structurals = structurals
params.subjects_dir = subjects_dir
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
params.tsss_dur = 4.  # tSSS duration
params.int_order = 8
params.st_correlation = .98
params.trans_to = trans_to  # "twa" (time-weighted avg) or "fixed" head pos; use "fixed" for within-subj sensor-level analyses
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
# SSP params
params.get_projs_from = np.arange(2)
params.ssp_ecg_reject = dict(grad=1500e-13, mag=4500e-15)
params.ecg_channel = 'ECG063'
params.ssp_eog_reject = dict(grad=2000e-13, mag=6000e-15)
params.veog_channel = 'EOG062'
params.veog_f_lims = (0.5, 2)
params.veog_t_lims = (-0.22, 0.22)
params.proj_nums = [[1, 1, 0],  # ECG: grad/mag/eeg
                    [0, 0, 0],  # EOG
                    [0, 0, 0],  # Continuous (from ERM)
                    [0, 0, 0],  # HEOG
                    [1, 1, 0]]  # VEOG
# naming
params.run_names = ['%s_pskt_01_' + pre_or_post, '%s_pskt_02_' + pre_or_post]
params.inv_names = ['%s']
params.inv_runs = [np.arange(1)]
params.runs_empty = []
# covariance
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
    do_sss=False,        # do tSSS
    do_score=False,      # do scoring
    gen_ssp=True,        # generate ssps
    apply_ssp=True,      # apply ssps
    write_epochs=False,  # epoching & filtering
    gen_covs=False,      # make covariance
    gen_fwd=False,       # generate fwd model
    gen_inv=False,       # general inverse
    gen_report=True,     # print report
    print_status=True    # show status
)
