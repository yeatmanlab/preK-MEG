#!/usr/bin/env python
"""
mnefun preprocessing of PSKT runs of the first PRE-K cohort.
"""

import os
import mnefun
from analysis.aux_functions import load_paths, load_params

# FLAGS
prepost = 'pre'  # 'pre' or 'post' intervention
headpos = 'twa'  # 'twa' (time-weighted average) or 'fixed'; use "fixed" for
#                   within-subj sensor-level analyses

# load general params
data_root, subjects_dir, _ = load_paths()
*_, subjects, cohort = load_params()

# load mnefun params from YAML
params = mnefun.read_params('mnefun_params.yaml')

# set additional params: general
params.work_dir = os.path.join(data_root, f'{prepost}_camp', f'{headpos}_hp',
                               'erp')
params.subjects_dir = subjects_dir
params.subject_indices = list(range(len(subjects)))

# set additional params: fetch_raw
params.subjects = subjects
params.structurals = [s.upper() for s in subjects]
params.dates = [(2013, 0, 00)] * len(subjects)
params.run_names = [f'%s_erp_{prepost}']
#                  [f'%s_pskt_{run:02}_{prepost}' for run in (1, 2)]

# set additional params: SSS
params.sss.trans_to = (0., 0., 0.04) if headpos == 'fixed' else headpos

# set additional params: report
common_kw = dict(analysis='Conditions')
sns_kw = dict(times='peaks', **common_kw)
snr_kw = dict(inv=f'%s-{params.lp_cut}-sss-meg-free-inv.fif', **common_kw)
src_kw = dict(views=['lat', 'caudal'], size=(800, 800), **snr_kw)
wht_kw = dict(cov=f'%s-{params.lp_cut}-sss-cov.fif', **common_kw)
conditions = ('words', 'faces', 'cars', 'aliens')
params.report['snr'] = [dict(name=cond, **snr_kw) for cond in conditions]
params.report['sensor'] = [dict(name=cond, **sns_kw) for cond in conditions]
params.report['source'] = [dict(name=cond, **src_kw) for cond in conditions]
params.report['whitening'] = [dict(name=cond, **wht_kw) for cond in conditions]
# for PSKT:
# analyses = ('pooled', 'separate', 'separate')
# conditions = ('pooled', 'ps', 'kt')
# views = ('lateral', 'ventral')
# inv = '%s-80-sss-meg-free-inv.fif'
# cov = '%s-80-sss-cov.fif'


# run it
mnefun.do_processing(
    params,
    fetch_raw=False,     # go get the Raw files
    do_sss=False,        # tSSS / maxwell filtering
    do_score=False,      # run scoring function to extract events
    gen_ssp=True,        # create SSP projectors
    apply_ssp=True,      # apply SSP projectors
    write_epochs=True,   # epoching & filtering
    gen_covs=True,       # make covariance
    gen_fwd=True,        # generate fwd model
    gen_inv=True,        # generate inverse
    gen_report=False,    # print report
    print_status=True    # show status
)
