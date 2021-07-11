#!/usr/bin/env python
"""
mnefun preprocessing of both PRE-K cohorts (original & replication).
"""

import os
import mne
import mnefun
from analysis.aux_functions import load_paths, load_params
from prek_score import prek_score

# mne.viz.set_3d_backend('mayavi')

# load general params
data_root, subjects_dir, _ = load_paths()
*_, subjects, cohort = load_params()

# load mnefun params from YAML
params = mnefun.read_params('mnefun_common_params.yaml')
# set additional params: general
params.subjects_dir = subjects_dir
params.score = prek_score
params.subjects = subjects
params.subject_indices = list(range(len(params.subjects)))
params.structurals = [s.upper() for s in params.subjects]
params.dates = [(2013, 0, 00)] * len(params.subjects)

# loop over pre/post intervention recordings, and over head pos transforms
for prepost in ('pre', 'post'):
    for headpos in ('twa', 'fixed'):
        for experiment in ('pskt',):  # XXX ('pskt', 'erp') or ('combined',)
            # set variable-contingent params
            params.work_dir = os.path.join(data_root, f'{prepost}_camp',
                                           f'{headpos}_hp', experiment)
            params.trans_to = (0., 0., 0.04) if headpos == 'fixed' else headpos
            # run filenames
            erp_run_names = [f'%s_erp_{prepost}']
            pskt_run_names = [f'%s_pskt_{run:02}_{prepost}' for run in (1, 2)]
            if experiment == 'combined':
                params.run_names = (erp_run_names + pskt_run_names)
            elif experiment == 'pskt':
                params.run_names = pskt_run_names
            else:
                params.run_names = erp_run_names

            # set additional params: report
            common_kw = dict(analysis='Conditions')
            sns_kw = dict(times='peaks', **common_kw)
            snr_kw = dict(inv=f'%s-{params.lp_cut}-sss-meg-inv.fif',
                          **common_kw)
            src_kw = dict(views=['lateral', 'ventral'], size=(800, 800),
                          **snr_kw)
            wht_kw = dict(cov=f'%s-{params.lp_cut}-sss-cov.fif', **common_kw)
            conds = params.in_names  # experimental conditions
            params.report['snr'] = [dict(name=c, **snr_kw) for c in conds]
            params.report['sensor'] = [dict(name=c, **sns_kw) for c in conds]
            params.report['source'] = [dict(name=c, **src_kw) for c in conds]
            params.report['whitening'] = [dict(name=c, **wht_kw)
                                          for c in conds]

            # run it
            mnefun.do_processing(
                params,
                fetch_raw=False,      # go get the Raw files
                do_sss=True,          # tSSS / maxwell filtering
                do_score=True,        # run scoring function to extract events
                gen_ssp=True,         # create SSP projectors
                apply_ssp=True,       # apply SSP projectors
                write_epochs=True,    # epoching & filtering
                gen_covs=True,        # make covariance
                gen_fwd=True,         # generate fwd model
                gen_inv=True,         # generate inverse
                gen_report=True,      # print report
                print_status=True     # show status
            )
