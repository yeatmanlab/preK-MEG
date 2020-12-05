#!/usr/bin/env python
"""
mnefun preprocessing of both PRE-K cohorts (original & replication).
"""

import os
import mne
import mnefun
from analysis.aux_functions import load_paths, load_params
from prek_score import prek_score

mne.viz.set_3d_backend('mayavi')

# load general params
data_root, subjects_dir, _ = load_paths()
*_, subjects, cohort = load_params()

for paradigm in ('pskt', 'erp'):
    # load mnefun params from YAML
    params = mnefun.read_params('mnefun_common_params.yaml')

    # set additional params: general
    params.subjects_dir = subjects_dir
    params.subject_indices = list(range(len(subjects)))
    params.score = prek_score
    # set additional params: fetch_raw
    params.subjects = subjects
    params.structurals = [s.upper() for s in subjects]
    params.dates = [(2013, 0, 00)] * len(subjects)
    # load more params from paradigm-specific YAML
    extra_params = mnefun.read_params(f'mnefun_{paradigm}_params.yaml')
    vars(params).update(vars(extra_params))
    del extra_params

    # loop over pre/post intervention recordings, and over head pos transforms
    for prepost in ('pre', 'post'):
        for headpos in ('twa', 'fixed'):
            # skip unwanted combination (no sensor-space analysis of ERP runs)
            if paradigm == 'erp' and headpos == 'fixed':
                continue

            # set variable-contingent params
            params.work_dir = os.path.join(data_root, f'{prepost}_camp',
                                           f'{headpos}_hp', paradigm)

            params.trans_to = (0., 0., 0.04) if headpos == 'fixed' else headpos

            if paradigm == 'pskt':
                params.run_names = [f'%s_{paradigm}_{run:02}_{prepost}'
                                    for run in (1, 2)]
            else:
                params.run_names = [f'%s_{paradigm}_{prepost}']
                # set additional params: report
                common_kw = dict(analysis='Conditions')
                sns_kw = dict(times='peaks', **common_kw)
                snr_kw = dict(inv=f'%s-{params.lp_cut}-sss-meg-free-inv.fif', **common_kw)  # noqa E501
                src_kw = dict(views=['lateral', 'ventral'], size=(800, 800), **snr_kw)  # noqa E501
                wht_kw = dict(cov=f'%s-{params.lp_cut}-sss-cov.fif', **common_kw)  # noqa E501
                conds = params.in_names  # experimental conditions
                params.report['snr'] = [dict(name=c, **snr_kw) for c in conds]
                params.report['sensor'] = [dict(name=c, **sns_kw) for c in conds]  # noqa E501
                params.report['source'] = [dict(name=c, **src_kw) for c in conds]  # noqa E501
                params.report['whitening'] = [dict(name=c, **wht_kw) for c in conds]  # noqa E501

            # run it
            mnefun.do_processing(
                params,
                fetch_raw=False,     # go get the Raw files
                do_sss=False,        # tSSS / maxwell filtering
                do_score=False,      # run scoring function to extract events
                gen_ssp=False,       # create SSP projectors
                apply_ssp=False,     # apply SSP projectors
                write_epochs=False,  # epoching & filtering
                gen_covs=False,      # make covariance
                gen_fwd=False,       # generate fwd model
                gen_inv=False,       # generate inverse
                gen_report=False,    # print report
                print_status=True    # show status
            )
