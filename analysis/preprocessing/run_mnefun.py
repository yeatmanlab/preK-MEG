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
for paradigm in ('erp', 'pskt'):
    # load mnefun params from YAML
    params = mnefun.read_params('mnefun_common_params.yaml')
    # load more params from paradigm-specific YAML
    extra_params = _flat_params_read(f'mnefun_{paradigm}_params.yaml')
    vars(params).update(extra_params)
    del extra_params
    # set additional params: general
    params.subjects_dir = subjects_dir
    params.score = prek_score
    params.subjects = subjects
    # XXX XXX subj_1936 has impossibly noisy data in PSKT runs XXX XXX
    if paradigm == 'pskt':
        params.subjects.remove('prek_1936')
    params.subject_indices = list(range(len(params.subjects)))
    params.structurals = [s.upper() for s in params.subjects]
    params.dates = [(2013, 0, 00)] * len(params.subjects)

    # loop over pre/post intervention recordings, and over head pos transforms
    for headpos in ('twa', 'fixed'):
        for prepost in ('pre', 'post'):
            # skip unwanted combination (no sensor-space analysis of ERP runs)
            if paradigm == 'erp' and headpos == 'fixed':
                continue

# loop over pre/post intervention recordings, and over head pos transforms
for prepost in ('pre', 'post'):
    for headpos in ('fixed',):  # 'twa', 'fixed'
        # set variable-contingent params
        params.work_dir = os.path.join(data_root, f'{prepost}_camp',
                                       f'{headpos}_hp', 'combined')
        params.trans_to = (0., 0., 0.04) if headpos == 'fixed' else headpos
        # run filenames
        erp_run_names = [f'%s_erp_{prepost}']
        pskt_run_names = [f'%s_pskt_{run:02}_{prepost}' for run in (1, 2)]
        params.run_names = erp_run_names + pskt_run_names

        # set additional params: report
        common_kw = dict(analysis='Conditions')
        sns_kw = dict(times='peaks', **common_kw)
        snr_kw = dict(inv=f'%s-{params.lp_cut}-sss-meg-inv.fif', **common_kw)
        src_kw = dict(views=['lateral', 'ventral'], size=(800, 800), **snr_kw)
        wht_kw = dict(cov=f'%s-{params.lp_cut}-sss-cov.fif', **common_kw)
        conds = params.in_names  # experimental conditions
        params.report['snr'] = [dict(name=c, **snr_kw) for c in conds]
        params.report['sensor'] = [dict(name=c, **sns_kw) for c in conds]
        params.report['source'] = [dict(name=c, **src_kw) for c in conds]
        params.report['whitening'] = [dict(name=c, **wht_kw) for c in conds]

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
            is_erp = paradigm == 'erp'

            mnefun.do_processing(
                params,
                fetch_raw=False,      # go get the Raw files
                do_sss=True,          # tSSS / maxwell filtering
                do_score=is_erp,      # run scoring function to extract events
                gen_ssp=True,         # create SSP projectors
                apply_ssp=True,       # apply SSP projectors
                write_epochs=is_erp,  # epoching & filtering
                gen_covs=is_erp,      # make covariance
                gen_fwd=is_erp,       # generate fwd model
                gen_inv=is_erp,       # generate inverse
                gen_report=is_erp,    # print report
                print_status=True     # show status
            )
