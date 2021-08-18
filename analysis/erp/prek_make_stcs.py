#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maggie Clarke, Daniel McCloy

Make original and morphed Source Time Course files
"""

import os
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator
from analysis.aux_functions import (load_paths, load_params, yamload,
                                    load_inverse_params, PREPROCESS_JOINTLY)

# load params
*_, subjects, cohort = load_params(experiment='erp')

# inverse params
inverse_params = load_inverse_params()
method = inverse_params['method']
constr = inverse_params['orientation_constraint']
constr = f'-{constr}' if constr in ('free', 'fixed') else ''  # '' == loose
ori = inverse_params['estimate_type']
ori = ori if ori in ('vector', 'normal') else None  # None == 'magnitude'

# config paths
data_root, subjects_dir, _ = load_paths()
subfolder = 'combined' if PREPROCESS_JOINTLY else 'erp'

# config other
conditions = ['words', 'faces', 'cars', 'aliens']
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10
paramfile = os.path.join('..', 'preprocessing', 'mnefun_common_params.yaml')
with open(paramfile, 'r') as f:
    params = yamload(f)
lp_cut = params['preprocessing']['filtering']['lp_cut']
del params

# for morph to fsaverage
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]

# loop over subjects
for s in subjects:
    already_morphed = False
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        print(f'processing {s} {prepost}_camp')
        # paths for this subject / timepoint
        this_subj = os.path.join(data_root,
                                 f'{prepost}_camp', 'twa_hp', subfolder, s)
        inv_path = os.path.join(this_subj, 'inverse',
                                f'{s}-{lp_cut}-sss-meg{constr}-inv.fif')
        evk_path = os.path.join(this_subj, 'inverse',
                                f'Conditions_{lp_cut}-sss_eq_{s}-ave.fif')
        stc_path = os.path.join(this_subj, 'stc')
        # prepare output dir
        if not os.path.isdir(stc_path):
            os.mkdir(stc_path)
        # load evoked data and inverse
        inv = read_inverse_operator(inv_path)
        evokeds = [mne.read_evokeds(evk_path, condition=c) for c in conditions]
        # make STCs
        stcs = [apply_inverse(evk, inv, lambda2, method=method,
                              pick_ori=ori) for evk in evokeds]
        # save STCs
        for idx, stc in enumerate(stcs):
            out_fname = (f'{s}_{prepost}Camp_{method}_'
                         f'{evokeds[idx].comment}')
            stc.save(os.path.join(stc_path, out_fname))
        # morph to fsaverage. Doesn't recalculate for `post_camp` since
        # anatomy hasn't changed; uses the most recent STC from the above
        # saving loop (morph only needs the anatomy, not the MEG data)
        if not already_morphed:
            morph = mne.compute_source_morph(stc, subject_from=s.upper(),
                                             subject_to='fsaverage',
                                             subjects_dir=subjects_dir,
                                             spacing=fsaverage_vertices,
                                             smooth=smoothing_steps)
            already_morphed = True
        morphed_stcs = [morph.apply(stc) for stc in stcs]
        # save morphed STCs
        for idx, stc in enumerate(morphed_stcs):
            out_fname = (f'{s}FSAverage_{prepost}Camp_{method}_'
                         f'{evokeds[idx].comment}')
            stc.save(os.path.join(stc_path, out_fname))
