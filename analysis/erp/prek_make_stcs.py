#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maggie Clarke, Daniel McCloy

Make original and morphed Source Time Course files
"""

import os
import yaml
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator

mne.set_log_level('WARNING')

# config paths
subjects_dir = '/mnt/scratch/prek/anat'
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
# config other
conditions = ['words', 'faces', 'cars', 'aliens']
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10

# load subjects
with open(os.path.join('..', '..', 'params', 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f, Loader=yaml.FullLoader)

# for morph to fsaverage
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]

# loop over pre/post measurement time
for prepost in ('pre', 'post'):
    subj_root = '/mnt/scratch/prek/{prepost}_camp/twa_hp'
    # loop over subjects
    for s in subjects:
        print(f'processing {s}')
        # paths for this subject
        this_subj = os.path.join(subj_root, s)
        inv_path = os.path.join(this_subj, 'inverse', f'{s}-80-sss-meg-inv.fif')
        evk_path = os.path.join(this_subj, 'inverse', f'Conditions_80-sss_eq_{s}-ave.fif')  # noqa
        stc_path = os.path.join(this_subj, 'stc')
        # prepare output dir
        if not os.path.isdir(stc_path):
            os.mkdir(stc_path)
        # load evoked data and inverse
        inv = read_inverse_operator(inv_path)
        evokeds = [mne.read_evokeds(evk_path, condition=c) for c in conditions]
        # make STCs with each method
        for method in methods:
            stcs = [apply_inverse(evk, inv, lambda2, method=method,
                                  pick_ori=None) for evk in evokeds]
            # save STCs
            for idx, stc in enumerate(stcs):
                out_fname = f'{s}_{method}_{evokeds[idx].comment}'
                stc.save(os.path.join(stc_path, out_fname))
            # morph to fsaverage. uses the most recent STC from the above loop
            # (morph should end up the same for all stcs for one subject)
            morph = mne.compute_source_morph(stc, subject_from=s.upper(),
                                             subject_to='fsaverage',
                                             subjects_dir=subjects_dir,
                                             spacing=fsaverage_vertices,
                                             smooth=smoothing_steps)
            morphed_stcs = [morph.apply(stc) for stc in stcs]
            # save morphed STCs
            for idx, stc in enumerate(morphed_stcs):
                out_fname = f'{s}_{method}_fsaverage_{evokeds[idx].comment}'
                stc.save(os.path.join(stc_path, out_fname))
