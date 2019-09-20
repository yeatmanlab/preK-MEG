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
subj_root = '/mnt/scratch/prek/pre_camp/twa_hp'
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

for s in subjects:
    print(f'processing {s}')
    # paths for this subject
    this_subj = os.path.join(subj_root, s)
    src_path = os.path.join(subjects_dir, s.upper(), 'bem', f'{s.upper()}-oct-6-src.fif')
    fwd_path = os.path.join(this_subj, 'forward', f'{s}-sss-fwd.fif')
    cov_path = os.path.join(this_subj, 'covariance', f'{s}-80-sss-cov.fif')
    inv_path = os.path.join(this_subj, 'inverse', f'{s}-80-sss-meg-inv.fif')
    evk_path = os.path.join(this_subj, 'inverse', f'Conditions_80-sss_eq_{s}-ave.fif')
    stc_path = os.path.join(this_subj, 'stc')
    # do the heavy lifting
    source = mne.read_source_spaces(src_path)
    verts_from = [source[0]['vertno'], source[1]['vertno']]
    fwd = mne.read_forward_solution(fwd_path)
    cov = mne.read_cov(cov_path)
    inv = read_inverse_operator(inv_path)
    evokeds = [mne.read_evokeds(evk_path, condition=c) for c in conditions]
    # prepare output dir
    if not os.path.isdir(stc_path):
        os.mkdir(stc_path)
    # make STCs with each method
    for method in methods:
        stcs = [apply_inverse(evk, inv, lambda2, method=method, pick_ori=None)
                for evk in evokeds]
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
