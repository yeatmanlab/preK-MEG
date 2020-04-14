#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Convert SSVEP epochs to STCs.
"""

import os
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True
n_jobs = 10
smoothing_steps = 10

# config paths
data_root, subjects_dir, results_dir = load_paths()
epo_dir = os.path.join(results_dir, 'pskt', 'epochs')
stc_dir = os.path.join(results_dir, 'pskt', 'stc', 'subject-specific')
morph_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
for _dir in (stc_dir, morph_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')

# PSD settings
psd_kwargs = dict(fmin=0, fmax=20, bandwidth=0.1, adaptive=False,
                  n_jobs=n_jobs)

# for morph to fsaverage
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]

# loop over subjects
for s in subjects:
    already_morphed = False
    # loop over timepoints
    for timepoint in timepoints:
        # load inverse. inverse is only located in the ERP folder tree, not in
        # PSKT (TODO: this may change at some point)
        inv_path = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                'erp', s, 'inverse', f'{s}-80-sss-meg-inv.fif')
        inverse = mne.minimum_norm.read_inverse_operator(inv_path)
        # load epochs
        fname = f'{s}-{timepoint}_camp-pskt-epo.fif'
        fpath = os.path.join(epo_dir, fname)
        epochs = mne.read_epochs(fpath, proj=True)

        # make STC
        stcs_generator = mne.minimum_norm.compute_source_psd_epochs(
            epochs, inverse, label=None, nave=1, return_generator=True,
            **psd_kwargs)

        # save
        fpath = os.path.join(stc_dir, f'{timepoint}_camp', s)
        os.makedirs(fpath, exist_ok=True)
        for ix, stc in enumerate(stcs_generator):
            fname = f'{s}-{timepoint}_camp-pskt-{ix:02}'
            stc.save(os.path.join(fpath, fname))
            # compute morph for this subject
            if not already_morphed:
                morph = mne.compute_source_morph(stc, subject_from=s.upper(),
                                                 subject_to='fsaverage',
                                                 subjects_dir=subjects_dir,
                                                 spacing=fsaverage_vertices,
                                                 smooth=smoothing_steps)
                already_morphed = True
            # morph and save
            morphed_stc = morph.apply(stc)
            fname = f'{s}FSAverage-{timepoint}_camp-pskt-{ix:02}'
            fpath = os.path.join(morph_dir, f'{timepoint}_camp', s)
            os.makedirs(fpath, exist_ok=True)
            stc.save(os.path.join(fpath, fname))
