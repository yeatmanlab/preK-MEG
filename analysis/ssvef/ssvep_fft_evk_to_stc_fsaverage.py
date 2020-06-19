#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load frequency-domain SSVEP evokeds, apply inverse, & morph to FSAverage.
"""

import os
import numpy as np
import mne
from analysis.aux_functions import load_paths, load_params

# flags
mne.cuda.init_cuda()

# config paths
data_root, subjects_dir, results_dir = load_paths()
fft_dir = os.path.join(results_dir, 'pskt', 'fft-evoked')
stc_dir = os.path.join(results_dir, 'pskt', 'stc', 'subject-specific')
morph_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
for _dir in (stc_dir, morph_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# inverse params
constraints = ('-free', '', '-fixed')     # empty string == loose
estim_types = ('vector', None, 'normal')  # None == vector magnitude

# for morph to fsaverage
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]

# loop over subjects
for s in subjects:
    has_morph = False
    # loop over timepoints
    for timepoint in timepoints:
        # (TODO: separately for "ps" and "kt" trials)
        stub = f'{s}-{timepoint}_camp-pskt{subdiv}'
        fname = f'{stub}-fft-ave.fif'
        evoked_spect = mne.read_evokeds(os.path.join(fft_dir, fname))
        assert len(evoked_spect) == 1
        evoked_spect = evoked_spect[0]
        # loop over cortical estimate orientation constraints
        for constr in constraints:
            constr_dir = constr.lstrip('-') if len(constr) else 'loose'
            # load inverse operator. it is only located in the ERP folder tree,
            # not in PSKT (TODO: this may change at some point)
            inv_path = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                    'erp', s, 'inverse',
                                    f'{s}-80-sss-meg{constr}-inv.fif')
            inverse = mne.minimum_norm.read_inverse_operator(inv_path)
            # loop over estimate types
            for estim_type in estim_types:
                # make the output dirs
                estim_dir = 'magnitude' if estim_type is None else estim_type
                for _dir in (stc_dir, morph_dir):
                    os.mkdirs(os.path.join(_dir, constr_dir, estim_dir),
                              exist_ok=True)
                # apply inverse & save
                stc = mne.minimum_norm.apply_inverse(
                    evoked_spect, inverse, lambda2, pick_ori=estim_type)
                assert stc.tstep == np.diff(evoked_spect.times[:2])
                fname = f'{stub}-fft'
                fpath = os.path.join(stc_dir, constr_dir, estim_dir, fname)
                stc.save(fpath, ftype='h5')
                # compute morph for this subject
                if not has_morph:
                    morph = mne.compute_source_morph(
                        stc, subject_from=s.upper(), subject_to='fsaverage',
                        subjects_dir=subjects_dir, spacing=fsaverage_vertices,
                        smooth=smoothing_steps)
                    has_morph = True
                # morph to fsaverage & save
                morphed_stc = morph.apply(stc)
                fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
                fpath = os.path.join(morph_dir, constr_dir, estim_dir, fname)
                morphed_stc.save(fpath, ftype='h5')
