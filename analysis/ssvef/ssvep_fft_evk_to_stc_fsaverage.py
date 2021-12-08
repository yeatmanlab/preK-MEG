#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load frequency-domain SSVEP evokeds, apply inverse, & morph to FSAverage.
"""

import os
import numpy as np
import mne
from sswef_helpers.aux_functions import (load_paths, load_params,
                                    load_fsaverage_src, load_inverse_params,
                                    yamload)

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
*_, subjects, cohort = load_params(experiment='pskt')
inverse_params = load_inverse_params()
inverse_method = inverse_params['method']
ori = inverse_params['orientation_constraint']
estim = inverse_params['estimate_type']
paramfile = os.path.join('..', 'preprocessing', 'mnefun_common_params.yaml')
with open(paramfile, 'r') as f:
    params = yamload(f)
lp_cut = params['preprocessing']['filtering']['lp_cut']
del params

# config other
timepoints = ('pre', 'post')
snr = 3.
lambda2 = 1. / snr ** 2
smoothing_steps = 10

# inverse params
# constraints = ('-free', '', '-fixed')  # empty string == loose
# estim_types = ('vector', None, 'normal')  # None == vector magnitude
constraints = ('' if ori == 'loose' else f'-{ori}',)
estim_types = (None if estim == 'magnitude' else estim,)

# for morph to fsaverage
fsaverage_src = load_fsaverage_src()
fsaverage_vertices = [s['vertno'] for s in fsaverage_src]

# loop over subjects
for s in subjects:
    has_morph = False
    # loop over timepoints
    for timepoint in timepoints:
        for condition in ('ps', 'kt', 'all'):
            stub = f'{s}-{timepoint}_camp-pskt'
            fname = f'{stub}-{condition}-fft-ave.fif'
            evoked_spect = mne.read_evokeds(os.path.join(fft_dir, fname))
            assert len(evoked_spect) == 1
            evoked_spect = evoked_spect[0]
            # loop over cortical estimate orientation constraints
            for constr in constraints:
                constr_dir = constr.lstrip('-') if len(constr) else 'loose'
                # load inverse operator
                inv_fname = f'{s}-{lp_cut}-sss-meg{constr}-inv.fif'
                inv_path = os.path.join(data_root, f'{timepoint}_camp',
                                        'twa_hp', 'pskt', s, 'inverse',
                                        inv_fname)
                inverse = mne.minimum_norm.read_inverse_operator(inv_path)
                # loop over estimate types
                for estim_type in estim_types:
                    if constr == '-fixed' and estim_type == 'normal':
                        continue  # not implemented
                    # make the output dirs
                    estim_dir = ('magnitude' if estim_type is None else
                                 estim_type)
                    out_dir = f'{constr_dir}-{estim_dir}'
                    for _dir in (stc_dir, morph_dir):
                        os.makedirs(os.path.join(_dir, out_dir), exist_ok=True)
                    # apply inverse & save
                    stc = mne.minimum_norm.apply_inverse(
                        evoked_spect, inverse, lambda2, pick_ori=estim_type,
                        method=inverse_method)
                    assert stc.tstep == np.diff(evoked_spect.times[:2])
                    fname = f'{stub}-{condition}-fft'
                    fpath = os.path.join(stc_dir, out_dir, fname)
                    stc.save(fpath, ftype='h5')
                    # compute morph for this subject
                    if not has_morph:
                        morph = mne.compute_source_morph(
                            stc,
                            subject_from=s.upper(),
                            subject_to='fsaverage',
                            subjects_dir=subjects_dir,
                            spacing=fsaverage_vertices,
                            smooth=smoothing_steps)
                        has_morph = True
                    # morph to fsaverage & save
                    morphed_stc = morph.apply(stc)
                    fname = f'{s}FSAverage-{timepoint}_camp-pskt-{condition}-fft'  # noqa E501
                    fpath = os.path.join(morph_dir, out_dir, fname)
                    print('Saving stc to %s' % fpath)
                    morphed_stc.save(fpath, ftype='h5')
