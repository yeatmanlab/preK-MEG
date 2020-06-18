#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs, compute FFT, apply inverse, & morph to FSAverage.
"""

import os
import numpy as np
from scipy.fft import rfft, rfftfreq
import mne
from analysis.aux_functions import load_paths, load_params

# flags
mne.cuda.init_cuda()


# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
evk_dir = os.path.join(results_dir, 'pskt', 'evoked')
psd_dir = os.path.join(results_dir, 'pskt', 'psd')
stc_dir = os.path.join(results_dir, 'pskt', 'stc', 'subject-specific')
morph_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
for _dir in (evk_dir, stc_dir, morph_dir, psd_dir):
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
        stub = f'{s}-{timepoint}_camp-pskt{subdiv}'
        # load epochs (TODO: separately for "ps" and "kt" trials)
        fname = f'{stub}-epo.fif'
        epochs = mne.read_epochs(os.path.join(in_dir, fname), proj=True)
        # create & save evoked
        evoked = epochs.average()
        fname = f'{stub}-ave.fif'
        evoked.save(os.path.join(evk_dir, fname))
        del epochs
        # FFT
        spacing = 1. / evoked.info['sfreq']
        spectrum = rfft(evoked.data, workers=-2)
        freqs = rfftfreq(evoked.times.size, spacing)
        fname = f'{stub}-sensor_fft.npz'
        np.savez(os.path.join(psd_dir, fname), spectrum=spectrum, freqs=freqs)
        # convert to fake evoked object
        evoked_spect = mne.EvokedArray(spectrum, evoked.info, nave=evoked.nave)
        evoked_spect.times = freqs
        evoked_spect.info['sfreq'] = 1. / np.diff(freqs[:2])[0]
        del evoked, spectrum
        # go to source space. inverse is only located in the ERP folder tree,
        # not in PSKT (TODO: this may change at some point)
        inv_path = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                'erp', s, 'inverse', f'{s}-80-sss-meg-inv.fif')
        inverse = mne.minimum_norm.read_inverse_operator(inv_path)
        stc = mne.minimum_norm.apply_inverse(
            evoked_spect, inverse, lambda2, pick_ori='normal')
        assert stc.tstep == np.diff(freqs[:2])
        assert np.all(stc.times == freqs)
        fname = f'{stub}-fft'
        stc.save(os.path.join(stc_dir, fname), ftype='h5')
        # compute morph for this subject
        if not has_morph:
            morph = mne.compute_source_morph(stc, subject_from=s.upper(),
                                             subject_to='fsaverage',
                                             subjects_dir=subjects_dir,
                                             spacing=fsaverage_vertices,
                                             smooth=smoothing_steps)
            has_morph = True
        # morph and save
        morphed_stc = morph.apply(stc)
        assert np.all(morphed_stc.times == freqs)
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
        morphed_stc.save(os.path.join(morph_dir, fname), ftype='h5')
