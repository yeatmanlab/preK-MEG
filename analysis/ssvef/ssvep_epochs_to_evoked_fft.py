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
fft_dir = os.path.join(results_dir, 'pskt', 'fft-evoked')
for _dir in (evk_dir, fft_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
*_, subjects, cohort = load_params()

# config other
timepoints = ('pre', 'post')

# loop over subjects
for s in subjects:
    # loop over timepoints
    for timepoint in timepoints:
        stub = f'{s}-{timepoint}_camp-pskt'
        # load epochs
        fname = f'{stub}-epo.fif'
        epochs = mne.read_epochs(os.path.join(in_dir, fname), proj=True)
        # create & save evoked (all trials, & separately for PS and KT trials)
        for condition in list(epochs.event_id) + [list(epochs.event_id)]:
            evoked = epochs[condition].average()
            label = 'all' if isinstance(condition, list) else condition
            fname = f'{stub}-{label}-ave.fif'
            evoked.save(os.path.join(evk_dir, fname))
            # FFT
            spacing = 1. / evoked.info['sfreq']
            spectrum = rfft(evoked.data, workers=-2)
            freqs = rfftfreq(evoked.times.size, spacing)
            # convert to fake evoked object and save
            evoked_spect = mne.EvokedArray(spectrum, evoked.info,
                                           nave=evoked.nave)
            evoked_spect.times = freqs
            evoked_spect.info['sfreq'] = 1. / np.diff(freqs[:2])[0]
            del evoked, spectrum
            fname = f'{stub}-{label}-fft-ave.fif'
            evoked_spect.save(os.path.join(fft_dir, fname))
