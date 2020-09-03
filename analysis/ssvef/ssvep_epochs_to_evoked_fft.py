#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Load SSVEP epochs, compute FFT, apply inverse, & morph to FSAverage.
"""

import os
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
import mne
from analysis.aux_functions import load_paths, load_params

# flags
mne.cuda.init_cuda()
cohorts = 'r_only'

# config paths
data_root, subjects_dir, results_dir = load_paths(cohorts=cohorts)
in_dir = os.path.join(results_dir, 'pskt', 'epochs')
evk_dir = os.path.join(results_dir, 'pskt', 'evoked')
fft_dir = os.path.join(results_dir, 'pskt', 'fft-evoked')
for _dir in (evk_dir, fft_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params(cohorts=cohorts)

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

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
        # convert to fake evoked object and save
        evoked_spect = mne.EvokedArray(spectrum, evoked.info, nave=evoked.nave)
        evoked_spect.times = freqs
        evoked_spect.info['sfreq'] = 1. / np.diff(freqs[:2])[0]
        del evoked, spectrum
        fname = f'{stub}-fft-ave.fif'
        evoked_spect.save(os.path.join(fft_dir, fname))
