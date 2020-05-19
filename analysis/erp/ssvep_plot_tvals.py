#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot uncorrected t-maps.
"""

import os
import numpy as np
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()

# flags
save_movie = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'tvals')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config other
timepoints = ('pre', 'post')

# load an STC as a template
fname = 'GrandAvg-pre_camp-pskt-5_sec-fft-snr'
stc = mne.read_source_estimate(os.path.join(stc_dir, fname))

precamp_fname = 'GrandAvg-pre_camp'
postcamp_fname = 'GrandAvg-post_camp'
median_split_fname = 'LowerVsUpperKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

for prefix in (precamp_fname, postcamp_fname, median_split_fname,
               intervention_fname):
    fname = f'{prefix}-tvals.npy'
    tvals = np.load(os.path.join(tval_dir, fname))
    stc.data = tvals
    # plot the brain
    brain = stc.plot(smoothing_steps='nearest', time_unit='s',
                     time_label='t-value', **brain_plot_kwargs)
    for freq in (2, 4, 6, 12):
        brain.set_time(freq)
        img_fname = f'{prefix}-{freq:02}_Hz.png'
        img_path = os.path.join(fig_dir, img_fname)
        brain.save_image(img_path)
    if save_movie:
        movie_fname = f'{prefix}.mov'
        brain.save_movie(os.path.join(fig_dir, movie_fname), **movie_kwargs)
    del brain
