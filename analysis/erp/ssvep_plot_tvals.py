#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot uncorrected t-maps.
"""

import os
import re
import numpy as np
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

mlab.options.offscreen = True

# flags
hemi = 'lh'

# config paths
data_root, subjects_dir, results_dir = load_paths()
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'tvals')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load an STC as a template
fname = 'GrandAvg-pre_camp-pskt-5_sec-fft-snr'
stc = mne.read_source_estimate(os.path.join(stc_dir, fname))
stc.data = np.zeros_like(stc.data)
if hemi == 'lh':
    attr = 'lh_data'
elif hemi == 'rh':
    attr = 'rh_data'
else:
    attr = 'data'

grandavg_fname = 'GrandAvg-PreAndPost_camp'
median_split_fname = 'LowerVsUpperKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

for prefix in (grandavg_fname, median_split_fname, intervention_fname):
    for freq in (2, 4, 6):
        fname = f'{prefix}-{freq}_Hz-SNR-{hemi}-tvals.npy'
        tvals = np.load(os.path.join(tval_dir, fname))
        tvals = tvals.transpose()  # (freqs, verts) → (verts, freqs)
        # cram in the data
        bin_idx = np.argmin(np.abs(stc.times - freq))
        stc.data[:, bin_idx] = tvals
        # plot the brain
        brain = stc.plot(smoothing_steps='nearest', time_unit='s',
                         time_label='t-value', initial_time=freq,
                         **brain_plot_kwargs)
        img_fname = re.sub(r'\.npy$', '.png', fname)
        img_path = os.path.join(fig_dir, img_fname)
        brain.save_image(img_path)
