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
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'tvals')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# load fsaverage source space to get vertices
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
vertices = [fsaverage_src[hemi]['vertno'] for hemi in (0, 1)]

grandavg_fname = 'GrandAvg-PreAndPost_camp'
median_split_fname = 'LowerVsUpperKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

for prefix in (grandavg_fname, median_split_fname, intervention_fname):
    for freq in (2, 4, 6):
        fname = f'{prefix}-{freq}_Hz-SNR-{hemi}-tvals.npy'
        tvals = np.load(os.path.join(tval_dir, fname))
        tvals = tvals.transpose()  # (freqs, verts) → (verts, freqs)
        stc = mne.SourceEstimate(np.concatenate([tvals, np.zeros_like(tvals)]),
                                 vertices, tmin=freq, tstep=0.2,
                                 subject='fsaverage')
        # plot the brain
        brain = stc.plot(smoothing_steps='nearest', time_unit='s',
                         time_label='frequency (Hz)', initial_time=freq,
                         **brain_plot_kwargs)
        img_fname = re.sub(r'\.npy$', '.png', fname)
        img_path = os.path.join(fig_dir, img_fname)
        brain.save_image(img_path)
