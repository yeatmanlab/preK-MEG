#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
from glob import glob
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'brain', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'brain', 'fig')
for _dir in (stc_dir, fig_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config other
timepoints = ('pre', 'post')

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # load this subject's STC
        fname = f'{s}FSAverage-{timepoint}_camp-pskt-multitaper-stc.h5'
        stc = mne.read_source_estimate(fname)
        # plot it
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        for freq in (2, 4, 6, 12, 16):
            brain.set_time(freq)
            fname = f'{s}-{timepoint}_camp-pskt-multitaper-{freq:02}_Hz.png'
            fpath = os.path.join(fig_dir, fname)
            brain.save_image(fpath)
        del brain
