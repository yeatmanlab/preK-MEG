#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_psd_params

# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
fig_dir = os.path.join(results_dir, 'pskt', 'fig', 'brain')
os.makedirs(fig_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()
psd_params = load_psd_params()

# config other
timepoints = ('pre', 'post')
subdivide_epochs = psd_params['epoch_dur']
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # load this subject's STC
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-multitaper-stc.h5'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname))
        # plot it
        brain = stc.magnitude().plot(subject='fsaverage', **brain_plot_kwargs)
        for freq in (2, 4, 6, 12):
            brain.set_time(freq)
            fname = f'{s}-{timepoint}_camp-pskt{subdiv}-multitaper-{freq:02}_Hz.png'  # noqa E501
            brain.save_image(os.path.join(fig_dir, fname))
        del brain
