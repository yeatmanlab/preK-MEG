#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot frequency-domain STCs.
"""

import os
import numpy as np
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, div_by_adj_bins

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

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # load this subject's STC
        fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
        stc = mne.read_source_estimate(os.path.join(in_dir, fname))
        # convert complex values to magnitude & normalize to "SNR"
        magn_data = np.abs(stc.data)
        snr_data = div_by_adj_bins(magn_data)
        for kind, _data in dict(magnitude=magn_data, snr=snr_data).items():
            stc.data = _data
            # plot it
            brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
            for freq in (2, 4, 6, 12):
                brain.set_time(freq)
                fname = f'{s}-{timepoint}_camp-pskt{subdiv}-fft-{kind}-{freq:02}_Hz.png'  # noqa E501
                brain.save_image(os.path.join(fig_dir, fname))
            del brain
