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
from analysis.aux_functions import load_paths, load_params, div_by_adj_bins

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
brain_plot_kwargs.update(time_label='freq=%0.2f Hz')

# config other
timepoints = ('pre', 'post')
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# inverse params
constraints = ('free', 'loose', 'fixed')
estim_types = ('vector', 'magnitude', 'normal')

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # loop over cortical estimate orientation constraints
        for constr in constraints:
            # loop over estimate types
            for estim_type in estim_types:
                if constr == 'fixed' and estim_type == 'normal':
                    continue  # not implemented
                # load this subject's STC
                fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft'
                fpath = os.path.join(in_dir, f'{constr}-{estim_type}', fname)
                stc = mne.read_source_estimate(fpath)
                # convert complex values to magnitude & normalize to "SNR"
                magn_data = np.abs(stc.data)
                snr_data = div_by_adj_bins(magn_data)
                data_dict = dict(magnitude=magn_data, snr=snr_data)
                # prepare output folder
                out_dir = os.path.join(fig_dir, f'{constr}-{estim_type}')
                os.makedirs(out_dir, exist_ok=True)
                # loop over data kinds
                for kind, _data in data_dict.items():
                    stc.data = _data
                    # plot it
                    brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
                    for freq in (2, 4, 6, 12):
                        brain.set_time(freq)
                        fname = f'{s}-{timepoint}_camp-pskt{subdiv}-fft-{kind}-{freq:02}_Hz.png'  # noqa E501
                        brain.save_image(os.path.join(out_dir, fname))
                    del brain
