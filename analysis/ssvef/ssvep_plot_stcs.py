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
brain_plot_kwargs, movie_kwargs, subjects, cohort = load_params()
brain_plot_kwargs.update(time_label='freq=%0.2f Hz')

# config other
timepoints = ('pre', 'post')
conditions = ('ps', 'kt', 'all')

# inverse params
constraints = ('free',)  # ('free', 'loose', 'fixed')
estim_types = ('magnitude',)  # ('vector', 'magnitude', 'normal')
morphed = True

# loop over timepoints
for timepoint in timepoints:
    # loop over subjects
    for s in subjects:
        # loop over trial types
        for condition in conditions:
            # loop over cortical estimate orientation constraints
            for constr in constraints:
                # loop over estimate types
                for estim_type in estim_types:
                    if constr == 'fixed' and estim_type == 'normal':
                        continue  # not implemented
                    # load this subject's STC
                    fsavg = 'FSAverage' if morphed else ''
                    fname = f'{s}{fsavg}-{timepoint}_camp-pskt-{condition}-fft'
                    fpath = os.path.join(in_dir, f'{constr}-{estim_type}',
                                         fname)
                    stc = mne.read_source_estimate(fpath)
                    # convert complex values to magnitude & normalize to "SNR"
                    abs_data = np.abs(stc.data)
                    snr_data = div_by_adj_bins(abs_data)
                    data_dict = dict(amp=abs_data, snr=snr_data)
                    # prepare output folder
                    out_dir = os.path.join(fig_dir, f'{constr}-{estim_type}')
                    os.makedirs(out_dir, exist_ok=True)
                    # loop over data kinds & plot
                    kwargs = (dict(surface='white') if estim_type != 'vector'
                              else dict(brain_alpha=1, overlay_alpha=0,
                                        vector_alpha=1))
                    for kind, _data in data_dict.items():
                        stc.data = _data
                        brain = stc.plot(subject='fsaverage',
                                         **brain_plot_kwargs, **kwargs)
                        for freq in (2, 4, 6, 12):
                            brain.set_time(freq)
                            bname = f'{fname}-{kind}-{freq:02}_Hz.png'
                            brain.save_image(os.path.join(out_dir, bname))
                        del brain
