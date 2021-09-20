#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot group-level frequency-domain STCs.
"""

import os
import numpy as np
import mne
from analysis.aux_functions import (load_paths, load_params, div_by_adj_bins,
                                    set_brain_view_distance)

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')

# load params
brain_plot_kwargs, _, subjects, cohort = load_params(experiment='pskt')
brain_plot_kwargs.update(time_label='freq=%0.2f Hz', surface='white',
                         background='white', size=(1000, 1400))

# load groups
groups = dict(GrandAvg=subjects)

# inverse params
constraint = 'free'
estim_type = 'magnitude'
out_dir = f'{constraint}-{estim_type}'

# config other
timepoints = ('pre', 'post')
conditions = ('all',)  # ('ps', 'kt', 'all')

# loop over trial types
for condition in conditions:
    # load in all individual subject data first, to get colormap lims.
    # Not very memory efficient but shouldn't be too bad
    all_data = list()
    for s in groups['GrandAvg']:
        for timepoint in timepoints:
            fname = (f'{s}FSAverage-{timepoint}_camp-pskt-'
                     f'{condition}-fft-stc.h5')
            fpath = os.path.join(in_dir, out_dir, fname)
            stc = mne.read_source_estimate(fpath, subject='fsaverage')
            all_data.append(stc.data)
    abs_data = np.abs(all_data)
    snr_data = div_by_adj_bins(abs_data)
    # compute separate lims for untransformed data and SNR
    abs_lims = (100, 130, 280)
    snr_lims = (2, 2.5, 7.5)
    # create grand average STCs
    snr_stc = stc.copy()
    abs_stc = stc.copy()
    snr_stc.data = snr_data.mean(axis=0)
    abs_stc.data = abs_data.mean(axis=0)
    del stc
    # plot untransformed data & SNR data
    for kind, lims, stc in zip(
            ['amp', 'snr'], [abs_lims, snr_lims], [abs_stc, snr_stc]):
        metric = 'SNR' if kind == 'snr' else 'dSPM'
        brain_plot_kwargs.update(time_label=f'{metric} (%0.2f Hz)')
        # plot stc
        clim = dict(kind='value', lims=lims)
        brain = stc.plot(subject='fsaverage', clim=clim, **brain_plot_kwargs)
        for freq in (2, 6):
            brain.set_time(freq)
            set_brain_view_distance(brain,
                                    views=brain_plot_kwargs['views'],
                                    hemi=brain_plot_kwargs['hemi'],
                                    distance=400)  # trial-and-error
            fname = (f'fig1-{cohort}-GrandAvg-pre_and_post_camp-pskt'
                     f'-{condition}-fft-{kind}-{freq:02}_Hz.png')
            brain.save_image(fname)
        brain.close()
        del brain
