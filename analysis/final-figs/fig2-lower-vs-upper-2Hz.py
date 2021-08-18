#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot group-level frequency-domain STCs.
"""

import os
import numpy as np
import mne
from mne.stats import ttest_ind_no_p
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    div_by_adj_bins, set_brain_view_distance)

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')

# load params
brain_plot_kwargs, _, subjects, cohort = load_params(experiment='pskt')
brain_plot_kwargs.update(time_label='freq=%0.2f Hz', surface='white',
                         background='white', size=(1000, 1400))

# load groups
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(letter_knowledge_group)

# inverse params
constraint = 'free'
estim_type = 'magnitude'
out_dir = f'{constraint}-{estim_type}'

# config other
timepoint = 'pre'
conditions = ('all',)

# loop over trial types
for condition in conditions:
    # use the same lims we used for the grand average data
    abs_lims = (100, 130, 280)
    snr_lims = (2, 2.5, 7.5)
    # split the groups
    data_dict = dict()
    for subgroup in ('Lower', 'Upper'):
        fft_data = list()
        for s in groups[f'{subgroup}Knowledge']:
            fname = (f'{s}FSAverage-{timepoint}_camp-pskt-'
                     f'{condition}-fft-stc.h5')
            fpath = os.path.join(in_dir, out_dir, fname)
            stc = mne.read_source_estimate(fpath, subject='fsaverage')
            fft_data.append(stc.data)
        fft_data = np.array(fft_data)
        snr_data = div_by_adj_bins(fft_data)
        data_dict[f'{subgroup}-fft'] = fft_data
        data_dict[f'{subgroup}-snr'] = snr_data
        # create within-group grand average STC
        snr_stc = stc.copy()
        fft_stc = stc.copy()
        snr_stc.data = snr_data.mean(axis=0)
        fft_stc.data = fft_data.mean(axis=0)
        del stc
        # plot untransformed data & SNR data
        for kind, lims, stc in zip(
                ['fft', 'snr'], [abs_lims, snr_lims], [fft_stc, snr_stc]):
            metric = 'SNR' if kind == 'snr' else 'dSPM'
            brain_plot_kwargs.update(time_label=f'{metric} (%0.2f Hz)')
            # plot stc
            clim = dict(kind='value', lims=lims)
            brain = stc.plot(subject='fsaverage', clim=clim,
                             **brain_plot_kwargs)
            freq = 2
            brain.set_time(freq)
            set_brain_view_distance(brain,
                                    views=brain_plot_kwargs['views'],
                                    hemi=brain_plot_kwargs['hemi'],
                                    distance=400)  # trial-and-error
            fname = (f'fig2-{cohort}-{subgroup}-pre_camp-pskt'
                     f'-{condition}-fft-{kind}-{freq:02}_Hz.png')
            brain.save_image(fname)
            brain.close()
            del brain

# recompute t-test data, just to make sure
# planned comparison: group split on pre-intervention letter awareness test
# 2-sample t-test on differences between SNRs
sigma = 1e-3  # hat adjustment for low variance
for kind in ('fft', 'snr'):
    median_split = list()
    for group in ('Upper', 'Lower'):
        this_data = data_dict[f'{group}-{kind}']
        median_split.append(this_data)
    median_split_tvals = np.array([
        ttest_ind_no_p(a, b, sigma=sigma) for a, b in zip(
            median_split[0].transpose(2, 0, 1),
            median_split[1].transpose(2, 0, 1))]).T
    tval_stc = fft_stc.copy()
    tval_stc.data = median_split_tvals
    # config
    tval_lims = (2, 2.8, 3.8)  # (1.75, 2, 3.25)
    metric = 'SNR' if kind == 'snr' else 'dSPM'
    brain_plot_kwargs.update(time_label=f't-value ({metric}, %0.2f Hz)')
    # plot stc
    clim = dict(kind='value', pos_lims=tval_lims)
    brain = tval_stc.plot(subject='fsaverage', clim=clim, **brain_plot_kwargs)
    freq = 2
    brain.set_time(freq)
    set_brain_view_distance(brain,
                            views=brain_plot_kwargs['views'],
                            hemi=brain_plot_kwargs['hemi'],
                            distance=400)  # trial-and-error
    fname = (f'fig2-{cohort}-Lower_vs_Upper_ttest-pre_camp-pskt'
             f'-{condition}-fft-{kind}-{freq:02}_Hz.png')
    brain.save_image(fname)
    brain.close()
    del brain
