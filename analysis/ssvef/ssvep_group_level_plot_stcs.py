#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot group-level frequency-domain STCs.
"""

import os
import numpy as np
from mayavi import mlab
import mne
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    div_by_adj_bins)


# flags
mlab.options.offscreen = True
mne.cuda.init_cuda()
mne.viz.set_3d_backend('mayavi')

# config paths
_, _, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'stc', 'morphed-to-fsaverage')
stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc')
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'brain')
for _dir in (stc_dir, fig_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, movie_kwargs, subjects, cohort = load_params()
brain_plot_kwargs.update(time_label='freq=%0.2f Hz')

# load groups
intervention_group, letter_knowledge_group = load_cohorts()
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# inverse params
constraints = ('free', 'loose', 'fixed')
estim_types = ('vector', 'magnitude', 'normal')

# config other
timepoints = ('pre', 'post')
trial_dur = 20
subdivide_epochs = 5
subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''

# loop over cortical estimate orientation constraints
for constr in constraints:
    # loop over estimate types
    for estim_type in estim_types:
        if constr == 'fixed' and estim_type == 'normal':
            continue  # not implemented
        # make the output directory if needed
        out_dir = f'{constr}-{estim_type}'
        os.makedirs(os.path.join(fig_dir, out_dir), exist_ok=True)

        # load in all individual subject data first, to get colormap limits.
        # Not very memory efficient but shouldn't be too bad
        all_data = list()
        for s in groups['GrandAvg']:
            for timepoint in timepoints:
                fname = f'{s}FSAverage-{timepoint}_camp-pskt{subdiv}-fft-stc.h5'  # noqa E501
                fpath = os.path.join(in_dir, out_dir, fname)
                stc = mne.read_source_estimate(fpath, subject='fsaverage')
                all_data.append(stc.data)
        abs_data = np.abs(all_data)
        snr_data = div_by_adj_bins(abs_data)
        # compute separate lims for untransformed data and SNR
        cmap_percentiles = (91, 95, 99)
        abs_lims = tuple(np.percentile(abs_data, cmap_percentiles))
        snr_lims = tuple(np.percentile(snr_data, cmap_percentiles))
        # clean up
        del all_data, abs_data, snr_data

        # unify appearance of surface/vector plots
        kwargs = (dict(surface='white') if estim_type != 'vector' else
                  dict(brain_alpha=1, overlay_alpha=0, vector_alpha=1))
        # loop over timepoints
        for timepoint in timepoints:
            # loop over cohort groups
            for group in groups:
                # only do pretest knowledge comparison for pre-camp timepoint
                if group.endswith('Knowledge') and timepoint == 'post':
                    continue
                # load and plot untransformed data & SNR data
                for kind, lims in zip(['amp', 'snr'], [abs_lims, snr_lims]):
                    # load stc
                    fname = f'{cohort}-{group}-{timepoint}_camp-pskt{subdiv}-fft-{kind}'  # noqa E501
                    fpath = os.path.join(stc_dir, out_dir, fname)
                    stc = mne.read_source_estimate(fpath, subject='fsaverage')
                    # plot stc
                    clim = dict(kind='value', lims=lims)
                    brain = stc.plot(subject='fsaverage', clim=clim,
                                     **brain_plot_kwargs, **kwargs)
                    for freq in (2, 4, 6, 12):
                        brain.set_time(freq)
                        fpath = os.path.join(fig_dir, out_dir,
                                             f'{fname}-{freq:02}_Hz.png')
                        brain.save_image(fpath)
                    del brain
