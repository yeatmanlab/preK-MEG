#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot uncorrected t-value maps.
"""

import os
import numpy as np
from mayavi import mlab
import mne
from analysis.aux_functions import load_paths, load_params, load_inverse_params

mlab.options.offscreen = True
mne.cuda.init_cuda()

# flags
save_movie = False
cohorts = 'all'

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params(cohorts=cohorts)
inverse_params = load_inverse_params()

# config paths
data_root, subjects_dir, results_dir = load_paths(cohorts=cohorts)
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

stc_dir = os.path.join(results_dir, 'pskt', 'group-level', 'stc',
                       chosen_constraints)
tval_dir = os.path.join(results_dir, 'pskt', 'group-level', 'tvals',
                        chosen_constraints)
fig_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'tvals',
                       chosen_constraints)
for _dir in (fig_dir,):
    os.makedirs(_dir, exist_ok=True)

# config other
timepoints = ('pre', 'post')
freqs_of_interest = (0, 1, 2, 3, 4, 5, 6, 7, 12)
precamp_fname = 'GrandAvg-pre_camp'
postcamp_fname = 'GrandAvg-post_camp'
median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

# load an STC as a template
fname = f'{cohorts}_GrandAvg-pre_camp-pskt-5_sec-fft-amp'
stc = mne.read_source_estimate(os.path.join(stc_dir, fname))

for prefix in (precamp_fname, postcamp_fname, median_split_fname,
               intervention_fname):
    fname = f'{prefix}-tvals.npy'
    tvals = np.load(os.path.join(tval_dir, fname))
    stc.data = tvals
    # set the colormap lims
    clim = dict(kind='value')
    lims = tuple(np.percentile(tvals, (99, 99.9, 99.99)))
    pos_lims = tuple(np.percentile(tvals, (95, 99, 99.9)))
    pos = (dict(lims=lims) if prefix.startswith('Gran') else
           dict(pos_lims=pos_lims))
    clim.update(pos)
    # plot the brain
    brain = stc.plot(smoothing_steps='nearest', clim=clim, time_unit='s',
                     time_label='t-value (%0.2f Hz)', **brain_plot_kwargs)
    if save_movie:
        movie_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig',
                                 'movie_frames', chosen_constraints, prefix)
        os.makedirs(movie_dir, exist_ok=True)
        # don't use brain.save_image_sequence because you can't include actual
        # time (freq) value in output filename (only a time index)
        for freq in stc.times:
            brain.set_time(freq)
            img_fname = f'{prefix}-{freq:04.1f}_Hz.png'
            img_path = os.path.join(movie_dir, img_fname)
            brain.save_image(img_path)
            # also save to main directory
            if freq in freqs_of_interest:
                img_fname = f'{prefix}-{freq:04.1f}_Hz.png'
                img_path = os.path.join(fig_dir, img_fname)
                brain.save_image(img_path)
    else:
        for freq in freqs_of_interest:
            brain.set_time(freq)
            img_fname = f'{prefix}-{freq:04.1f}_Hz.png'
            img_path = os.path.join(fig_dir, img_fname)
            brain.save_image(img_path)
    del brain
