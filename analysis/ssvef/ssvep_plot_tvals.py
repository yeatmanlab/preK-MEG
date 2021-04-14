#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot uncorrected t-value maps.
"""

import os
import numpy as np
import mne
from analysis.aux_functions import load_paths, load_params, load_inverse_params

# flags
save_movie = False

# load params
brain_plot_kwargs, movie_kwargs, subjects, cohort = load_params()
inverse_params = load_inverse_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
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
conditions = ('all', 'ps', 'kt')
freqs_of_interest = (0., 1., 2., 3., 4., 5., 6., 7., 12.)
precamp_fname = 'GrandAvg-pre_camp'
postcamp_fname = 'GrandAvg-post_camp'
precamp_t_fname = 'DataMinusNoise1samp-pre_camp'
postcamp_t_fname = 'DataMinusNoise1samp-post_camp'
median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

for condition in conditions:
    print(f'Plotting t-vals for {condition}')
    # load an STC as a template
    fname = f'{cohort}-GrandAvg-pre_camp-pskt-5_sec-{condition}-fft-amp'
    stc = mne.read_source_estimate(os.path.join(stc_dir, fname))

    for prefix in (precamp_fname,
                   postcamp_fname,
                   precamp_t_fname,
                   postcamp_t_fname,
                   median_split_fname,
                   intervention_fname):
        suffix = 'grandavg' if prefix.startswith('GrandAvg') else 'tvals'
        fname = f'{prefix}-{condition}-{suffix}.npy'
        tvals = np.load(os.path.join(tval_dir, fname))
        stc.data = tvals
        # sanity check against the group_level_aggregate_stcs step
        if prefix.startswith('GrandAvg'):
            check_fname = os.path.join(
                stc_dir,
                f'original-{prefix}-pskt-5_sec-{condition}-fft-snr-stc.h5')
            check_stc = mne.read_source_estimate(check_fname)
            msg = '\n'.join((check_fname, '≠', os.path.join(tval_dir, fname)))
            np.testing.assert_allclose(check_stc.data, tvals, err_msg=msg)
        # set the colormap lims
        clim = dict(kind='value')
        if prefix.startswith('Gran'):
            clim['lims'] = np.percentile(tvals, (95, 99, 99.5))
        else:
            # roughly 0.05, 0.01, 0.001 for roughly 20 subjects (which should
            # be close enough for most of our t-tests)
            clim['pos_lims'] = (2, 2.8, 3.8)
        # plot the brain
        brain = stc.plot(smoothing_steps='nearest', clim=clim, time_unit='s',
                         time_label='t-value (%0.2f Hz)', **brain_plot_kwargs)
        if save_movie:
            movie_dir = os.path.join(
                results_dir, 'pskt', 'group-level', 'fig', 'movie_frames',
                chosen_constraints, prefix)
            os.makedirs(movie_dir, exist_ok=True)
            # don't use brain.save_image_sequence because you can't include
            # actual time (freq) value in output filename (only a time index)
            for freq in stc.times:
                brain.set_time(freq)
                img_fname = f'{prefix}-{condition}-{freq:04.1f}_Hz.png'
                img_path = os.path.join(movie_dir, img_fname)
                brain.save_image(img_path)
                # also save to main directory
                if freq in freqs_of_interest:
                    img_fname = f'{prefix}-{condition}-{freq:04.1f}_Hz.png'
                    img_path = os.path.join(fig_dir, img_fname)
                    brain.save_image(img_path)
        else:
            for freq in freqs_of_interest:
                brain.set_time(freq)
                img_fname = f'{prefix}-{condition}-{freq:04.1f}_Hz.png'
                img_path = os.path.join(fig_dir, img_fname)
                brain.save_image(img_path)
        brain.close()
        del brain
