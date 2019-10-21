#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses across all subjects, separately for pre- and
post-camp recordings, for each pairwise condition contrast.
"""

import os
from itertools import combinations
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

mlab.options.offscreen = True

# config paths
_, _, results_dir = load_paths()
avg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')
contr_out_path = os.path.join(results_dir, 'condition_contrasts')
if not os.path.isdir(contr_out_path):
    os.makedirs(contr_out_path, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM',)  # dSPM, sLORETA, eLORETA

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# generate contrast pairs
contrasts = combinations(conditions, 2)

# make contrast STCs & movies
group = f'GrandAvgN{len(subjects)}FSAverage'
# loop over pre/post measurement time
for prepost in ('pre', 'post'):
    # loop over algorithms
    for method in methods:
        condition_dict = dict()
        # load the STC for each condition
        for cond in conditions:
            avg_fname = f'{group}_{prepost}Camp_{method}_{cond}.stc'
            avg_fpath = os.path.join(avg_path, avg_fname)
            condition_dict[cond] = mne.read_source_estimate(avg_fpath)
        # make the condition contrasts
        for (cond_0, cond_1) in contrasts:
            stc = condition_dict[cond_0] - condition_dict[cond_1]
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            contr_fname = f'{group}_{prepost}Camp_{method}_{contr}.stc'
            stc.save(os.path.join(contr_out_path, contr_fname))

            # make movie
            brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
            mov_fname = f'{contr_fname[:-4]}.mov'
            brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
