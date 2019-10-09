#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses across all subjects, for each pairwise
condition contrast
"""

import os
from itertools import combinations
import mne
from aux_functions import load_params

# config paths
subj_root = '/mnt/scratch/prek/pre_camp/twa_hp/'
avg_path = os.path.join(subj_root, 'grand_averages')
mov_path = os.path.join(subj_root, 'movies')
# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA
# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# generate contrast pairs
contrasts = combinations(conditions, 2)

# make contrast STCs & movies
for method in methods:
    condition_dict = dict()
    # load the STC for each condition
    for cond in conditions:
        avg_fname = f'fsaverage_{method}_{cond}_GrandAvgN{n_subj}.stc'
        avg_fpath = os.path.join(avg_path, avg_fname)
        condition_dict[cond] = mne.read_source_estimate(avg_fpath)
    # make the contrasts
    for (cond_0, cond_1) in contrasts:
        stc = condition_dict[cond_0] - condition_dict[cond_1]
        contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
        contr_fname = f'fsaverage_{method}_{contr}_GrandAvgN{n_subj}.stc'
        stc.save(os.path.join(avg_path, contr_fname))

        # make movie
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{contr_fname[:-4]}.mov'
        brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
