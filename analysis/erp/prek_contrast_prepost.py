#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses across all subjects, showing the
postcamp-minus-precamp change in each condition, and in each pairwise condition
contrast.
"""

import os
from itertools import combinations
import mne
from aux_functions import load_params

# config paths
project_root = '/mnt/scratch/prek'
avg_path = os.path.join(project_root, 'results', 'group_averages')
mov_path = os.path.join(project_root, 'results', 'movies')
prepost_out_path = os.path.join(project_root, 'results', 'prepost_contrasts')
if not os.path.isdir(prepost_out_path):
    os.makedirs(prepost_out_path, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# generate contrast pairs
contrasts = combinations(conditions, 2)

# make contrast STCs & movies
group = f'GrandAvgN{len(subjects)}FSAverage'
# loop over algorithms
for method in methods:
    condition_dict = dict()
    # load the STC for each condition, pre- and post-camp
    for cond in conditions:
        prepost_dict = dict()
        # subtract: post-camp minus pre-camp
        for prepost in ('pre', 'post'):
            fname = f'{group}_{prepost}Camp_{method}_{cond}.stc'
            fpath = os.path.join(avg_path, fname)
            prepost_dict[prepost] = mne.read_source_estimate(fpath)
        stc = prepost_dict['post'] - prepost_dict['pre']
        prepost_fname = f'{group}_PostCampMinusPreCamp_{method}_{cond}.stc'
        stc.save(os.path.join(prepost_out_path, prepost_fname))
        # make movie
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{prepost_fname[:-4]}.mov'
        brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
        # retain post-minus-pre for each condition, for condition contrasts
        condition_dict[cond] = stc
    # clean up
    del prepost_dict, brain

    # make the condition contrasts
    for (cond_0, cond_1) in contrasts:
        stc = condition_dict[cond_0] - condition_dict[cond_1]
        contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
        contr_fname = f'{group}_PostCampMinusPreCamp_{method}_{contr}.stc'
        stc.save(os.path.join(prepost_out_path, contr_fname))
        # make movie
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{contr_fname[:-4]}.mov'
        brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
