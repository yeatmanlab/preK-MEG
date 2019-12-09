#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses across all subjects and across cohort groups,
showing the postcamp-minus-precamp change in each condition, and in each
pairwise condition contrast.
"""

import os
from itertools import combinations
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_cohorts

mlab.options.offscreen = True
mne.cuda.init_cuda()

# config paths
_, _, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')
prepost_path = os.path.join(results_dir, 'prepost_contrasts')
if not os.path.isdir(prepost_path):
    os.makedirs(prepost_path, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, _ = load_cohorts()

# generate contrast pairs (need a list so we can loop over it twice)
contrasts = list(combinations(conditions, 2))

# assemble groups to iterate over
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)

# loop over algorithms
for method in methods:
    # loop over groups
    for group_name, group_members in groups.items():
        group = f'{group_name}N{len(group_members)}FSAverage'
        # load the STC for each condition, pre- and post-camp
        condition_dict = dict()
        for cond in conditions:
            # subtract: post-camp minus pre-camp
            prepost_dict = dict()
            for prepost in ('pre', 'post'):
                fname = f'{group}_{prepost}Camp_{method}_{cond}'
                fpath = os.path.join(groupavg_path, fname)
                prepost_dict[prepost] = mne.read_source_estimate(fpath)
            stc = prepost_dict['post'] - prepost_dict['pre']
            prepost_fname = f'{group}_PostCampMinusPreCamp_{method}_{cond}'
            stc.save(os.path.join(prepost_path, prepost_fname))
            # retain post-minus-pre for each condition, for condition contrasts
            condition_dict[cond] = stc
            # make movie
            brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
            mov_fname = f'{prepost_fname}.mov'
            brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
            # clean up
            del brain
        del prepost_dict

        # do the condition contrasts (e.g., PostMinusPre_FacesMinusCars)
        for (cond_0, cond_1) in contrasts:
            stc = condition_dict[cond_0] - condition_dict[cond_1]
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            contr_fname = f'{group}_PostCampMinusPreCamp_{method}_{contr}'
            stc.save(os.path.join(prepost_path, contr_fname))
            # make movie
            brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
            mov_fname = f'{contr_fname}.mov'
            brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
            del brain
