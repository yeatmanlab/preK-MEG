#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses across groups of subjects, separately for
pre- and post-camp recordings, for each pairwise condition contrast.
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
contrast_path = os.path.join(results_dir, 'condition_contrasts')
if not os.path.isdir(contrast_path):
    os.makedirs(contrast_path, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
_, letter_knowledge_group = load_cohorts()

# only one group to iterate over here
groups = letter_knowledge_group

# generate contrast pairs (need a list so we can loop over it twice)
contrasts = list(combinations(conditions, 2))

# loop over algorithms
for method in methods:
    # loop over groups
    group_dict = dict()
    for group_name, group_members in groups.items():
        group = f'{group_name}N{len(group_members)}FSAverage'
        group_dict[group_name] = dict()
        # load the STC for each condition
        for cond in conditions:
            avg_fname = f'{group}_preCamp_{method}_{cond}'
            avg_fpath = os.path.join(groupavg_path, avg_fname)
            group_dict[group_name][cond] = mne.read_source_estimate(avg_fpath)
        # make the condition contrasts
        for (cond_0, cond_1) in contrasts:
            stc = (group_dict[group_name][cond_0] -
                   group_dict[group_name][cond_1])
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            contr_fname = f'{group}_preCamp_{method}_{contr}'
            stc.save(os.path.join(contrast_path, contr_fname))
            group_dict[group_name][contr] = stc

            # make movie
            brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
            mov_fname = f'{contr_fname}.mov'
            brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
            # clean up
            mlab.close(all=True)
            del brain

    # do the group contrast (UpperMinusLower)
    for con in conditions + contrasts:
        stc = (group_dict['UpperKnowledge'][con] -
               group_dict['LowerKnowledge'][con])
        con_fname = f'UpperMinusLowerKnowledge_preCamp_{method}_{con}'
        stc.save(os.path.join(contrast_path, con_fname))

        # make movie
        brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{con_fname}.mov'
        brain.save_movie(os.path.join(mov_path, mov_fname), **movie_kwargs)
        # clean up
        mlab.close(all=True)
        del brain
