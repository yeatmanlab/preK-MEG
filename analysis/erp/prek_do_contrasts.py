#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Make average Source Time Courses for various subject groups, conditions,
measurement timepoints, and source localization algorithms, and generate
movies for each.
"""

import os
from itertools import combinations
import mne
from sswef_helpers.aux_functions import (load_paths, load_params, load_cohorts,
                                    load_inverse_params)

mne.cuda.init_cuda()
overwrite = False
make_movies = False

# load params
(brain_plot_kwargs, movie_kwargs, subjects,
 cohort) = load_params(experiment='erp')
inverse_params = load_inverse_params()
method = inverse_params['method']
_, _, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')


# variables to loop over; subtractions between conditions are (lists of) tuples
timepoints = ('preCamp', 'postCamp')
conditions = ['words', 'faces', 'cars', 'aliens']
contrasts = {f'{contr[0].capitalize()}Minus{contr[1].capitalize()}': contr
             for contr in list(combinations(conditions[:-1], 2))}

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts(experiment='erp')

# assemble groups to iterate over

groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# LOOP ONCE THROUGH ALL CONDITIONS TO LOAD THE STCs INTO A DICT
stc_dict = dict()
for group_name, group_members in groups.items():
    group = f'{group_name}N{len(group_members)}FSAverage'
    stc_dict[group] = dict()
    # loop over pre/post measurement time
    for timepoint in timepoints:
        # skip conditions we don't need / care about
        if group_name.endswith('Knowledge') and timepoint == 'postCamp':
            continue
        stc_dict[group][timepoint] = dict()
        # loop over conditions
        for cond in conditions:
            fname = f'{group}_{timepoint}_{method}_{cond}'
            avg_fpath = os.path.join(groupavg_path, fname)
            stc = mne.read_source_estimate(avg_fpath)
            stc_dict[group][timepoint][cond] = stc

        # CONTRAST TRIAL CONDITIONS
        for contr_key, (contr_0, contr_1) in contrasts.items():
            stc = (stc_dict[group][timepoint][contr_0] -
                   stc_dict[group][timepoint][contr_1])
            stc_dict[group][timepoint][contr_key] = stc
            # save the contrast STC
            fname = f'{cohort}_{group}_{timepoint}_{method}_{contr_key}'
            stc.save(os.path.join(groupavg_path, fname))

    # CONTRAST POST-MINUS-PRE
    timepoint = 'PostCampMinusPreCamp'
    stc_dict[group][timepoint] = dict()
    for con in conditions + list(contrasts):
        # skip conditions we don't need / care about
        if group_name.endswith('Knowledge'):
            continue
        stc = (stc_dict[group]['postCamp'][con] -
               stc_dict[group]['preCamp'][con])
        stc_dict[group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
        stc.save(os.path.join(groupavg_path, fname))

# CONTRAST PRE-INTERVENTION LETTER KNOWLEDGE
timepoint = 'preCamp'
group_name = 'UpperMinusLowerKnowledge'
n_subj = {g: len(groups[g]) for g in letter_knowledge_group}
n = '-'.join([str(n_subj[g]) for g in letter_knowledge_group])
group = f'{group_name}N{n}FSAverage'
stc_dict[group] = dict()
stc_dict[group][timepoint] = dict()
keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in letter_knowledge_group}
for con in conditions + list(contrasts):
    stc = (stc_dict[keys['UpperKnowledge']][timepoint][con] -
           stc_dict[keys['LowerKnowledge']][timepoint][con])
    stc_dict[group][timepoint][con] = stc
    # save the contrast STC
    fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
    stc.save(os.path.join(groupavg_path, fname))

# CONTRAST EFFECT OF INTERVENTION ON COHORTS
if cohort != 'replication':
    timepoint = 'PostCampMinusPreCamp'
    group_name = 'LetterMinusLanguageIntervention'
    n_subj = {g: len(groups[g]) for g in intervention_group}
    n = '-'.join([str(n_subj[g]) for g in intervention_group])
    group = f'{group_name}N{n}FSAverage'
    stc_dict[group] = dict()
    stc_dict[group][timepoint] = dict()
    keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in intervention_group}
    for con in conditions + list(contrasts):
        stc = (stc_dict[keys['LetterIntervention']][timepoint][con] -
               stc_dict[keys['LanguageIntervention']][timepoint][con])
        stc_dict[group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
        stc.save(os.path.join(groupavg_path, fname))

# MAKE THE MOVIES
if make_movies:
    for group, timepoint_dict in stc_dict.items():
        for timepoint, condition_dict in timepoint_dict.items():
            for con, stc in condition_dict.items():
                # if the movie already exists and overwrite=False, skip it
                mov_fname = f'{cohort}_{group}_{timepoint}_{method}_{con}.mov'
                print(f'making movie {mov_fname}')
                mov_fpath = os.path.join(mov_path, mov_fname)
                if os.path.exists(mov_fpath) and not overwrite:
                    print(f'skipping {mov_fname}')
                    continue
                # make movie
                brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
                brain.save_movie(mov_fpath, **movie_kwargs)
                # clean up
                del brain
