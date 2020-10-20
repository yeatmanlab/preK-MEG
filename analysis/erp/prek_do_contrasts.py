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
from mayavi import mlab
import mne
from analysis.aux_functions import load_paths, load_params, load_cohorts

#mlab.options.offscreen = True
mne.cuda.init_cuda()
overwrite = False

# load params
brain_plot_kwargs, movie_kwargs, subjects, cohort = load_params()
_, _, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')


# variables to loop over; subtractions between conditions are (lists of) tuples
methods = ('dSPM',)  # dSPM, sLORETA, eLORETA
timepoints = ('preCamp', 'postCamp')
conditions = ['words', 'faces', 'cars', 'aliens']
contrasts = {f'{contr[0].capitalize()}Minus{contr[1].capitalize()}': contr
             for contr in list(combinations(conditions[:-1], 2))}

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# assemble groups to iterate over

groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

# LOOP ONCE THROUGH ALL CONDITIONS TO LOAD THE STCs INTO A DICT
stc_dict = dict()
# loop over source localization algorithms
for method in methods:
    stc_dict[method] = dict()
    # loop once through to load them all in
    for group_name, group_members in groups.items():
        print('Working on group %s.' % group_name)
        group = f'{group_name}N{len(group_members)}FSAverage'
        stc_dict[method][group] = dict()
        # loop over pre/post measurement time
        for timepoint in timepoints:
            print('Timepoint %s' % timepoint)
            # skip conditions we don't need / care about
            if group_name.endswith('Knowledge') and timepoint == 'postCamp':
                continue
            stc_dict[method][group][timepoint] = dict()
            # loop over conditions
            for cond in conditions:
                print('Adding in condition %s' % cond)
                fname = f'{group}_{timepoint}_{method}_{cond}'
                avg_fpath = os.path.join(groupavg_path, fname)
                stc = mne.read_source_estimate(avg_fpath)
                stc_dict[method][group][timepoint][cond] = stc

            # CONTRAST TRIAL CONDITIONS
            for contr_key, (contr_0, contr_1) in contrasts.items():
                print('Working on contrast %s.' % contr_key)
                stc = (stc_dict[method][group][timepoint][contr_0] -
                       stc_dict[method][group][timepoint][contr_1])
                stc_dict[method][group][timepoint][contr_key] = stc
                # save the contrast STC
                fname = f'{cohort}_{group}_{timepoint}_{method}_{contr_key}'
                print('Saving stc %s' % fname)
                stc.save(os.path.join(groupavg_path, fname))

        # CONTRAST POST-MINUS-PRE
        timepoint = 'PostCampMinusPreCamp'
        print('Working on pre minus posts contrasts.')
        stc_dict[method][group][timepoint] = dict()
        for con in conditions + list(contrasts):
            # skip conditions we don't need / care about
            if group_name.endswith('Knowledge'):
                continue
            stc = (stc_dict[method][group]['postCamp'][con] -
                   stc_dict[method][group]['preCamp'][con])
            stc_dict[method][group][timepoint][con] = stc
            # save the contrast STC
            fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
            print('Saving stc %s' % fname)
            stc.save(os.path.join(groupavg_path, fname))

    # CONTRAST PRE-INTERVENTION LETTER KNOWLEDGE
    timepoint = 'preCamp'
    group_name = 'UpperMinusLowerKnowledge'
    print('Working on letter knowledge contrasts.')
    n_subj = {g: len(groups[g]) for g in letter_knowledge_group}
    n = '-'.join([str(n_subj[g]) for g in letter_knowledge_group])
    group = f'{group_name}N{n}FSAverage'
    stc_dict[method][group] = dict()
    stc_dict[method][group][timepoint] = dict()
    keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in letter_knowledge_group}
    for con in conditions + list(contrasts):
        stc = (stc_dict[method][keys['UpperKnowledge']][timepoint][con] -
               stc_dict[method][keys['LowerKnowledge']][timepoint][con])
        stc_dict[method][group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
        print('Saving stc %s' % fname)
        stc.save(os.path.join(groupavg_path, fname))

    # CONTRAST EFFECT OF INTERVENTION ON COHORTS
    if cohort == 'replication':
        print('Skipping intervention type contrast for replication cohort.')
        continue
    timepoint = 'PostCampMinusPreCamp'
    group_name = 'LetterMinusLanguageIntervention'
    n_subj = {g: len(groups[g]) for g in intervention_group}
    n = '-'.join([str(n_subj[g]) for g in intervention_group])
    group = f'{group_name}N{n}FSAverage'
    stc_dict[method][group] = dict()
    stc_dict[method][group][timepoint] = dict()
    keys = {g: f'{g}N{n_subj[g]}FSAverage' for g in intervention_group}
    for con in conditions + list(contrasts):
        stc = (stc_dict[method][keys['LetterIntervention']][timepoint][con] -
               stc_dict[method][keys['LanguageIntervention']][timepoint][con])
        stc_dict[method][group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{cohort}_{group}_{timepoint}_{method}_{con}'
        stc.save(os.path.join(groupavg_path, fname))

# MAKE THE MOVIES
for method, group_dict in stc_dict.items():
    for group, timepoint_dict in group_dict.items():
        for timepoint, condition_dict in timepoint_dict.items():
            for con, stc in condition_dict.items():
                # if the movie already exists and overwrite=False, skip it
                mov_fname = f'{cohort}_{group}_{timepoint}_{method}_{con}.mov'
                print('Making movie %s' % mov_fname)
                mov_fpath = os.path.join(mov_path, mov_fname)
                if os.path.exists(mov_fpath) and not overwrite:
                    print(f'skipping {mov_fname}')
                    continue
                # make movie
                brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
                brain.save_movie(mov_fpath, **movie_kwargs)
                # clean up
                del brain
