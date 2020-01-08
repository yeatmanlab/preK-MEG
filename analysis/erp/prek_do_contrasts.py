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
from aux_functions import load_paths, load_params, load_cohorts

mlab.options.offscreen = True
mne.cuda.init_cuda()
overwrite = True
dry_run = True

# config paths
_, _, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# variables to loop over; subtractions between conditions are (lists of) tuples
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA
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
        group = f'{group_name}N{len(group_members)}FSAverage'
        stc_dict[method][group] = dict()
        # loop over pre/post measurement time
        for timepoint in timepoints:
            stc_dict[method][group][timepoint] = dict()
            # loop over conditions
            for cond in conditions:
                fname = f'{group}_{timepoint}_{method}_{cond}'
                avg_fpath = os.path.join(groupavg_path, fname)
                if dry_run:
                    print(f'dry run: loading {fname}')
                else:
                    stc = mne.read_source_estimate(avg_fpath)
                    stc_dict[method][group][timepoint][cond] = stc

            # CONTRAST TRIAL CONDITIONS
            for contr_key, (contr_0, contr_1) in contrasts.items():
                if dry_run:
                    print(f'dry run: subtracting {method}_{group}_{timepoint}_{contr_key}')  # noqa
                else:
                    stc = (stc_dict[method][group][timepoint][contr_0] -
                           stc_dict[method][group][timepoint][contr_1])
                    stc_dict[method][group][timepoint][contr_key] = stc
                # save the contrast STC
                fname = f'{group}_{timepoint}_{method}_{contr_key}'
                if dry_run:
                    print(f'dry run: saving {fname}')
                else:
                    stc.save(os.path.join(groupavg_path, fname))

        # CONTRAST POST-MINUS-PRE
        timepoint = 'PostCampMinusPreCamp'
        for con in conditions + list(contrasts):
            if dry_run:
                print(f'dry run: subtracting {method}_{group}_{timepoint}_{con}')  # noqa
            else:
                stc = (stc_dict[method][group]['postCamp'][con] -
                       stc_dict[method][group]['preCamp'][con])
                stc_dict[method][group][timepoint][con] = stc
            # save the contrast STC
            fname = f'{group}_{timepoint}_{method}_{con}'
            if dry_run:
                print(f'dry run: saving {fname}')
            else:
                stc.save(os.path.join(groupavg_path, fname))

    # CONTRAST PRE-INTERVENTION LETTER KNOWLEDGE
    group_name = 'UpperMinusLowerKnowledge'
    n_subj = '-'.join([str(len(groups[g])) for g in ('UpperKnowledge',
                                                     'LowerKnowledge')])
    group = f'{group_name}N{n_subj}FSAverage'
    timepoint = 'preCamp'
    for con in conditions + list(contrasts):
        if dry_run:
            print(f'dry run: subtracting {method}_{group}_{timepoint}_{con}')
        else:
            stc = (stc_dict[method]['UpperKnowledge'][timepoint][con] -
                   stc_dict[method]['LowerKnowledge'][timepoint][con])
            stc_dict[method][group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{group}_{timepoint}_{method}_{con}'
        if dry_run:
            print(f'dry run: saving {fname}')
        else:
            stc.save(os.path.join(groupavg_path, fname))

    # CONTRAST EFFECT OF INTERVENTION ON COHORTS
    group_name = 'LetterMinusLanguageIntervention'
    n_subj = '-'.join([str(len(groups[g])) for g in ('LetterIntervention',
                                                     'LanguageIntervention')])
    group = f'{group_name}N{n_subj}FSAverage'
    timepoint = 'PostCampMinusPreCamp'
    for con in conditions + list(contrasts):
        if dry_run:
            print(f'dry run: subtracting {method}_{group}_{timepoint}_{con}')
        else:
            stc = (stc_dict[method]['LetterIntervention'][timepoint][con] -
                   stc_dict[method]['LanguageIntervention'][timepoint][con])
            stc_dict[method][group][timepoint][con] = stc
        # save the contrast STC
        fname = f'{group}_{timepoint}_{method}_{con}'
        if dry_run:
            print(f'dry run: saving {fname}')
        else:
            stc.save(os.path.join(groupavg_path, fname))

# MAKE THE MOVIES
if dry_run:
    print('dry run: skipping movies')
    exit()
for method, group_dict in stc_dict.items():
    for group, timepoint_dict in group_dict.items():
        for timepoint, condition_dict in timepoint_dict.items():
            for con, stc in condition_dict:
                # if the movie already exists and overwrite=False, skip it
                mov_fname = f'{group}_{timepoint}_{method}_{con}.mov'
                mov_fpath = os.path.join(mov_path, mov_fname)
                if os.path.exists(mov_fpath) and not overwrite:
                    print(f'skipping {mov_fname}')
                    continue
                # make movie
                brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
                brain.save_movie(mov_fpath, **movie_kwargs)
                # clean up
                del brain
