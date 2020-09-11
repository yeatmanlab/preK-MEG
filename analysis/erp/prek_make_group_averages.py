#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maggie Clarke, Daniel McCloy

Make average Source Time Course across all subjects, for each condition
"""

import os
from mayavi import mlab
import mne
from analysis.aux_functions import load_paths, load_params, load_cohorts

mlab.options.offscreen = True
mne.cuda.init_cuda()
overwrite = False

# load params
brain_plot_kwargs, movie_kwargs, subjects, cohort = load_params()
data_root, _, results_dir = load_paths()
groupavg_path = os.path.join(results_dir, 'group_averages')
mov_path = os.path.join(results_dir, 'movies')
for _dir in (groupavg_path, mov_path):
    if not os.path.isdir(_dir):
        os.makedirs(_dir, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars', 'aliens')
methods = ('dSPM',)  # dSPM, sLORETA, eLORETA


# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# assemble groups to iterate over
groups = dict(GrandAvg=subjects)
groups.update(intervention_group)
groups.update(letter_knowledge_group)

print(groups)

# loop over algorithms
for method in methods:
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        # loop over experimental conditions
        for cond in conditions:
            # loop over groups
            for group_name, group_members in groups.items():
                print(f'Working on group {group_name}.')
                avg = 0
                group = f'{group_name}N{len(group_members)}FSAverage'
                avg_fname = f'{group}_{prepost}Camp_{method}_{cond}'
                mov_fname = f'{avg_fname}.mov'
                # if the movie already exists, so must the STC, so skip both
                if (os.path.exists(os.path.join(mov_path, mov_fname)) and
                        not overwrite):
                    print(f'skipping {avg_fname}')
                    continue
                # we only compare incoming knowledge for pre-intervention data
                if prepost == 'post' and group_name.endswith('Knowledge'):
                    continue
                # make cross-subject average
                for s in group_members:
                    print(f'Adding subject {s} to group average.')
                    this_subj = os.path.join(data_root, f'{prepost}_camp',
                                             'twa_hp', 'erp', s)
                    fname = f'{s}FSAverage_{prepost}Camp_{method}_{cond}'
                    stc_path = os.path.join(this_subj, 'stc', fname)
                    avg += mne.read_source_estimate(stc_path)
                avg /= len(group_members)
                # save group average STCs
                avg.save(os.path.join(groupavg_path, avg_fname))
