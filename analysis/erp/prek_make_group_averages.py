#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maggie Clarke, Daniel McCloy

Make average Source Time Course across all subjects, for each condition
"""

import os
import yaml
from functools import partial
import mne

# config paths
project_root = '/mnt/scratch/prek'
subjects_dir = os.path.join(project_root, 'anat')
avg_out_path = os.path.join(project_root, 'results', 'group_averages')
mov_out_path = os.path.join(project_root, 'results', 'movies')
for _dir in (avg_out_path, mov_out_path):
    if not os.path.isdir(_dir):
        os.makedirs(_dir, exist_ok=True)

# config other
conditions = ('words', 'faces', 'cars', 'aliens')
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# load params
paramdir = os.path.join('..', '..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)
with open(os.path.join(paramdir, 'brain_plot_params.yaml'), 'r') as f:
    brain_plot_kwargs = yamload(f)
with open(os.path.join(paramdir, 'movie_params.yaml'), 'r') as f:
    movie_kwargs = yamload(f)
with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
    subjects = yamload(f)
with open(os.path.join(paramdir, 'skip_subjects.yaml'), 'r') as f:
    skips = yamload(f)
subjects = sorted(set(subjects) - set(skips))

# make group averages & movies
group = f'GrandAvgN{len(subjects)}FSAverage'
# loop over pre/post measurement time
for prepost in ('pre', 'post'):
    # loop over experimental conditions
    for cond in conditions:
        # loop over algorithms
        for method in methods:
            avg = 0
            # make cross-subject average
            for s in subjects:
                this_subj = os.path.join(project_root, f'{prepost}_camp',
                                         'twa_hp', s)
                stc_path = os.path.join(
                    this_subj, 'stc',
                    f'{s}FSAverage_{prepost}Camp_{method}_{cond}-lh.stc')
                avg += mne.read_source_estimate(stc_path)
            avg /= len(subjects)
            # save
            avg_fname = f'{group}_{prepost}Camp_{method}_{cond}.stc'
            avg.save(os.path.join(avg_out_path, avg_fname))
            # make movie
            brain = avg.plot(subject='fsaverage', **brain_plot_kwargs)
            mov_fname = f'{avg_fname[:-4]}.mov'
            brain.save_movie(os.path.join(mov_out_path, mov_fname),
                             **movie_kwargs)
