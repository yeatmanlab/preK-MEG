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
subj_root = '/mnt/scratch/prek/pre_camp/twa_hp/'
subjects_dir = '/mnt/scratch/prek/anat'
avg_out_path = os.path.join(subj_root, 'grand_averages')
mov_out_path = os.path.join(subj_root, 'movies')
for _dir in (avg_out_path, mov_out_path):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
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

# make grand average & movie
for cond in conditions:
    for method in methods:
        avg = 0
        # make cross-subject average
        for s in subjects:
            stc_path = os.path.join(subj_root, s, 'stc',
                                    f'{s}_{method}_fsaverage_{cond}-lh.stc')
            avg += mne.read_source_estimate(stc_path)
        avg /= len(subjects)
        # save
        avg_fname = f'fsaverage_{method}_{cond}_GrandAvgN{len(subjects)}.stc'
        avg.save(os.path.join(avg_out_path, avg_fname))
        # make movie
        brain = avg.plot(subject='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{avg_fname[:-4]}.mov'
        brain.save_movie(os.path.join(mov_out_path, mov_fname), **movie_kwargs)
