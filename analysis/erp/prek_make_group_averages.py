#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maggie Clarke, Daniel McCloy

Make average Source Time Course across all subjects, for each condition
"""

import os
import yaml
import mne

# config paths
subj_root = '/mnt/scratch/prek/pre_camp/twa_hp/'
avg_out_path = os.path.join(subj_root, 'grand_averages')
mov_out_path = os.path.join(subj_root, 'movies')
for _dir in (avg_out_path, mov_out_path):
    if not os.path.isdir(_dir):
        os.mkdir(_dir)
# config other
conditions = ('words', 'faces', 'cars', 'aliens')
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA
brain_plot_kwargs = dict(views=['lat', 'med', 'ven'], hemi='split',
                         size=(1000, 800), subjects_dir=subjects_dir,
                         colormap='cool')
# load subjects
with open(os.path.join('..', '..', 'subjects.yaml'), 'r') as f:
    subjects = yaml.load(f, Loader=yaml.FullLoader)

# make grand average & movie
avg = 0

for cond in conditions:
    for method in methods:
        # make cross-subject average
        for s in subjects:
            stc_path = os.path.join(subj_root, s, 'stc',
                                    f'{s}_{method}_fsaverage_{cond}-lh.stc')
            avg += mne.read_source_estimate(stc_path)
        avg /= len(subjects)
        # save
        avg_fname = f'fsaverage_{method}_{cond}_GrandAvg_N{len(subjects)}.stc'
        avg.save(os.path.join(avg_out_path, avg_fname))
        # make movie
        brain = avg.plot(subject ='fsaverage', **brain_plot_kwargs)
        mov_fname = f'{avg_fname[:-4]}.mov'
        brain.save_movie(os.path.join(mov_out_path, mov_fname),
                         framerate=30, time_dilation=25,
                         interpolation='linear')
