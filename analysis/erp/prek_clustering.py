#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Perform spatiotemporal clustering on STCs.

Treat each subject as an observation of shape n_times × n_vertices, and stack
all observations for a given pair of conditions (e.g., "faces" vs. "words") to
perform parametric spatiotemporal clustering.
"""

import os
from functools import partial
from itertools import combinations
import numpy as np
import mne
from mne.stats import spatio_temporal_cluster_1samp_test
from aux_functions import load_paths, load_params, prep_cluster_stats

mne.cuda.init_cuda()
rng = np.random.RandomState(seed=15485863)  # the one millionth prime
n_jobs = 10
threshold = None  # or dict(start=0, step=0.2) for TFCE

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
# cluster results get different subfolders depending on threshold setting
cluster_dir = os.path.join(results_dir, 'clustering')
if isinstance(threshold, dict):
    cluster_dir = os.path.join(cluster_dir,
                               'tfce_{start}_{step}'.format_map(threshold))
elif threshold is not None:
    cluster_dir = os.path.join(cluster_dir, f'thresh_{threshold}')
if not os.path.isdir(cluster_dir):
    os.makedirs(cluster_dir, exist_ok=True)

# set cache dir
cache_dir = os.path.join(data_root, 'cache')
if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)
mne.set_cache_dir(cache_dir)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# generate contrast pairs (need list so can reuse)
contrasts = list(combinations(conditions, 2))

# the "group" name used in the contrast filenames
group = f'GrandAvgN{len(subjects)}FSAverage'

# load fsaverage source space to get connectivity matrix
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
conn_matrix = mne.spatial_src_connectivity(fsaverage_src)

# labels to exclude (medial wall), from 10.1016/j.neuroimage.2010.06.010
label_names = ('G_and_S_paracentral',      # 3
               'G_and_S_cingul-Ant',       # 6
               'G_and_S_cingul-Mid-Ant',   # 7
               'G_and_S_cingul-Mid-Post',  # 8
               'G_cingul-Post-dorsal',     # 9
               'G_cingul-Post-ventral',    # 10
               'G_front_sup',              # 16
               'G_oc-temp_med-Parahip',    # 23
               'G_precuneus',              # 30
               'G_rectus',                 # 31
               'G_subcallosal',            # 32
               'S_cingul-Marginalis',      # 46
               'S_pericallosal',           # 66
               'S_suborbital',             # 70
               'S_subparietal',            # 71
               )
regexp = '|'.join(label_names)
medial_wall = dict()
for hemi in ('lh', 'rh'):
    medial_wall[hemi] = mne.read_labels_from_annot(
        subject='fsaverage', parc='aparc.a2009s', hemi=hemi,
        subjects_dir=subjects_dir, regexp=regexp)
    assert len(medial_wall[hemi]) == len(label_names)
    # merge the labels using the sum(..., start) hack
    medial_wall[hemi] = sum(medial_wall[hemi][1:], medial_wall[hemi][0])
    assert len(medial_wall[hemi].name.split('+')) == len(label_names)
spatial_exclude = np.concatenate(
    [medial_wall['lh'].vertices,
     medial_wall['rh'].vertices + len(fsaverage_src[0]['vertno'])])

# prepare clustering function
one_samp_test = partial(spatio_temporal_cluster_1samp_test,
                        threshold=threshold,
                        n_permutations=1024,
                        connectivity=conn_matrix,
                        n_jobs=n_jobs,
                        seed=rng,
                        spatial_exclude=spatial_exclude,
                        buffer_size=1024)

# loop over algorithms
for method in methods:
    condition_dict = dict()
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        condition_dict[prepost] = dict()
        # load the individual subject STCs for each condition
        for cond in conditions:
            condition_dict[prepost][cond] = list()
            # loop over subjects
            for s in subjects:
                this_subj = os.path.join(data_root, f'{prepost}_camp',
                                         'twa_hp', 'erp', s, 'stc')
                stc_fname = f'{s}FSAverage_{prepost}Camp_{method}_{cond}'
                stc_fpath = os.path.join(this_subj, stc_fname)
                stc = mne.read_source_estimate(stc_fpath)
                stc_data = stc.data.transpose(1, 0)  # need (subj, time, space)
                condition_dict[prepost][cond].append(stc_data)
            condition_dict[prepost][cond] = \
                np.array(condition_dict[prepost][cond])

        # do the condition subtraction
        for (cond_0, cond_1) in contrasts:
            X = (condition_dict[prepost][cond_0] -
                 condition_dict[prepost][cond_1])
            cluster_results = one_samp_test(X)
            stats = prep_cluster_stats(cluster_results)
            # save clustering results
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            out_fname = f'{group}_{prepost}Camp_{method}_{contr}.npz'
            out_fpath = os.path.join(cluster_dir, out_fname)
            np.savez(out_fpath, **stats)

    # do the post-pre subtraction for single conditions
    for cond in conditions:
        X = condition_dict['post'][cond] - condition_dict['pre'][cond]
        cluster_results = one_samp_test(X)
        stats = prep_cluster_stats(cluster_results)
        # save clustering results
        out_fname = f'{group}_PostCampMinusPreCamp_{method}_{cond}.npz'
        out_fpath = os.path.join(cluster_dir, out_fname)
        np.savez(out_fpath, **stats)

    # do the post-pre subtraction for contrasts
    for (cond_0, cond_1) in contrasts:
        X = (condition_dict['post'][cond_0] - condition_dict['post'][cond_1] -
             (condition_dict['pre'][cond_0] - condition_dict['pre'][cond_1]))
        cluster_results = one_samp_test(X)
        stats = prep_cluster_stats(cluster_results)
        # save clustering results
        contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
        out_fname = f'{group}_PostCampMinusPreCamp_{method}_{contr}.npz'
        out_fpath = os.path.join(cluster_dir, out_fname)
        np.savez(out_fpath, **stats)
