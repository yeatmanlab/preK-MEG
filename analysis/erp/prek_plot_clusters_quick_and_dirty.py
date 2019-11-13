#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot movies with significant cluster regions highlighted.
"""

import os
from itertools import combinations
import numpy as np
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10

# load params
brain_plot_kwargs, movie_kwargs, subjects = load_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
img_path = os.path.join(results_dir, 'clustering', 'images')
obj_path = os.path.join(results_dir, 'clustering', 'stcs')
for folder in (img_path, obj_path):
    if not os.path.isdir(folder):
        os.mkdir(folder)
cluster_dir = os.path.join(results_dir, 'clustering')

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# generate contrast pairs (need a list so we can use it twice)
contrasts = list(combinations(conditions, 2))

# the "group" name used in the contrast filenames
group = f'GrandAvgN{len(subjects)}FSAverage'


# loop over algorithms
for method in methods:
    condition_dict = dict()
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        # loop over condition
        for (cond_0, cond_1) in contrasts:
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            # load the STC
            stc_fname = f'{group}_{prepost}Camp_{method}_{contr}'
            stc_fpath = os.path.join(results_dir, 'condition_contrasts',
                                     stc_fname)
            stc = mne.read_source_estimate(stc_fpath)
            # load the cluster results
            cluster_fname = f'{stc_fname[:-4]}.npz'
            cluster_fpath = os.path.join(cluster_dir, cluster_fname)
            cluster_dict = np.load(cluster_fpath, allow_pickle=True)
            # KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters.
            # We need to reconstruct the tuple that is output by the clustering
            # function:
            clu = (cluster_dict['tvals'], cluster_dict['clusters'],
                   cluster_dict['pvals'], cluster_dict['hzero'])
            # cunvert into a quasi-STC object where the first timepoint shows
            # all clusters, each subsequent time point shows a single cluster,
            # and the colormap indicates the duration for which the cluster was
            # significant
            cluster_stc = mne.stats.summarize_clusters_stc(clu)
            cluster_stc.save(os.path.join(obj_path, f'{stc_fname}_clusters'))
            # plot the clusters
            brain = cluster_stc.plot(**brain_plot_kwargs)
            outfile = os.path.join(img_path, f'{stc_fname}_clusters.png')
            brain.save_image(outfile)
