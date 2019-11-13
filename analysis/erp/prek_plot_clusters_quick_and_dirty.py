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
cluster_dir = os.path.join(results_dir, 'clustering')
cluster_stc_dir = os.path.join(results_dir, 'clustering', 'stcs')
img_dir = os.path.join(results_dir, 'clustering', 'images')
for folder in (img_dir, cluster_stc_dir):
    if not os.path.isdir(folder):
        os.mkdir(folder)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# generate contrast pairs (need a list so we can use it twice)
contrasts = list(combinations(conditions, 2))

# the "group" name used in the contrast filenames
group = f'GrandAvgN{len(subjects)}FSAverage'


# workhorse function
def make_cluster_stc(group, prepost, method, con, results_dir,
                     results_subdir, cluster_dir, cluster_stc_dir, img_dir):
    """NB: 'con' can be either condition (str) or contrast (tuple)."""
    # we don't need to load the STC, but we need its name
    stc_fname = f'{group}_{prepost}Camp_{method}_{con}'
    # load the cluster results
    cluster_fname = f'{stc_fname}.npz'
    cluster_fpath = os.path.join(cluster_dir, cluster_fname)
    cluster_dict = np.load(cluster_fpath, allow_pickle=True)
    # KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters
    # We need to reconstruct the tuple that is output by the clustering
    # function:
    clu = (cluster_dict['tvals'], cluster_dict['clusters'],
           cluster_dict['pvals'], cluster_dict['hzero'])
    # convert into a quasi-STC object where the first timepoint shows
    # all clusters, each subsequent time point shows a single cluster,
    # and the colormap indicates the duration for which the cluster was
    # significant
    cluster_stc_fpath = os.path.join(cluster_stc_dir, f'{stc_fname}_clusters')
    has_signif_clusters = False
    try:
        cluster_stc = mne.stats.summarize_clusters_stc(clu)
        has_signif_clusters = True
    except RuntimeError:
        txt_path = os.path.join(
            img_dir, f'{stc_fname}_NO-SIGNIFICANT-CLUSTERS.txt')
        with open(txt_path, 'w') as f:
            f.write('no significant clusters')
    if has_signif_clusters:
        cluster_stc.save(cluster_stc_fpath)
        # plot the clusters
        brain = cluster_stc.plot(**brain_plot_kwargs)
        img_fpath = os.path.join(img_dir, f'{stc_fname}_clusters.png')
        brain.save_image(img_fpath)


# loop over algorithms
for method in methods:
    common_kwargs = dict(group=group, method=method, results_dir=results_dir,
                         cluster_dir=cluster_dir,
                         cluster_stc_dir=cluster_stc_dir, img_dir=img_dir)
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        # loop over contrasts
        for (cond_0, cond_1) in contrasts:
            contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
            make_cluster_stc(prepost=prepost, con=contr,
                             results_subdir='condition_contrasts',
                             **common_kwargs)
    # post-pre subtraction for single conditions
    for cond in conditions:
        make_cluster_stc(prepost='PostCampMinusPre', con=cond,
                         results_subdir='prepost_contrasts',
                         **common_kwargs)
    # post-pre subtraction for condition contrasts
    for (cond_0, cond_1) in contrasts:
        contr = f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
        make_cluster_stc(prepost='PostCampMinusPre', con=contr,
                         results_subdir='prepost_contrasts',
                         **common_kwargs)
