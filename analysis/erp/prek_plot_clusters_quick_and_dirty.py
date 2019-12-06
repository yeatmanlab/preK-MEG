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
cluster_root = os.path.join(results_dir, 'clustering')
# get most current cluster subfolder
with open(os.path.join(cluster_root, 'most-recent-clustering.txt'), 'r') as f:
    cluster_dir = f.readline().strip()
cluster_stc_dir = os.path.join(cluster_dir, 'stcs')
img_dir = os.path.join(cluster_dir, 'images')
for folder in (img_dir, cluster_stc_dir):
    if not os.path.isdir(folder):
        os.mkdir(folder)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# generate contrast names
contrasts = [f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
             for (cond_0, cond_1) in combinations(conditions, 2)]

# the "group" name used in the contrast filenames
group = f'GrandAvgN{len(subjects)}FSAverage'


# workhorse function
def make_cluster_stc(group, prepost, method, con, results_dir,
                     results_subdir, cluster_dir, cluster_stc_dir, img_dir):
    """NB: 'con' can be either condition string ('faces') or contrast string
    ('FacesMinusCars')."""
    # load the STC
    stc_fname = f'{group}_{prepost}Camp_{method}_{con}'
    stc_fpath = os.path.join(results_dir, results_subdir, stc_fname)
    stc = mne.read_source_estimate(stc_fpath)
    vertices = stc.vertices
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
        cluster_stc = mne.stats.summarize_clusters_stc(clu, vertices=vertices,
                                                       tstep=stc.sfreq * 1000)
        has_signif_clusters = True
    except RuntimeError:
        txt_path = os.path.join(
            img_dir, f'{stc_fname}_NO-SIGNIFICANT-CLUSTERS.txt')
        with open(txt_path, 'w') as f:
            f.write('no significant clusters')
    if has_signif_clusters:
        cluster_stc.save(cluster_stc_fpath)
        # plot the clusters
        clim_dict = dict(kind='value', pos_lims=[0.5, 1, len(stc.times)])
        brain = cluster_stc.plot(**brain_plot_kwargs,
                                 smoothing_steps='nearest',
                                 clim=clim_dict,
                                 time_unit='ms',
                                 time_label='auto')
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
        for contrast in contrasts:
            make_cluster_stc(prepost=prepost, con=contrast,
                             results_subdir='condition_contrasts',
                             **common_kwargs)
    # post-pre subtraction for single conditions
    for condition in conditions:
        make_cluster_stc(prepost='PostCampMinusPre', con=condition,
                         results_subdir='prepost_contrasts',
                         **common_kwargs)
    # post-pre subtraction for condition contrasts
    for contrast in contrasts:
        make_cluster_stc(prepost='PostCampMinusPre', con=contrast,
                         results_subdir='prepost_contrasts',
                         **common_kwargs)
