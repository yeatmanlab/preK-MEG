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
frames_dir = os.path.join(cluster_dir, 'frames')
mov_dir = os.path.join(results_dir, 'movies', 'clustering')
if not os.path.isdir(mov_dir):
    os.mkdir(mov_dir)

# config other
conditions = ('words', 'faces', 'cars')  # we purposely omit 'aliens' here
methods = ('dSPM', 'sLORETA')  # dSPM, sLORETA, eLORETA

# generate contrast names
contrasts = [f'{cond_0.capitalize()}Minus{cond_1.capitalize()}'
             for (cond_0, cond_1) in combinations(conditions, 2)]

# the "group" name used in the contrast filenames
group = f'GrandAvgN{len(subjects)}FSAverage'

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)


# workhorse function
def make_cluster_movie(group, prepost, method, con, results_dir,
                       results_subdir, cluster_dir, mov_dir):
    """NB: 'con' can be either condition (str) or contrast (tuple)."""
    # load the STC
    stc_fname = f'{group}_{prepost}Camp_{method}_{con}'
    stc_fpath = os.path.join(results_dir, results_subdir, stc_fname)
    stc = mne.read_source_estimate(stc_fpath)
    # load the cluster results
    cluster_fname = f'{stc_fname}.npz'
    cluster_fpath = os.path.join(cluster_dir, cluster_fname)
    cluster_dict = np.load(cluster_fpath, allow_pickle=True)
    # keys: clusters tvals pvals hzero good_cluster_idxs n_clusters
    signif_clu = cluster_dict['good_cluster_idxs'][0]
    # prepare output directory
    this_frames_dir = os.path.join(frames_dir, f'{stc_fname}_frames')
    os.makedirs(this_frames_dir, exist_ok=True)
    # plot the brain
    brain = stc.plot(subject='fsaverage', **brain_plot_kwargs)
    # loop over time points
    for t_idx, time in enumerate(stc.times):
        brain.set_time(time)
        # loop over clusters
        for clu in signif_clu:
            temporal_idxs, spatial_idxs = cluster_dict['clusters'][clu]
            # only draw it if it's happening at the current time point
            if t_idx in temporal_idxs:
                # figure out which hemisphere the cluster is in
                if all(spatial_idxs <= len(stc.vertices[0])):
                    hemi = 0
                elif all(spatial_idxs > len(stc.vertices[0])):
                    hemi = 1
                    spatial_idxs = spatial_idxs - len(stc.vertices[0])
                else:
                    err = ("you seem to have a cluster that spans "
                           "hemispheres, this shouldn't happen.")
                    raise RuntimeError(err)
                # select the vertices in the cluster & convert to Label
                verts = np.unique(stc.vertices[hemi][spatial_idxs])
                hemi_str = ('lh', 'rh')[hemi]
                label = mne.Label(verts, hemi=hemi_str,
                                  subject=stc.subject)
                # fill in verts that are surrounded by cluster verts
                label = label.fill(fsaverage_src)
                brain.add_label(label, borders=True, color='m')
        frame_fname = f'{stc_fname}_{t_idx:03}.png'
        brain.save_image(os.path.join(this_frames_dir, frame_fname))
        brain.remove_labels()


# loop over algorithms
for method in methods:
    common_kwargs = dict(group=group, method=method, results_dir=results_dir,
                         cluster_dir=cluster_dir, mov_dir=mov_dir)
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        # loop over contrasts
        for contrast in contrasts:
            make_cluster_movie(prepost=prepost, con=contrast,
                               results_subdir='condition_contrasts',
                               **common_kwargs)
    # post-pre subtraction for single conditions
    for condition in conditions:
        make_cluster_movie(prepost='PostCampMinusPre', con=condition,
                           results_subdir='prepost_contrasts',
                           **common_kwargs)
    # post-pre subtraction for condition contrasts
    for contrast in contrasts:
        make_cluster_movie(prepost='PostCampMinusPre', con=contrast,
                           results_subdir='prepost_contrasts',
                           **common_kwargs)
