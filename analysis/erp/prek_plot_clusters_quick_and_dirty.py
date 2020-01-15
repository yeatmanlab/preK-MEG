#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot movies with significant cluster regions highlighted.
"""

import os
import re
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


def make_cluster_stc(cluster_fname):
    # load the STC
    stc_fname = re.sub(r'(_[lr]h)?\.npz$', '', cluster_fname)
    stc_fpath = os.path.join(results_dir, 'group_averages', stc_fname)
    stc = mne.read_source_estimate(stc_fpath)
    # pick correct hemisphere(s)
    if cluster_fname.rstrip('.npz').endswith('_lh'):
        vertices = stc.vertices[0]
    elif cluster_fname.rstrip('.npz').endswith('_rh'):
        vertices = stc.vertices[1]
    else:
        vertices = stc.vertices
    stc_tstep_ms = 1000 * stc.tstep  # in milliseconds
    # load the cluster results
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
    cluster_stc_fpath = os.path.join(cluster_stc_dir, stc_fname + '.stc')
    has_signif_clusters = False
    try:
        cluster_stc = mne.stats.summarize_clusters_stc(clu, vertices=vertices,
                                                       tstep=stc_tstep_ms)
        has_signif_clusters = True
    except RuntimeError:
        txt_fname = stc_fname.replace('.stc', '_NO-SIGNIFICANT-CLUSTERS.txt')
        txt_fpath = os.path.join(img_dir, txt_fname)
        with open(txt_fpath, 'w') as f:
            f.write('no significant clusters')
    if has_signif_clusters:
        cluster_stc.save(cluster_stc_fpath)
        # plot the clusters
        stc_dur_ms = 1000 * (stc.times[-1] - stc.times[0])
        clim_dict = dict(kind='value', pos_lims=[0, stc_tstep_ms, stc_dur_ms])
        brain = cluster_stc.plot(smoothing_steps='nearest',
                                 clim=clim_dict,
                                 time_unit='ms',
                                 time_label='temporal extent',
                                 **brain_plot_kwargs)
        img_fname = cluster_fname.replace('.npz', '_clusters.png')
        brain.save_image(os.path.join(img_dir, img_fname))


cluster_fnames = sorted(os.listdir(cluster_dir))
for cluster_fname in cluster_fnames:
    make_cluster_stc(cluster_fname)
