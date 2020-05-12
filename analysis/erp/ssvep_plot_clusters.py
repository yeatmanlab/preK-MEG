#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot clustering results for SSVEP data.
"""

import os
import re
import numpy as np
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params

mlab.options.offscreen = True

# config paths
_, subjects_dir, results_dir = load_paths()
group_dir = os.path.join(results_dir, 'pskt', 'group-level')
in_dir = os.path.join(group_dir, 'cluster')
stc_dir = os.path.join(in_dir, 'stcs')
img_dir = os.path.join(group_dir, 'fig', 'cluster', 'brain')
for _dir in (stc_dir, img_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
brain_plot_kwargs, _, subjects = load_params()

# config other
freqs = (2, 4, 6)
hemi = 'lh'

# get some params from group-level STC
fname = f'GrandAvg-pre_camp-pskt-5_sec-fft-snr'
stc = mne.read_source_estimate(os.path.join(group_dir, 'stc', fname))
# pick correct hemisphere(s)
vertices = stc.vertices
if hemi == 'lh':
    vertices[1] = np.array([])
elif hemi == 'rh':
    vertices[0] = np.array([])
stc_tstep_hz = stc.tstep  # in hertz
stc_dur = stc.times[-1] - stc.times[0]
del stc

for freq in freqs:
    median_split_fname = f'LowerVsUpperKnowledge-pre_camp-{freq}_Hz-SNR-{hemi}.npz'  # noqa E501
    intervention_fname = f'LetterVsLanguageIntervention-PostMinusPre_camp-{freq}_Hz-SNR-{hemi}.npz'  # noqa E501
    for fname in (median_split_fname, intervention_fname):
        # load the cluster results
        fpath = os.path.join(in_dir, fname)
        cluster_dict = np.load(fpath, allow_pickle=True)
        # KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters
        # We need to reconstruct the tuple that is output by the clustering
        # function:
        clu = (cluster_dict['tvals'], cluster_dict['clusters'],
               cluster_dict['pvals'], cluster_dict['hzero'])
        # convert into a quasi-STC object where the first timepoint shows
        # all clusters, each subsequent time point shows a single cluster,
        # and the colormap indicates the duration for which the cluster was
        # significant
        has_signif_clusters = False
        try:
            cluster_stc = mne.stats.summarize_clusters_stc(
                clu, vertices=vertices, tstep=stc_tstep_hz)
            has_signif_clusters = True
        except RuntimeError:
            txt_fname = fname.replace('.npz', '_NO-SIGNIFICANT-CLUSTERS.txt')
            txt_fpath = os.path.join(img_dir, txt_fname)
            with open(txt_fpath, 'w') as f:
                f.write('no significant clusters')

        if has_signif_clusters:
            # save the quasi-STC
            cluster_stc.save(os.path.join(stc_dir, fname.replace('.npz', '')))
            # get indices for which clusters are significant
            signif_clu = cluster_dict['good_cluster_idxs'][0]
            # plot the clusters (saves as PNG image)
            clim_dict = dict(kind='value', pos_lims=[0, stc_tstep_hz, stc_dur])
            for time_idx, this_time in enumerate(cluster_stc.times):
                if time_idx == 0:
                    continue
                brain = cluster_stc.plot(smoothing_steps='nearest',
                                         clim=clim_dict,
                                         time_unit='s',
                                         time_label='frequency (Hz)',
                                         initial_time=this_time,
                                         **brain_plot_kwargs)
                cluster_idx = signif_clu[time_idx - 1]
                img_fname = re.sub(r'\.npz$', f'_cluster{cluster_idx:05}.png',
                                   fname)
                img_path = os.path.join(img_dir, img_fname)
                brain.save_image(img_path)
