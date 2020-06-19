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
from analysis.aux_functions import load_paths, load_params

mlab.options.offscreen = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
in_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster')
stc_dir = os.path.join(in_dir, 'stcs')
img_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'cluster')
for _dir in (stc_dir, img_dir):
    os.makedirs(_dir, exist_ok=True)

precamp_fname = 'GrandAvg-pre_camp'
postcamp_fname = 'GrandAvg-post_camp'
median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

# load params
brain_plot_kwargs, _, subjects = load_params()

# config other
freqs = (2, 4, 6, 12)

# get some params from group-level STC
stc_fname = 'GrandAvg-pre_camp-pskt-5_sec-fft-snr'
stc = mne.read_source_estimate(os.path.join(results_dir, 'pskt', 'group-level',
                                            'stc', stc_fname))
all_freqs = stc.times
vertices = stc.vertices
stc_tstep_hz = stc.tstep  # in hertz
stc_dur = stc.times[-1] - stc.times[0]

for freq in freqs:
    for prefix in (precamp_fname, postcamp_fname, median_split_fname,
                   intervention_fname):
        # load the cluster results
        fname = f'{prefix}-{freq}_Hz-SNR-clusters.npz'
        fpath = os.path.join(in_dir, fname)
        cluster_dict = np.load(fpath, allow_pickle=True)
        # KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters
        # We need to reconstruct the tuple that is output by the clustering
        # function:
        clu = (cluster_dict['tvals'], cluster_dict['clusters'],
               cluster_dict['pvals'], cluster_dict['hzero'])
        # since this is TFCE clustering, don't bother with
        # mne.stats.summarize_clusters_stc, just stick the p-values into the
        # existing STC.
        stc.data = (1 - cluster_dict['pvals'])[:, np.newaxis]
        lims = (0.99, 0.999, 0.9999)
        clim_dict = dict(kind='value', lims=lims)
        # save the STC
        stc.save(os.path.join(stc_dir, fname.replace('.npz', '')))
        brain = stc.plot(smoothing_steps='nearest',
                         clim=clim_dict,
                         time_unit='s',
                         time_label=f'1 minus pvalue ({freq} Hz)',
                         initial_time=0,
                         **brain_plot_kwargs)
        img_fname = re.sub(r'\.npz$', '.png', fname)
        img_path = os.path.join(img_dir, img_fname)
        brain.save_image(img_path)

        ## convert into a quasi-STC object where the first timepoint shows
        ## all clusters, each subsequent time point shows a single cluster,
        ## and the colormap indicates the duration for which the cluster was
        ## significant
        #has_signif_clusters = False
        #try:
        #    cluster_stc = mne.stats.summarize_clusters_stc(
        #        clu, vertices=vertices, tstep=stc_tstep_hz)
        #    has_signif_clusters = True
        #except RuntimeError:
        #    txt_fname = fname.replace('.npz', '_NO-SIGNIFICANT-CLUSTERS.txt')
        #    txt_fpath = os.path.join(img_dir, txt_fname)
        #    with open(txt_fpath, 'w') as f:
        #        f.write('no significant clusters')
        #
        #if has_signif_clusters:
        #    # save the quasi-STC
        #    cluster_stc.save(os.path.join(stc_dir, fname.replace('.npz', '')))
        #    # get indices for which clusters are significant
        #    signif_clu = cluster_dict['good_cluster_idxs'][0]
        #    # plot the clusters (saves as PNG image)
        #    clim_dict = dict(kind='value', pos_lims=[0, stc_tstep_hz, stc_dur])
        #    for time_idx, this_time in enumerate(cluster_stc.times):
        #        if time_idx == 0:
        #            continue
        #        brain = cluster_stc.plot(smoothing_steps='nearest',
        #                                 clim=clim_dict,
        #                                 time_unit='s',
        #                                 time_label='frequency (Hz)',
        #                                 initial_time=this_time,
        #                                 **brain_plot_kwargs)
        #        cluster_idx = signif_clu[time_idx - 1]
        #        img_fname = re.sub(r's\.npz$', f'_{cluster_idx:05}.png', fname)
        #        img_path = os.path.join(img_dir, img_fname)
        #        brain.save_image(img_path)
