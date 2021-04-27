#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot clustering results for SSVEP data.
"""

import datetime
import os
import pathlib
import re
import numpy as np
import mne
from analysis.aux_functions import load_paths, load_params, load_inverse_params
import faulthandler
faulthandler.enable()

# load params
brain_plot_kwargs, _, subjects, cohort = load_params()
inverse_params = load_inverse_params()

# config paths
data_root, subjects_dir, results_dir = load_paths()
chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                      ).format_map(inverse_params)

cluster_dir = os.path.join(results_dir, 'pskt', 'group-level', 'cluster',
                           chosen_constraints)
stc_dir = os.path.join(cluster_dir, 'stcs')
img_dir = os.path.join(results_dir, 'pskt', 'group-level', 'fig', 'cluster',
                       chosen_constraints)
for _dir in (stc_dir, img_dir):
    os.makedirs(_dir, exist_ok=True)

precamp_fname = 'DataMinusNoise1samp-pre_camp'
postcamp_fname = 'DataMinusNoise1samp-post_camp'
median_split_fname = 'UpperVsLowerKnowledge-pre_camp'
intervention_fname = 'LetterVsLanguageIntervention-PostMinusPre_camp'

# config other
freqs = (2, 4, 6, 12)
conditions = ('all', 'ps', 'kt')

# get some params from group-level STC
stc_fname = f'{cohort}-GrandAvg-pre_camp-pskt-5_sec-all-fft-snr'
stc_fpath = os.path.join(results_dir, 'pskt', 'group-level', 'stc',
                         chosen_constraints, stc_fname)
stc = mne.read_source_estimate(stc_fpath)
all_freqs = stc.times
vertices = stc.vertices
stc_tstep_hz = stc.tstep  # in hertz
stc_dur = stc.times[-1] - stc.times[0]

for condition in conditions:
    print(f'{condition}')
    for freq in freqs:
        print(f'  {freq} Hz')
        for prefix in (precamp_fname, postcamp_fname, median_split_fname,
                       intervention_fname):
            # load the cluster results
            fname = f'{prefix}-{condition}-{freq}_Hz-SNR-clusters.npz'
            fpath = os.path.join(cluster_dir, fname)
            cluster_dict = np.load(fpath, allow_pickle=True)
            dt = datetime.datetime.fromtimestamp(
                pathlib.Path(fpath).stat().st_mtime)
            print(f'    {prefix.ljust(48)} (modified {dt})')
            # KEYS: clusters tvals pvals hzero good_cluster_idxs n_clusters
            # handle case of no signif. clusters
            assert len(cluster_dict['pvals']) == len(cluster_dict['clusters'])
            pvals = np.ones_like(cluster_dict['tvals'])
            if cluster_dict['n_clusters'] == 0:
                pass  # nothing to do
            elif cluster_dict['tfce']:
                assert isinstance(cluster_dict['threshold'][()], dict)
                pvals[:] = cluster_dict['pvals']
            else:
                assert not cluster_dict['tfce']
                assert not isinstance(
                    cluster_dict['threshold'], (dict, type(None)))
                pvals = np.ones_like(cluster_dict['tvals'])
                for cl, p in zip(cluster_dict['clusters'],
                                 cluster_dict['pvals']):
                    pvals[tuple(cl)] = p
            stc.data = (-np.log10(pvals) *
                        np.sign(np.atleast_2d(cluster_dict['tvals']))).T
            assert stc.data.shape == (20484, 1)
            lims = (1, 1.3, 5)  # equivalent to p=0.1, 0.05, 0.00001
            clim_dict = dict(kind='value', pos_lims=lims)
            # save the STC
            stc.save(os.path.join(stc_dir, fname.replace('.npz', '')))
            brain = stc.plot(smoothing_steps='nearest',
                             clim=clim_dict,
                             time_unit='s',
                             time_label=f'-np.log10(p) ({freq} Hz)',
                             initial_time=0,
                             **brain_plot_kwargs)
            img_fname = re.sub(r'\.npz$', '.png', fname)
            img_path = os.path.join(img_dir, img_fname)
            brain.save_image(img_path)
            brain.close()
            del brain
