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
from analysis.aux_functions import (load_paths, load_params, load_cohorts,
                                    get_dataframe_from_label,
                                    plot_label_and_timeseries)

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10
cohorts='all'

# load params
brain_plot_kwargs, _, subjects = load_params(cohorts=cohorts)
# config paths
data_root, subjects_dir, results_dir = load_paths(cohorts=cohorts)
cluster_root = os.path.join(results_dir, 'clustering')
# get most current cluster subfolder
with open(os.path.join(cluster_root, 'most-recent-clustering.txt'), 'r') as f:
    cluster_dir = f.readline().strip()
cluster_stc_dir = os.path.join(cluster_dir, 'stcs')
print('cluster directory: %s' % cluster_stc_dir)
img_dir = os.path.join(cluster_dir, 'images')
timeseries_dir = os.path.join(cluster_dir, 'time-series')
for folder in (img_dir, cluster_stc_dir, timeseries_dir):
    os.makedirs(folder, exist_ok=True)

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts(cohorts=cohorts)

# assemble groups info
if cohorts == 'r_only':
    groups_dict = dict(grandavg=subjects,
                   letter=intervention_group['LetterIntervention'],
                   lower=letter_knowledge_group['LowerKnowledge'],
                   upper=letter_knowledge_group['UpperKnowledge'])
else:
    groups_dict = dict(grandavg=subjects,
                       language=intervention_group['LanguageIntervention'],
                       letter=intervention_group['LetterIntervention'],
                       lower=letter_knowledge_group['LowerKnowledge'],
                       upper=letter_knowledge_group['UpperKnowledge'])
for group, members in groups_dict.items():
    groups_dict[group] = [f'prek_{n}' for n in members]

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)


# helper function: plot the clusters as pseudo-STC
def plot_clusters(stc, cluster_stc, signif_clu):
    """Plot the clusters.

    Each "time" in a cluster STC shows one significant cluster, except for time
    zero, which shows the sum of all significant clusters. So here we plot each
    "time" as a separate image (skipping time zero).
    """
    stc_tstep_ms = 1000 * stc.tstep  # in milliseconds
    stc_dur_ms = 1000 * (stc.times[-1] - stc.times[0])
    clim_dict = dict(kind='value', pos_lims=[0, stc_tstep_ms, stc_dur_ms])
    for time_idx, this_time in enumerate(cluster_stc.times):
        if time_idx == 0:
            continue
        brain = cluster_stc.plot(smoothing_steps='nearest',
                                 clim=clim_dict,
                                 time_unit='ms',
                                 time_label='temporal extent',
                                 initial_time=this_time,
                                 **brain_plot_kwargs)
        cluster_idx = signif_clu[time_idx - 1]
        img_fname = re.sub(r'\.npz$', f'_cluster{cluster_idx:05}.png',
                           cluster_fname)
        img_path = os.path.join(img_dir, img_fname)
        brain.save_image(img_path)
        return img_path


# helper function: extract condition names from filename
def get_condition_names(cluster_fname):
    # extract condition names from filenames
    print('Getting condition for stc %s.' % cluster_fname)
    avg_stc_fname = re.sub(r'(_[lr]h)?\.npz$', '', cluster_fname)
    _, group, timepoint, method, condition = avg_stc_fname.split('_')
    hemi_str = re.sub(r'\.npz$', '', cluster_fname)[-2:]
    # convert condition names as needed
    timepoints = dict(preCamp=['pre'], postCamp=['post'],
                      PostCampMinusPreCamp=['post', 'pre'])[timepoint]
    conditions = [x.lower() for x in condition.split('Minus')]
    groups = re.sub(r'(Intervention|Knowledge)$', '', group[:group.index('N')])
    groups = [x.lower() for x in groups.split('Minus')]
    return (groups, timepoints, method, conditions, hemi_str, avg_stc_fname)


# helper function: assemble DataFrame of timeseries for each subj in each
# condition pertinent to the current cluster result
def get_label_from_cluster(stc, src, hemi, cluster):
    """Extract the mean (across vertices) time series for each subject.

    Parameters
    ----------

    stc : instance of SourceEstimate

    src : instance of SourceSpace

    hemi : 'lh' | 'rh' | 'both'

    cluster : tuple
        length 2 tuple of array-like (temporal_indices, spatial_indices)

    Returns
    -------
    label : instance of Label
    """
    hemi_nverts = len(src[0]['vertno'])

    # make sure we're right about which hemisphere the cluster is in
    temporal_idxs, spatial_idxs = cluster
    if all(spatial_idxs <= hemi_nverts):
        assert hemi == 'lh'
    elif all(spatial_idxs >= hemi_nverts):
        assert hemi == 'rh'
        spatial_idxs = spatial_idxs - hemi_nverts
    else:
        err = ("you seem to have a cluster that spans "
               "hemispheres, this shouldn't happen.")
        raise RuntimeError(err)
    hemi_idx = 0 if hemi == 'lh' else 1
    # select the vertices in the cluster & convert to Label
    verts = np.unique(stc.vertices[hemi_idx][spatial_idxs])
    label = mne.Label(verts, hemi=hemi, subject='fsaverage')
    label = label.restrict(src)
    return label


# workhorse function
def make_cluster_stc(cluster_fname):
    (groups, timepoints, method, conditions, hemi, avg_stc_fname
     ) = get_condition_names(cluster_fname)
    # load the STC
    avg_stc_fpath = os.path.join(results_dir, 'group_averages', avg_stc_fname)
    avg_stc = mne.read_source_estimate(avg_stc_fpath)
    # pick correct hemisphere(s)
    vertices = avg_stc.vertices
    if cluster_fname.rstrip('.npz').endswith('_lh'):
        vertices[1] = np.array([])
    elif cluster_fname.rstrip('.npz').endswith('_rh'):
        vertices[0] = np.array([])
    stc_tstep_ms = 1000 * avg_stc.tstep  # in milliseconds
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
    has_signif_clusters = False
    try:
        cluster_stc = mne.stats.summarize_clusters_stc(clu, vertices=vertices,
                                                       tstep=stc_tstep_ms)
        has_signif_clusters = True
    except RuntimeError:
        txt_fname = cluster_fname.replace('.npz', '_NO-SIGNIFICANT-CLUSTERS.txt')  # noqa E501
        txt_fpath = os.path.join(img_dir, txt_fname)
        with open(txt_fpath, 'w') as f:
            f.write('no significant clusters')
    if has_signif_clusters:
        # save the quasi-STC
        cluster_stc.save(os.path.join(cluster_stc_dir, avg_stc_fname))
        # get indices for which clusters are significant
        signif_clu = cluster_dict['good_cluster_idxs'][0]
        # plot the clusters (saves as PNG image)
        cluster_img_path = plot_clusters(avg_stc, cluster_stc, signif_clu)
        # for each significant cluster, extract the mean (across vertices) time
        # series for each subject, save to a CSV, and plot alongside the
        # cluster location.
        for cluster_idx in signif_clu:
            # get label
            cluster = cluster_dict['clusters'][cluster_idx]
            label = get_label_from_cluster(avg_stc, fsaverage_src, hemi,
                                           cluster)
            # get dataframe
            all_conditions = ('words', 'faces', 'cars')
            all_timepoints = ('post', 'pre')
            df = get_dataframe_from_label(label, fsaverage_src, [method],
                                          all_timepoints, all_conditions)
            # plot
            lineplot_kwargs = dict(hue='condition', hue_order=all_conditions,
                                   style='timepoint',
                                   style_order=all_timepoints)
            plot_label_and_timeseries(label, cluster_img_path, df, method,
                                      groups, timepoints, conditions,
                                      all_timepoints, all_conditions,
                                      cluster, lineplot_kwargs)


cluster_fnames = sorted([x.name for x in os.scandir(cluster_dir)
                         if x.is_file() and 'dSPM' in x.name])
for cluster_fname in cluster_fnames:
    make_cluster_stc(cluster_fname)
