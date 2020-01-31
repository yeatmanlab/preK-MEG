#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot movies with significant cluster regions highlighted.
"""

import os
import re
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, load_cohorts

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10
sns.set()

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
timeseries_dir = os.path.join(cluster_dir, 'time-series')
for folder in (img_dir, cluster_stc_dir, timeseries_dir):
    os.makedirs(folder, exist_ok=True)

# load cohort info (keys Language/LetterIntervention and Lower/UpperKnowledge)
intervention_group, letter_knowledge_group = load_cohorts()

# assemble groups to iterate over
groups_dict = dict(GrandAvg=subjects)
groups_dict.update(intervention_group)
groups_dict.update(letter_knowledge_group)

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
hemi_nverts = len(fsaverage_src[0]['vertno'])


# helper function: reconstruct group dictionary key from filename
def build_group_key(group):
    if group in ('letter', 'language'):
        key = f'{group.capitalize()}Intervention'
    elif group in ('upper', 'lower'):
        key = f'{group.capitalize()}Knowledge'
    elif group == 'grandavg':
        key = 'GrandAvg'
    return key


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
        brain.save_image(os.path.join(img_dir, img_fname))


# helper function: get timeseries in label for a particular subject
def get_subj_time_course(subj, timept, method, con, label):
    this_subj = os.path.join(data_root, f'{timept}_camp', 'twa_hp', 'erp',
                             subj, 'stc')
    fname = f'{subj}FSAverage_{timept}Camp_{method}_{con}'
    stc_path = os.path.join(this_subj, fname)
    stc = mne.read_source_estimate(stc_path)
    # extract label time course
    kwargs = dict(src=fsaverage_src, mode='pca_flip', allow_empty=True)
    time_course = mne.extract_label_time_course(stc, label, **kwargs)
    return np.squeeze(time_course)


# helper function: extract condition names from filename
def get_condition_names(cluster_fname):
    # extract condition names from filenames
    avg_stc_fname = re.sub(r'(_[lr]h)?\.npz$', '', cluster_fname)
    group, timepoint, method, condition = avg_stc_fname.split('_')
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
def extract_time_courses(avg_stc, cluster_fname, cluster_dict, cluster_idx):
    """Extract the mean (across vertices) time series for each subject."""
    (groups, timepoints, method, conditions, hemi_str, avg_stc_fname
     ) = get_condition_names(cluster_fname)

    # make sure we're right about which hemisphere the cluster is in
    temporal_idxs, spatial_idxs = cluster_dict['clusters'][cluster_idx]
    if all(spatial_idxs <= hemi_nverts):
        assert hemi_str == 'lh'
        hemi = 0
    elif all(spatial_idxs >= hemi_nverts):
        assert hemi_str == 'rh'
        hemi = 1
        spatial_idxs = spatial_idxs - hemi_nverts
    else:
        err = ("you seem to have a cluster that spans "
               "hemispheres, this shouldn't happen.")
        raise RuntimeError(err)

    # select the vertices in the cluster & convert to Label
    verts = np.unique(avg_stc.vertices[hemi][spatial_idxs])
    label = mne.Label(verts, hemi=hemi_str, subject='fsaverage')
    label = label.restrict(fsaverage_src)

    time_courses = dict()
    # loop over conditions to handle subtractions
    for timept in timepoints:
        time_courses[timept] = dict()
        for con in conditions:
            time_courses[timept][con] = dict()
            for grp in groups:
                time_courses[timept][con][grp] = dict()
                group_key = build_group_key(grp)
                group_members = groups_dict[group_key]
                for s in group_members:
                    tc = get_subj_time_course(s, timept, method, con, label)
                    time_courses[timept][con][grp][s] = tc
                # convert dict of each subj's time series to DataFrame
                df = pd.DataFrame(time_courses[timept][con][grp],
                                  index=range(len(avg_stc.times)))
                df['time'] = avg_stc.times
                df['group'] = grp
                time_courses[timept][con][grp] = df
            # combine DataFrames across groups
            if len(groups) == 1:
                time_courses[timept][con] = time_courses[timept][con][grp]
            else:
                dfs = (time_courses[timept][con][g] for g in groups)
                time_courses[timept][con] = pd.concat(dfs)
            time_courses[timept][con]['condition'] = con
        # combine DataFrames across conditions
        if len(conditions) == 1:
            time_courses[timept] = time_courses[timept][con]
        else:
            dfs = (time_courses[timept][c] for c in conditions)
            time_courses[timept] = pd.concat(dfs)
        time_courses[timept]['timepoint'] = timept
    # combine DataFrames across timepoints
    if len(timepoints) == 1:
        time_courses = time_courses[timept]
    else:
        dfs = (time_courses[t] for t in timepoints)
        time_courses = pd.concat(dfs)
    # clean up
    time_courses.reset_index(inplace=True)
    return time_courses


# workhorse function
def make_cluster_stc(cluster_fname):
    (groups, timepoints, method, conditions, hemi_str, avg_stc_fname
     ) = get_condition_names(cluster_fname)
    # load the STC
    stc_fpath = os.path.join(results_dir, 'group_averages', avg_stc_fname)
    stc = mne.read_source_estimate(stc_fpath)
    # pick correct hemisphere(s)
    vertices = stc.vertices
    if cluster_fname.rstrip('.npz').endswith('_lh'):
        vertices[1] = np.array([])
    elif cluster_fname.rstrip('.npz').endswith('_rh'):
        vertices[0] = np.array([])
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
        # plot the clusters
        plot_clusters(stc, cluster_stc, signif_clu)
        # for each significant cluster, extract the mean (across vertices) time
        # series for each subject, save to a CSV, and plot alongside the
        # cluster location.
        for cluster_idx in signif_clu:
            timeseries_dataframe = extract_time_courses(stc, cluster_fname,
                                                        cluster_dict,
                                                        cluster_idx)
            # save to CSV
            timeseries_fname = f'{avg_stc_fname}_cluster{cluster_idx:05}.csv'
            timeseries_dataframe.to_csv(os.path.join(timeseries_dir,
                                                     timeseries_fname))
            # plot time series alongside cluster location
            all_cols = timeseries_dataframe.columns.values
            subj_cols = timeseries_dataframe.columns.str.startswith('prek')
            id_vars = all_cols[np.logical_not(subj_cols)]
            df = pd.melt(timeseries_dataframe, id_vars=id_vars,
                         var_name='subj')
            # intitialize figure
            n_rows = len(conditions) + 1
            gridspec_kw = dict(height_ratios=[2] + [1] * len(conditions))
            fig, axs = plt.subplots(n_rows, 1, gridspec_kw=gridspec_kw)
            # draw the cluster brain image into first axes
            img_fname = re.sub(r'\.npz$', f'_cluster{cluster_idx:05}.png',
                               cluster_fname)
            cluster_image = imread(os.path.join(img_dir, img_fname))
            axs[0].imshow(cluster_image)
            # draw the timecourses
            plot_func = sns.lineplot
            plot_kwargs = dict()
            if len(conditions) > 1:
                plot_func = partial(sns.relplot, kind='line', row='condition')
            if len(groups) > 1:
                plot_kwargs.update(hue='group')
            if len(timepoints) > 1:
                plot_kwargs.update(style='timepoint')
            p = plot_func(x='time', y='value', data=df, ax=axs[1:],
                          **plot_kwargs)
            # save plot
            p.savefig(os.path.join(img_dir, img_fname))


cluster_fnames = sorted([x.name for x in os.scandir(cluster_dir)
                         if x.is_file()])
for cluster_fname in cluster_fnames:
    make_cluster_stc(cluster_fname)
