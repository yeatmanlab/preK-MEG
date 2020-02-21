#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot movies with significant cluster regions highlighted.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from mayavi import mlab
import mne
from aux_functions import load_paths, load_params, get_dataframe_from_label

mlab.options.offscreen = True
mne.cuda.init_cuda()
n_jobs = 10
sns.set(style='whitegrid', font_scale=0.8)

# plot setup
show_all_conditions = True
show_all_timepoints = True
show_all_groups = True

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

title_dict = dict(language='Language Intervention cohort',
                  letter='Letter Intervention cohort',
                  grandavg='All participants',
                  lower='Pre-test lower half of participants',
                  upper='Pre-test upper half of participants')

# load fsaverage source space
fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                  'fsaverage-ico-5-src.fif')
fsaverage_src = mne.read_source_spaces(fsaverage_src_path)
hemi_nverts = len(fsaverage_src[0]['vertno'])


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

    # get the dataframe
    time_courses = get_dataframe_from_label(label, fsaverage_src, [method],
                                            timepoints, conditions)
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
        # plot the clusters (saves as PNG image)
        plot_clusters(stc, cluster_stc, signif_clu)
        # for each significant cluster, extract the mean (across vertices) time
        # series for each subject, save to a CSV, and plot alongside the
        # cluster location.
        for cluster_idx in signif_clu:
            # get time series and save it
            df = extract_time_courses(stc, cluster_fname, cluster_dict,
                                      cluster_idx)
            timeseries_fname = f'{avg_stc_fname}_cluster{cluster_idx:05}.csv'
            timeseries_fpath = os.path.join(timeseries_dir, timeseries_fname)
            df.to_csv(timeseries_fpath, index=False)

            # plot time series alongside cluster location
            n_rows = (3 if show_all_groups and 'grandavg' not in groups else
                      len(groups) + 1)
            gridspec_kw = dict(height_ratios=[4] + [1] * (n_rows - 1))
            fig, axs = plt.subplots(n_rows, 1, gridspec_kw=gridspec_kw,
                                    figsize=(9, 13))
            # draw the cluster brain image into first axes
            img_fname = re.sub(r'\.npz$', f'_cluster{cluster_idx:05}.png',
                               cluster_fname)
            cluster_image = imread(os.path.join(img_dir, img_fname))
            axs[0].imshow(cluster_image)
            axs[0].set_axis_off()
            axs[0].set_title(cluster_fname)
            # prepare to plot
            all_conditions = ('words', 'faces', 'cars')
            all_timepoints = ('post', 'pre')
            all_interventions = ('letter', 'language')
            all_pretest_cohorts = ('lower', 'upper')
            # missing_conditions = list(set(all_conditions) - set(conditions))
            # missing_timepoints = list(set(all_timepoints) - set(timepoints))
            plot_kwargs = dict(hue='condition', hue_order=all_conditions,
                               style='timepoint', style_order=all_timepoints)
            grey_vals = ['0.75', '0.55', '0.35']
            color_vals = ['#004488', '#bb5566', '#ddaa33']
            colors = [color_vals[i] if c in conditions else grey_vals[i]
                      for i, c in enumerate(all_conditions)]

            # triage group definition
            if groups[0] in all_interventions:
                group_column = df['intervention']
                this_groups = all_interventions
            elif groups[0] in all_pretest_cohorts:
                group_column = df['pretest']
                this_groups = all_pretest_cohorts
            else:
                group_column = np.full(df.shape[:1], 'grandavg')
                this_groups = ['grandavg']
            # missing_group = list(set(this_groups) - set(groups))
            # draw the timecourses
            for group, ax in zip(this_groups, axs[1:]):
                # # narrow down data as warranted (for gray lines)
                # conds_idx = (np.in1d(df['condition'], missing_conditions)
                #              if show_all_conditions else
                #              np.in1d(df['condition'], conditions))
                # times_idx = (np.in1d(df['timepoint'], missing_timepoints)
                #              if show_all_timepoints else
                #              np.in1d(df['timepoint'], timepoints))
                # group_idx = (np.in1d(group_column, missing_group)
                #              if show_all_groups else
                #              np.in1d(group_column, group))
                # # draw the uninteresting lines in gray
                # if (show_all_conditions or show_all_timepoints):
                #     data = df.loc[group_idx & times_idx & conds_idx]
                #     with sns.color_palette(grey_vals):
                #         grey_kwargs = plot_kwargs.copy()
                #         grey_kwargs.update(legend=False, size=0.4,
                #                            err_kws=dict(alpha=0.1))
                #         sns.lineplot(x='time', y='value', data=data, ax=ax,
                #                      **grey_kwargs)
                # # draw the relevant lines in color
                # data = df.loc[(group_column == group) &
                #               np.in1d(df['timepoint'], timepoints) &
                #               np.in1d(df['condition'], conditions)]
                # draw the relevant lines in color
                data = df.loc[(group_column == group)]
                with sns.color_palette(colors):
                    sns.lineplot(x='time', y='value', data=data, ax=ax,
                                 **plot_kwargs)
                # indicate temporal span of cluster signif. difference
                temporal_idxs, _ = cluster_dict['clusters'][cluster_idx]
                xmin = stc.times[temporal_idxs.min()]
                xmax = stc.times[temporal_idxs.max()]
                ax.fill_betweenx((0, 4), xmin, xmax, color='k', alpha=0.1)
                # garnish
                ax.set_ylim(0, 4)
                ax.set_title(title_dict[group])
                # we only need one legend, suppress on subsequent plots
                plot_kwargs.update(legend=False)
            # save plot (overwrites the cluster image PNG)
            sns.despine()
            fig.savefig(os.path.join(img_dir, img_fname))
            plt.close(fig)


cluster_fnames = sorted([x.name for x in os.scandir(cluster_dir)
                         if x.is_file()])
for cluster_fname in cluster_fnames:
    make_cluster_stc(cluster_fname)
