#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot FFT and SNR spectra for a given label, for various splits of the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, FancyArrowPatch
import pandas as pd
import seaborn as sns

from sswef_helpers.aux_functions import load_paths, yamload


def nice_ticklabels(ticks):
    return list(map(str, [int(t) if t == int(t) else t for t in ticks]))


# config paths
data_root, subjects_dir, results_dir = load_paths()
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')
stats_dir = '../stats/'
timeseries_dir = stats_dir  # TODO: local (not colchuck)

# load data
method = 'dSPM'  # ('dSPM', 'MNE')
region = 'MPM_IOS_IOG_pOTS_lh'
fname = f'roi-{region}-timeseries-long.csv'
df = pd.read_csv(os.path.join(timeseries_dir, fname), index_col=0)
df = df.query(f'method=="{method}"')
# load temporal ROI
cluster_results_path = os.path.join(
    stats_dir, f'signif-spans-{region[4:]}-words_minus_cars.yml')
with open(cluster_results_path, 'r') as f:
    cluster_based_temporal_roi = np.squeeze(yamload(f))
# with open('peak-of-grand-mean.yaml', 'r') as f:
#     peak_properties = yamload(f)
# temporal_roi = np.array(peak_properties['temporal_roi'])
a_priori_temporal_roi = np.array([0.135, 0.235])
temporal_rois = {'a priori': a_priori_temporal_roi,
                 'cluster-based': cluster_based_temporal_roi}

# config plot
sns.set_style('white')
# medium-contrast pallete from
# https://personal.sron.nl/~pault/#fig:scheme_mediumcontrast
colors = dict(words=('#EE99AA',   # light red
                     '#994455'),  # dark red
              faces=('#6699CC',   # light blue
                     '#004488'),  # dark blue
              cars=('#DDAA33',    # yellow (subbed for light yellow #EECC66)
                    '#997700'))   # dark yellow

interventions = ('letter', 'language')
conditions = ('words', 'faces', 'cars')
timepoints = ('pre', 'post')
ymax = dict(MNE=1e-10, dSPM=3, sLORETA=2, fft=300, snr=8)[method]
bar_positions = [1.125, 1.375]
extratick = 1.25
extraticklabel = 'μ±SEM\nin window'
yticks = np.linspace(0, ymax, num=4)
xticks = np.linspace(0, 1, num=5)
yticklabels = nice_ticklabels(yticks)
xticklabels = nice_ticklabels(xticks) + [extraticklabel]
xticks = np.hstack([xticks, extratick])

# init figure
linefig = plt.figure(figsize=(6.5, 4))
line_ci_alpha = 0.25

# do lineplots
axs = linefig.subplots(2, 3, sharex=True, sharey=True)
for this_intervention, row in zip(interventions, axs):
    for this_condition, ax in zip(conditions, row):
        spec = ax.get_subplotspec()
        this_colors = colors[this_condition]
        this_df = df[(df['condition'] == this_condition) &
                     (df['intervention'] == this_intervention)]
        with sns.color_palette(this_colors):
            sns.lineplot(x='time', y='value', errorbar='se', hue='timepoint',
                         style='timepoint', hue_order=timepoints,
                         style_order=timepoints[::-1], ax=ax, data=this_df,
                         legend=False, linewidth=1,
                         err_kws=dict(alpha=line_ci_alpha, edgecolor='none'))
        # show temporal ROI
        for kind, t_roi in temporal_rois.items():
            kwargs = (
                dict(facecolor='k', alpha=0.1, edgecolor='none')
                if kind == 'a priori' else
                dict(facecolor='none', edgecolor='0.5', linewidth=0.8,
                     linestyle=':')
            )
            text_kwargs = dict(x=t_roi[1] + 0.02, s=f'{kind} window', size=6,
                               transform=ax.get_xaxis_transform(),
                               va='baseline')
            # extend the shading above/below axes so we can label the spans
            if spec.is_last_col():
                kwargs.update(clip_on=False)
                if spec.is_first_row() and kind == 'a priori':
                    ymin = -0.075
                    window_text = ax.text(y=ymin, **text_kwargs,
                                          weight='black', color='0.75')
                    kwargs.update(ymin=ymin)
                elif spec.is_last_row() and kind == 'cluster-based':
                    window_text = ax.text(y=1.025, **text_kwargs)
                    kwargs.update(ymax=1.075)
            ax.axvspan(*t_roi, zorder=-2, **kwargs)
            # add barplot with SEs
            agg_df = (this_df.loc[(this_df['time'] >= t_roi[0]) &
                                  (this_df['time'] <= t_roi[1])]
                             .drop(['method', 'pretest', 'roi', 'time'],
                                   axis=1)  # avoid `numeric_only` warning
                             .groupby(['subj', 'timepoint', 'condition'])
                             .agg(dict(value='mean',
                                       intervention=lambda x: x.unique()[0]))
                             .reset_index()
                             .filter(['timepoint', 'value']))
            height = (agg_df.groupby('timepoint')
                            .agg('mean')
                            .loc[list(timepoints), 'value']
                            .values)
            yerr = (
                agg_df.groupby('timepoint')
                      .agg(lambda x: np.std(x, ddof=1) / np.sqrt(x.shape[0]))
                      .loc[list(timepoints), 'value']
                      .values
            )
            bar_x = np.array([-0.05, 0.05])
            width = np.diff(bar_x) * 0.8
            bar_kwargs = dict(height=height, width=width, yerr=yerr,
                              ecolor=this_colors)
            if kind == 'a priori':
                color = to_rgba_array(this_colors)
                color += np.ones_like(color)
                color /= 2
                bar_kwargs.update(color=color, linewidth=0)
            else:
                bar_kwargs.update(color='w', edgecolor=this_colors,
                                  linestyle=':', linewidth=0.8,
                                  capstyle='butt', joinstyle='miter')
            bar_x += bar_positions[0 if kind == 'a priori' else 1]
            ax.bar(x=bar_x, **bar_kwargs)
            # add brackets
            if spec.is_last_col():
                connstyle = 'angle,angleA=0,angleB=90,rad=1'
                arrow_kwargs = dict(
                    connectionstyle=connstyle, arrowstyle='-[',
                    clip_on=False, transform=ax.get_xaxis_transform(),
                    color='0.6', linewidth=0.5, mutation_scale=6, shrinkB=0,
                )
                if spec.is_first_row() and kind == 'a priori':
                    posA = (0.9, -0.055)
                    posB = (bar_x.mean(), -0.025)
                    bracket = FancyArrowPatch(posA=posA, posB=posB,
                                              **arrow_kwargs)
                    ax.add_patch(bracket)
                elif spec.is_last_row() and kind == 'cluster-based':
                    posA = (1.025, 1.045)
                    posB = (bar_x.mean(), 1.025)
                    bracket = FancyArrowPatch(posA=posA, posB=posB,
                                              **arrow_kwargs)
                    ax.add_patch(bracket)

        # garnish
        ax.set(xlabel='', ylabel='',
               xlim=(-0.1, bar_positions[1] + 1.5 * width),
               ylim=(0, ymax),
               yticks=yticks, yticklabels=yticklabels,
               xticks=xticks, xticklabels=xticklabels)
        ax.xaxis.set_tick_params(labelsize=8)
        ax.grid(color='k', alpha=0.1, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('0.5')
        # titles
        if spec.is_first_col():
            ax.set_title(f'{this_intervention.capitalize()} Intervention',
                         loc='left', size='large')
        if spec.is_first_row():
            ax.text(x=0.95, y=0.95, s=this_condition.capitalize(),
                    ha='right', va='top', transform=ax.transAxes,
                    color=this_colors[1])

linefig.supxlabel('time (s)')
linefig.supylabel('dSPM value')
linefig.subplots_adjust(left=0.1, bottom=0.12, right=0.97, top=0.9,
                        hspace=0.3, wspace=0.1)

polygon_kwargs = dict(alpha=line_ci_alpha, linewidth=0)
line_kwargs = dict(linewidth=1, dash_capstyle='butt', solid_capstyle='butt')
legend_artists = {
    'pre-intervention': (
        Polygon(np.empty(shape=(0, 2)), color='#B4B4B4', **polygon_kwargs),
        Line2D([], [], color='0.6', linestyle='--', **line_kwargs)
    ),
    'post-intervention': (
        Polygon(np.empty(shape=(0, 2)), color='#5F5F5F', **polygon_kwargs),
        Line2D([], [], color='k', **line_kwargs)
    )
}
axs[0, -1].legend(handles=legend_artists.values(),
                  labels=legend_artists.keys(), ncol=2, loc='lower right',
                  frameon=True, bbox_to_anchor=(1.04, 1.02))

# save
linefig.savefig('lineplot-grid.png')
linefig.savefig('lineplot-grid.pdf', dpi=400)

# DIFFERENCE WAVES
# init figure
difffig = plt.figure(figsize=(6.5, 4))
xticks = np.linspace(0, 1, num=5)
yticks = np.linspace(-1, 1, num=5)
xticklabels = nice_ticklabels(xticks)
yticklabels = nice_ticklabels(yticks)

# do lineplots
axs = difffig.subplots(2, 3, sharex=True, sharey=True)
for this_intervention, row in zip(interventions, axs):
    for this_condition, ax in zip(conditions, row):
        spec = ax.get_subplotspec()
        this_colors = colors[this_condition]
        this_df = df[(df['condition'] == this_condition) &
                     (df['intervention'] == this_intervention)]
        # create difference waves
        diff_df = this_df.pivot(
            columns='timepoint',
            values='value',
            index=['time', 'condition', 'subj', 'intervention']
        )
        diff_df['value'] = diff_df['post'] - diff_df['pre']
        diff_df = (diff_df.drop(['pre', 'post'], axis='columns')
                          .reset_index()
                          .rename_axis(None, axis='columns'))
        with sns.color_palette(this_colors[1:]):
            sns.lineplot(x='time', y='value', errorbar='se', hue='condition',
                         ax=ax, data=diff_df, legend=False, linewidth=1,
                         err_kws=dict(alpha=line_ci_alpha, edgecolor='none'))
        # show temporal ROI
        for kind, t_roi in temporal_rois.items():
            kwargs = (
                dict(facecolor='k', alpha=0.1, edgecolor='none')
                if kind == 'a priori' else
                dict(facecolor='none', edgecolor='0.5', linewidth=0.8,
                     linestyle=':')
            )
            text_kwargs = dict(x=t_roi[1] + 0.02, s=f'{kind} window', size=6,
                               transform=ax.get_xaxis_transform(),
                               va='baseline')
            # extend the shading above/below axes so we can label the spans
            if spec.is_last_col():
                kwargs.update(clip_on=False)
                if spec.is_first_row() and kind == 'a priori':
                    ymin = -0.075
                    window_text = ax.text(y=ymin, **text_kwargs,
                                          weight='black', color='0.75')
                    kwargs.update(ymin=ymin)
                elif spec.is_last_row() and kind == 'cluster-based':
                    window_text = ax.text(y=1.025, **text_kwargs)
                    kwargs.update(ymax=1.075)
            ax.axvspan(*t_roi, zorder=-2, **kwargs)
        # garnish
        ax.set(xlabel='', ylabel='', xlim=(-0.1, 1), ylim=(-1, 1),
               yticks=yticks, yticklabels=yticklabels,
               xticks=xticks, xticklabels=xticklabels)
        ax.axhline(color='0.5', linewidth=1, zorder=-1)
        # ax.xaxis.set_tick_params(labelsize=8)
        ax.grid(color='k', alpha=0.1, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('0.5')
        # titles
        if spec.is_first_col():
            ax.set_title(f'{this_intervention.capitalize()} Intervention',
                         loc='left', size='large')
        if spec.is_first_row():
            ax.text(x=0.95, y=0.95, s=this_condition.capitalize(),
                    ha='right', va='top', transform=ax.transAxes,
                    color=this_colors[1])

difffig.supxlabel('time (s)')
difffig.supylabel('post- minus pre-intervention\ndSPM value (within-subect)')
difffig.subplots_adjust(left=0.14, bottom=0.12, right=0.97, top=0.9,
                        hspace=0.3, wspace=0.1)
# save
difffig.savefig('difference-waves-lineplot-grid.png')
difffig.savefig('difference-waves-lineplot-grid.pdf', dpi=400)
