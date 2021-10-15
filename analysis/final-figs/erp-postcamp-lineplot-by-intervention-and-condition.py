#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot FFT and SNR spectra for a given label, for various splits of the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from analysis.aux_functions import load_paths, yamload


def nice_ticklabels(ticks):
    return list(map(str, [int(t) if t == int(t) else t for t in ticks]))


# config paths
data_root, subjects_dir, results_dir = load_paths()
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')

# load data
method = 'dSPM'  # ('dSPM', 'MNE')
region = 'MPM_IOS_IOG_pOTS_lh'
fname = f'roi-{region}-timeseries-long.csv'
df = pd.read_csv(os.path.join(timeseries_dir, fname), index_col=0)
df = df.query(f'method=="{method}"')
# load temporal ROI
with open('peak-of-grand-mean.yaml', 'r') as f:
    peak_properties = yamload(f)
temporal_roi = np.array(peak_properties['temporal_roi'])

# config plot
sns.set_style('white')
colors = ('#994455',  # dark red
          '#004488',  # dark blue
          '#997700')  # dark yellow

interventions = ('letter', 'language')
conditions = ('words', 'faces', 'cars')
timepoints = ('pre', 'post')
ymax = dict(MNE=1e-10, dSPM=3, sLORETA=2, fft=300, snr=8)[method]
yticks = np.linspace(0, ymax, num=4)
yticklabels = nice_ticklabels(yticks)
# do custom xticks because of ROI
xticks = np.linspace(0, 1, num=5)
xticklabels = nice_ticklabels(xticks)

# init figure
fig = plt.figure(figsize=(6.5, 3.25))
brainfig, linefig = fig.subfigures(1, 2, width_ratios=(1, 4), wspace=0.1)
axs = fig.subplots(1, 2, sharex=True, sharey=True)

# add brain ROI
img_path = os.path.join('.', 'MPM_IOS_IOG_pOTS_lh.png')
brainax = brainfig.subplots()
brain_image = imread(img_path)
brainax.imshow(brain_image, interpolation=None)
brainax.set_axis_off()
brainfig.subplots_adjust(bottom=0, top=1, left=0, right=1)

# add timecourses
axs = linefig.subplots(1, 2, sharex=True, sharey=True)
for this_intervention, ax in zip(interventions, axs):
    this_df = df[(df['method'] == method) &
                 (df['timepoint'] == 'post') &
                 (df['intervention'] == this_intervention)]
    with sns.color_palette(colors):
        sns.lineplot(x='time', y='value', ci=68, hue='condition',
                     hue_order=conditions, ax=ax, data=this_df,
                     legend=False, linewidth=1,
                     err_kws=dict(alpha=0.25, edgecolor='none'))
    # fill temporal ROI
    ax.axvspan(*temporal_roi, facecolor='k', alpha=0.1, zorder=-2,
               edgecolor='none')
    # garnish
    ax.set(xlabel='', ylabel='', xlim=(-0.1, 1), ylim=(0, ymax),
           yticks=yticks, yticklabels=yticklabels,
           xticks=xticks, xticklabels=xticklabels)
    ax.set_title(f'{this_intervention.capitalize()} Interv. (post)',
                 loc='left', size='large')
    ax.grid(color='k', alpha=0.1)
    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')

legend_lines = [Line2D([], [], color=_col, label=_lab.capitalize(),
                linewidth=1) for _col, _lab in zip(colors, conditions)]
axs[1].legend(handles=legend_lines, ncol=1, loc='lower right',
              labelcolor='linecolor')

# add subfigure labels
kwargs = dict(weight='bold', size='large', ha='left', va='top')
brainfig.text(0.05, 0.98, 'A', **kwargs)
linefig.text(0.015, 0.98, 'B', **kwargs)
# add garnishes
linefig.supxlabel('time (s)')
linefig.supylabel('dSPM value')
linefig.subplots_adjust(left=0.1, bottom=0.13, right=0.97, top=0.92,
                        wspace=0.1)
# save
fig.savefig('lineplot-postcamp.png')
fig.savefig('lineplot-postcamp.pdf', dpi=400)
