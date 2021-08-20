#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot FFT and SNR spectra for a given label, for various splits of the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from analysis.aux_functions import load_paths


def nice_ticklabels(ticks):
    return list(map(str, [int(t) if t == int(t) else t for t in ticks]))


# config paths
data_root, subjects_dir, results_dir = load_paths()
timeseries_dir = os.path.join(results_dir, 'roi', 'time-series')

# load data
region = 'MPM_IOS_IOG_pOTS_lh'
fname = f'roi-{region}-timeseries-long.csv'
df = pd.read_csv(os.path.join(timeseries_dir, fname), index_col=0)
peak_of_grand_mean = 0.185
temporal_roi = np.array([-0.05, 0.05]) + peak_of_grand_mean

# config plot
sns.set_style('white')
colors = ('#994455',  # dark red
          '#004488',  # dark blue
          '#997700')  # dark yellow

interventions = ('letter', 'language')
conditions = ('words', 'faces', 'cars')
timepoints = ('pre', 'post')
method = 'dSPM'  # ('dSPM', 'MNE')
ymax = dict(MNE=1e-10, dSPM=4, sLORETA=2, fft=300, snr=8)[method]
yticks = np.linspace(0, ymax, num=5)
yticklabels = nice_ticklabels(yticks)
# do custom xticks because of ROI
xticks = np.linspace(0, 1, num=5)
xticklabels = nice_ticklabels(xticks)

# init figure
fig = plt.figure(figsize=(6.5, 4))
axs = fig.subplots(1, 2, sharex=True, sharey=True)

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
    ax.axvspan(*temporal_roi, color='0.9', zorder=-2)
    # garnish
    ax.set(xlabel='', ylabel='', xlim=(-0.1, 1), ylim=(0, ymax),
           yticks=yticks, yticklabels=yticklabels,
           xticks=xticks, xticklabels=xticklabels)
    ax.set_title(f'{this_intervention.capitalize()} Intervention (post)',
                 loc='left', size='large')
    ax.grid(color='0.9')
    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')

legend_lines = [Line2D([], [], color=_col, label=_lab.capitalize(),
                linewidth=1) for _col, _lab in zip(colors, conditions)]
axs[1].legend(handles=legend_lines, ncol=1, loc='upper right',
              labelcolor='linecolor')

fig.supxlabel('time (s)')
fig.supylabel('dSPM value')
fig.subplots_adjust(left=0.09, bottom=0.12, right=0.97, top=0.93,
                    wspace=0.1)
fig.savefig('lineplot-postcamp.png')
fig.savefig('lineplot-postcamp.pdf')
