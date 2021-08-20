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
method = 'dSPM'  # ('dSPM', 'MNE')
ymax = dict(MNE=1e-10, dSPM=4, sLORETA=2, fft=300, snr=8)[method]
yticks = np.linspace(0, ymax, num=5)
xticks = np.linspace(0, 1, num=5)
yticklabels = nice_ticklabels(yticks)
xticklabels = nice_ticklabels(xticks)

# init figure
fig = plt.figure(figsize=(6.5, 4))
brainfig, linefig = fig.subfigures(1, 2, width_ratios=(1, 3), wspace=0.1)

# add ROI
img_path = os.path.join('.', 'MPM_IOS_IOG_pOTS_lh.png')
brainax = brainfig.subplots()
brain_image = imread(img_path)
brainax.imshow(brain_image, interpolation=None)
brainax.set_axis_off()
brainfig.subplots_adjust(bottom=0.3, top=0.95)

# do lineplots
axs = linefig.subplots(2, 3, sharex=True, sharey=True)
for this_intervention, row in zip(interventions, axs):
    for this_condition, ax in zip(conditions, row):
        spec = ax.get_subplotspec()
        this_df = df[(df['condition'] == this_condition) &
                     (df['intervention'] == this_intervention) &
                     (df['method'] == method)]
        with sns.color_palette(colors[this_condition]):
            sns.lineplot(x='time', y='value', ci=68, hue='timepoint',
                         style='timepoint', hue_order=timepoints,
                         style_order=timepoints[::-1], ax=ax, data=this_df,
                         legend=False, linewidth=1,
                         err_kws=dict(alpha=0.25, edgecolor='none'))
        # fill temporal ROI
        ax.axvspan(*temporal_roi, color='0.9', zorder=-2)
        # garnish
        ax.set(xlabel='', ylabel='', xlim=(-0.1, 1), ylim=(0, ymax),
               yticks=yticks, yticklabels=yticklabels,
               xticks=xticks, xticklabels=xticklabels)
        ax.grid(color='0.9')
        for spine in ax.spines.values():
            spine.set_edgecolor('0.5')
        # titles
        if spec.is_first_col():
            ax.set_title(f'{this_intervention.capitalize()} Intervention',
                         loc='left', size='large')
        if spec.is_first_row():
            ax.text(0.95, 0.95, this_condition.capitalize(),
                    ha='right', va='top', transform=ax.transAxes,
                    color=colors[this_condition][1])

linefig.supxlabel('time (s)')
linefig.supylabel('dSPM value')
linefig.subplots_adjust(left=0.1, bottom=0.12, right=0.97, top=0.93,
                        hspace=0.3, wspace=0.1)

legend_lines = (Line2D([], [], color='0.6', label='pre-intervention',
                       linestyle='--', linewidth=1),
                Line2D([], [], color='k', label='post-intervention',
                       linewidth=1))
axs[1, 0].legend(handles=legend_lines, ncol=1, loc='lower right',
                 frameon=True, bbox_to_anchor=(-0.18, -0.06))

# add subfigure labels
kwargs = dict(weight='bold', size='large', ha='left', va='top')
brainfig.text(0.05, 0.98, 'A', **kwargs)
linefig.text(0.015, 0.98, 'B', **kwargs)

fig.savefig('lineplot-grid.png')
fig.savefig('lineplot-grid.pdf', dpi=400)
