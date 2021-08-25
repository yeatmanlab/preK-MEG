#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot spatial and temporal ROIs for the ERP experiment.
"""

import os
import yaml
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
import seaborn as sns

from analysis.aux_functions import load_paths


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
# find our temporal ROI
grand_mean = df.groupby('time').aggregate('mean').reset_index()
peak_indices, peak_props = find_peaks(grand_mean['value'], width=5)
peak_time = grand_mean['time'].iloc[peak_indices[0]]
peak_value = grand_mean['value'].iloc[peak_indices[0]]
temporal_roi = np.array([-0.05, 0.05]) + peak_time

with open('peak-of-grand-mean.yaml', 'w') as f:
    yaml.dump(dict(temporal_roi=np.around(temporal_roi, decimals=3).tolist(),
                   peak_time=round(float(peak_time), 3),
                   peak_value=float(peak_value)), f)

# config plot
sns.set_style('white')
ymax = dict(MNE=1e-10, dSPM=4, sLORETA=2, fft=300, snr=8)[method]
yticks = np.linspace(0, ymax, num=5)
xticks = np.linspace(0, 1, num=5)
yticklabels = nice_ticklabels(yticks)
xticklabels = nice_ticklabels(xticks)

# init figure
fig = plt.figure(figsize=(6, 3))
brainfig, linefig = fig.subfigures(1, 2, width_ratios=(1, 3), wspace=0.1)

# add brain ROI
img_path = os.path.join('.', 'MPM_IOS_IOG_pOTS_lh.png')
brainax = brainfig.subplots()
brain_image = imread(img_path)
brainax.imshow(brain_image, interpolation=None)
brainax.set_axis_off()
brainfig.subplots_adjust(bottom=0, top=1)

# add temporal ROI
ax = linefig.subplots(1, 1)
sns.lineplot(x='time', y='value', ci=68, ax=ax, data=df,
             legend=False, linewidth=1,
             err_kws=dict(alpha=0.25, edgecolor='none'))
# fill temporal ROI
ax.axvspan(*temporal_roi, facecolor='k', alpha=0.1, zorder=-2,
           edgecolor='none')
# garnish
ax.set_title('Grand mean across subjects and conditions',
             loc='left', size='large')
ax.set(xlabel='time (s)', ylabel='dSPM value',
       xlim=(-0.1, 1), ylim=(0, ymax),
       yticks=yticks, yticklabels=yticklabels,
       xticks=xticks, xticklabels=xticklabels)
ax.grid(color='k', alpha=0.1)

# add subfigure labels
kwargs = dict(weight='bold', size='large', ha='left', va='top')
brainfig.text(0.05, 0.98, 'A', **kwargs)
linefig.text(0.015, 0.98, 'B', **kwargs)

linefig.subplots_adjust(bottom=0.15, top=0.9)

fig.savefig('spatial-temporal-ROIs.png')
fig.savefig('spatial-temporal-ROIs.pdf', dpi=400)
