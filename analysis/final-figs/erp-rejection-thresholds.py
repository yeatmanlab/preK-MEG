#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot epoch rejection crossvalidation results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.image import imread
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns
import mne
from analysis.aux_functions import (load_paths, load_params, yamload,
                                    nice_ticklabels)

# flags
force_redraw_brain = False

# config paths
data_root, subjects_dir, _ = load_paths()
preproc_dir = os.path.join('..', 'preprocessing')

# load params
brain_plot_kwargs, _, subjects, _ = load_params(experiment='erp')

# config plot
for kwarg in ('time_viewer', 'show_traces'):
    del brain_plot_kwargs[kwarg]  # not used in Brain.__init__
brain_plot_kwargs.update(views=['cau', 'ven'], hemi='split', surf='inflated',
                         background='white', size=(2400, 3600))

# load epoch info
thresh_path = os.path.join(preproc_dir, 'epoch-rejection-thresholds.yaml')
crossval_path = os.path.join(preproc_dir, 'crossval-results.csv')
ptp_path = os.path.join(preproc_dir,
                        'unthresholded-epoch-peak-to-peak-amplitudes.csv')
with open(thresh_path, 'r') as f:
    thresholds = yamload(f)
crossval_df = pd.read_csv(crossval_path)
ptp_df = pd.read_csv(ptp_path)
# thresholds tried in grid search
gridsearch = dict(
    mag=np.linspace(3, 10, 15) * 1e-12,  # 3k-10k fT, in 500 fT steps
    grad=np.linspace(1, 4, 13) * 1e-10)  # 1k-4k fT/cm, in 250 fT/cm steps)

# prepare colormap
light_grey = '0.8'
cmap = get_cmap('viridis').copy()
cmap.set_extremes(under=light_grey, over='w')

# init figure
fig = plt.figure(figsize=(6.5, 9))
_fig, heatfig = fig.subfigures(2, 1, height_ratios=(2, 3), hspace=0.3)
brainfig, histfig = _fig.subfigures(1, 2, width_ratios=(2, 3), wspace=0.2)

# labels / scalings
axis_labels = dict(mag='magnetometers (fT)', grad='gradiometers (fT/cm)')
scalings = dict(mag=1e15, grad=1e13)

# plot the brain
img_fname = 'early-visual-cortex.png'
img_path = os.path.join('.', img_fname)
if force_redraw_brain or not os.path.exists(img_path):
    # load labels
    labels = mne.read_labels_from_annot(
        'fsaverage', 'HCPMMP1_combined', hemi='both',
        regexp='Early Visual Cortex', subjects_dir=subjects_dir)
    # plot labels
    brain = mne.viz.Brain('fsaverage', **brain_plot_kwargs)
    for ix, lab in enumerate(labels):
        lab.values.fill(1.)
        lab.smooth(subject='fsaverage', subjects_dir=subjects_dir, smooth=1)
        brain.add_label(lab, alpha=0.75, color='#44BB99')
        brain.add_label(lab, alpha=1, color='#44BB99', borders=True)
    brain.save_image(img_path)

# add brain ROI to main figure
brainax = brainfig.subplots()
brain_image = imread(img_path)
brainax.imshow(brain_image, interpolation=None)
brainax.set_axis_off()
brainax.set_title('HCPMMP1_combined\n"Early Visual Cortex"\n'
                  '(ROI used for epoch\nrejection cross-validation)')
brainfig.subplots_adjust(bottom=0, top=0.75)

# add histograms
axs = histfig.subplots(2, 1, sharex=False)
for (ch_type, _thresholds), ax in zip(gridsearch.items(), axs):
    data = ptp_df.query(f'ch_type == "{ch_type}"').filter(['ptp_ampl'])
    sns.histplot(data, ax=ax, legend=False)
    for _thresh in _thresholds:
        color = 'C1' if _thresh == thresholds[ch_type] else light_grey
        lw = 2 if _thresh == thresholds[ch_type] else 1
        ax.axvline(x=_thresh, color=color, linestyle='--', linewidth=lw)
        xticklabels = nice_ticklabels(
            np.rint(ax.get_xticks() * scalings[ch_type] / 1e3))
        ax.set(xlabel=axis_labels[ch_type], ylabel='',
               xticklabels=[f'{val}k' for val in xticklabels])
axs[0].set_title('Histograms of epoch peak-to-peak amplitudes', size='medium')
histfig.subplots_adjust(bottom=0.15, top=0.9, hspace=0.6)

# add r-value heatmap (rval=-1 is sentinel for too few epochs)
ax, cbar_ax = heatfig.subplots(1, 2, gridspec_kw=dict(width_ratios=(19, 1)))
data = crossval_df.pivot('mag', 'grad', 'rval').iloc[::-1]
positive_rvals = crossval_df.query('rval > 0')['rval']
sns.heatmap(data, cmap=cmap, vmin=positive_rvals.min(),
            vmax=positive_rvals.max(), ax=ax, cbar_ax=cbar_ax, square=True)
# make the tick labels nicer
xtickvals = nice_ticklabels(
    data.columns.values * scalings[data.columns.name] / 1e3)
ytickvals = nice_ticklabels(
    data.index.values * scalings[data.index.name] / 1e3)
xlabs = [f'{val}k' for val in xtickvals]
ylabs = [f'{val}k' for val in ytickvals]
ax.set(xticklabels=xlabs, yticklabels=ylabs,
       xlabel=axis_labels[data.columns.name],
       ylabel=axis_labels[data.index.name])

# make a grid
xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.set_xticks((xticks[1:] + xticks[:-1]) / 2, minor=True)
ax.set_yticks((yticks[1:] + yticks[:-1]) / 2, minor=True)
ax.grid(which='minor', color='w')
ax.tick_params(which='minor', bottom=False, left=False)
# highlight max cells
maxes = np.where(data == data.values.max())
for y, x in zip(*maxes):
    rect = Rectangle((x, y), width=1, height=1, edgecolor='C1',
                     facecolor='none', zorder=20, linewidth=1.5)
    ax.add_patch(rect)
# tweaks
ylab = 'Mean pre- versus post-intervention R-value (activation in label)'
cbar_ax.set_ylabel(ylab, labelpad=12)
heatfig.subplots_adjust(left=0.1, right=0.85, bottom=0.15, top=0.95)

fig.savefig('epoch-rejection-crossval.png')
