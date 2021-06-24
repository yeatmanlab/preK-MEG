#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot FFT and SNR spectra for a given label, for various splits of the data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis.aux_functions import load_paths

# config label
region = '2_Hz-LetterKnowledge'

# config paths
data_root, subjects_dir, results_dir = load_paths()
spectrum_dir = os.path.join(results_dir, 'pskt', 'roi')
df_path = os.path.join(spectrum_dir, f'roi-{region}-frequencies-long.csv')

# config other
methods = ('fft', 'snr')  # vary by row
conditions = (['all'], ['all'], ['ps'], ['kt'])  # vary by column
freqs_of_interest = [2, 4, 6, 12]
titles = ('All subjects/conditions',
          'Median split on literacy pretest (all trials)',
          'Median split on literacy pretest (pseudotext)',
          'Median split on literacy pretest (Korean text)')

# load data
df = pd.read_csv(df_path, index_col=0)
# init figure
fig, axs = plt.subplots(2, 4, sharex='col', sharey='row',
                        gridspec_kw=dict(width_ratios=(2, 1, 1, 1)))
fig.set_size_inches(24, 12)
# plot setup
sns.set_style('white')
colors = ['#AA3377', '#228833']


def get_data_subset(df, method, condition, freqs=None):
    cond_loc = np.in1d(df['condition'], condition)
    freq_loc = np.array(True) if freqs is None else np.in1d(df['freq'], freqs)
    return df.loc[(df['method'] == method) & cond_loc & freq_loc]


def garnish(ax, method, title):
    spec = ax.get_subplotspec()
    xlabel = 'Frequency bin'
    ylabel = ''
    if spec.is_first_col():
        ax.grid(color='k', linewidth=1, ls=":", dashes=(1, 5, 1, 5), alpha=0.6)
        ylabel = dict(fft='FFT magnitude', snr='SNR (Au)')[method]
        xlabel = 'Frequency (Hz)'
    xlabel = xlabel if spec.is_last_row() else ''
    ymax = dict(fft=300, snr=6.5)[method]
    ax.set(ylim=(0, ymax), xlabel=xlabel, ylabel=ylabel)
    if spec.is_first_row():
        ax.set(title=title)
    # ticks
    if spec.is_last_row():
        if spec.is_first_col():
            xticks = [0, 2, 4, 6, 12, 18, 24, 30, 36]
            xticklabels = list(map(str, xticks))
            for ix, label in enumerate(xticklabels):
                if label == '2':
                    xticklabels[ix] += '\n(oddball)'
                elif label == '6':
                    xticklabels[ix] += '\n(stim)'
            ax.set(xticks=xticks, xticklabels=xticklabels, xlim=(-1, 40))
        else:
            ax.set(xticklabels=[f'{fr} Hz' for fr in freqs_of_interest])
    # remove redundant legends
    upper_right = spec.is_first_row() and spec.is_last_col()
    if ax.legend_ is not None and not upper_right:
        ax.legend_.remove()
    # frame color
    ax.tick_params(color='0.5')
    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')


for method, row in zip(methods, axs):
    for ix, (ax, cond) in enumerate(zip(row, conditions)):
        skip_bins = 1 if method == 'fft' else 2  # omit DC
        subset_kwargs = dict(freqs=df['freq'].unique()[skip_bins:])
        plot_kwargs = dict(color='k')
        plot_func = sns.lineplot
        if ix:
            subset_kwargs = dict(freqs=freqs_of_interest)
            plot_kwargs = dict(hue='pretest', hue_order=['lower', 'upper'],
                               errwidth=1.5, errcolor='k', saturation=1)
            plot_func = sns.barplot
        # get data subset
        this_df = get_data_subset(df, method, cond, **subset_kwargs)
        # draw
        with sns.color_palette(colors if ix else None):
            plot_func(x='freq', y='value', data=this_df, ax=ax, **plot_kwargs)
        garnish(ax, method, title=titles[ix])
        # remove unnecessary ticks
        if ix:
            ax.tick_params(bottom=False)

# save plot (overwrites the cluster image PNG)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                    hspace=0.05, wspace=0.05)
fig.align_labels()
fig.savefig('fig-pskt-spectra-and-pretest-barplots.svg')
plt.close(fig)
