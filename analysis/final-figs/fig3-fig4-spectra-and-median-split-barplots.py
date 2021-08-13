#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot FFT and SNR spectra for a given label, for various splits of the data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# config label
region = 'IOS_IOG_pOTS'
# load data
df = pd.read_csv(f'pskt-in-label-{region}.csv')
# config plot
methods = ('fft', 'snr')  # vary by row
sns.set_style('white')


def get_data_subset(df, method, condition, freqs=None):
    cond_loc = (np.array(True) if condition is None else
                np.in1d(df['condition'], condition))
    freq_loc = np.array(True) if freqs is None else np.in1d(df['freq'], freqs)
    return df.loc[(df['method'] == method) & cond_loc & freq_loc]


def garnish(ax, method, title, xlabel, legend=True):
    spec = ax.get_subplotspec()
    xlabel = xlabel if spec.is_last_row() else ''
    ylabel = method.upper() if spec.is_first_col() else ''
    ymax = dict(fft=200, snr=4.5)[method]
    ax.set(ylim=(0, ymax), xlabel=xlabel, ylabel=ylabel)
    if spec.is_first_row():
        ax.set(title=title)
    # remove redundant legends, and reverse item order in legend that we keep
    if ax.legend_ is not None:
        ax.legend_.remove()
    if legend and spec.is_first_row() and spec.is_last_col():
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Pretest')
    # frame color
    ax.tick_params(color='0.5')
    for spine in ax.spines.values():
        spine.set_edgecolor('0.5')


# fig 3: grand average spectra
fig, axs = plt.subplots(2, 1, sharex=True)
fig.set_size_inches(8, 6)
for method, ax in zip(methods, axs):
    # get data subset
    skip_bins = 1 if method == 'fft' else 2  # omit DC
    freqs = df['freq'].unique()[skip_bins:]
    this_df = get_data_subset(df, method, condition=None, freqs=freqs)
    # draw
    sns.lineplot(x='freq', y='value', data=this_df, ax=ax, color='k', ci=68)
    # garnish
    ax.grid(color='k', linewidth=1, ls=":", dashes=(1, 5, 1, 5), alpha=0.6)
    garnish(ax, method, xlabel='Frequency (Hz)',
            title='All subjects/conditions', legend=False)
# custom ticks
xticks = [0, 2, 4, 6, 12, 18, 24, 30, 36]
xticklabels = list(map(str, xticks))
for ix, label in enumerate(xticklabels):
    if label == '2':
        xticklabels[ix] += '\n(oddball)'
    elif label == '6':
        xticklabels[ix] += '\n(stim)'
ax.set(xticks=xticks, xticklabels=xticklabels, xlim=(-1, 40))
# save
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.95, hspace=0.05)
fig.align_labels()
fig.savefig('fig3-pskt-grandavg-spectra-fft-snr.png')
plt.close(fig)


# fig 4: cohort comparison barplots
freqs_of_interest = [2, 4, 6, 12]
conditions = (['ps', 'kt'], ['ps'], ['kt'])  # vary by column
titles = ('all trials', 'pseudotext', 'Korean text')
colors = ['#AA3377', '#228833']
# init figure
fig, axs = plt.subplots(2, 3, sharex=True, sharey='row')
fig.set_size_inches(8, 6)
for method, row in zip(methods, axs):
    for ax, cond, title in zip(row, conditions, titles):
        # get data subset
        this_df = get_data_subset(df, method, cond, freqs=freqs_of_interest)
        # draw
        with sns.color_palette(colors):
            sns.barplot(x='freq', y='value', data=this_df, ax=ax,
                        hue='pretest', hue_order=['lower', 'upper'], ci=68,
                        errwidth=2, capsize=0, saturation=1)
        garnish(ax, method, xlabel='', title=title)
        # remove unnecessary ticks
        ax.tick_params(bottom=False)
        ax.set(xticklabels=[f'{fr} Hz' for fr in freqs_of_interest])
fig.suptitle('Median split on literacy pretest', weight='bold', size='larger')
fig.supxlabel('Frequency bin')
# save
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.1, top=0.9,
                    hspace=0.05, wspace=0.05)
fig.align_labels()
fig.savefig('fig4-pskt-pretest-barplots.png')
plt.close(fig)
