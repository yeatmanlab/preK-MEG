#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract results from rejection threshold cross-validation and write to YAML;
compute and plot epoch peak-to-peak histograms and epochs retained per
condition.
"""

import os
import yaml
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mne
from analysis.aux_functions import load_paths, load_params, yamload

# load general params
data_root, subjects_dir, _ = load_paths()
*_, subjects, cohort = load_params(experiment='erp', skip=False)

# load lowpass
paramfile = 'mnefun_common_params.yaml'
with open(paramfile, 'r') as f:
    params = yamload(f)
lp_cut = params['preprocessing']['filtering']['lp_cut']
del params

# load crossval results, extract best thresholds, and write to YAML
crossval_df = pd.read_csv('crossval-results.csv')
best_rval_row = (crossval_df.query('rval == rval.max()')
                            .query('mag == mag.max()')
                            .query('grad == grad.max()'))
thresholds = best_rval_row.drop(columns='rval').to_dict('records')[0]
with open('epoch-rejection-thresholds.yaml', 'w') as f:
    yaml.dump(thresholds, f)

# prep for iteration
event_dict = dict(words=10, faces=20, cars=30, aliens=40)
peak_to_peak_df = pd.DataFrame()
trial_count_df = pd.DataFrame()

for tpt in ('pre', 'post'):
    for subj in subjects:
        # load raw and events, and epoch without any peak-to-peak rejection
        slug = f'/mnt/scratch/prek/{tpt}_camp/twa_hp/erp/{subj}'
        raw_fname = f'{slug}/sss_pca_fif/{subj}_erp_{tpt}_allclean_fil{lp_cut}_raw_sss.fif'  # noqa E501
        eve_fname = f'{slug}/lists/ALL_{subj}_erp_{tpt}-eve.lst'
        try:
            raw = mne.io.read_raw_fif(raw_fname)
        except FileNotFoundError:
            continue
        events = mne.read_events(eve_fname)
        epochs = mne.Epochs(raw, events, event_id=event_dict, reject=None,
                            preload=True)
        del raw

        # tabulate peak-to-peak values for each epoch
        for ch_type in ('mag', 'grad'):
            this_epochs = epochs.copy().pick(ch_type)
            for epoch in this_epochs:
                row = pd.DataFrame(dict(
                    subj=[subj], timepoint=[tpt], ch_type=[ch_type],
                    min=[epoch.min()], max=[epoch.max()],
                    ptp_ampl=[epoch.max() - epoch.min()]))
                peak_to_peak_df = pd.concat((peak_to_peak_df, row))

        # apply rejection thresholds and tally remaining epochs per condition
        epochs.drop_bad(thresholds)
        record = {k: [Counter(epochs.events[:, 2])[v]]
                  for k, v in epochs.event_id.items()}
        row = pd.DataFrame(dict(subj=[subj], timepoint=[tpt], **record))
        trial_count_df = pd.concat((trial_count_df, row))

# add in subject-level R-values
fname = (f'pre-post-correlations-mag{int(thresholds["mag"] * 1e15)}fT'
         f'-grad{int(thresholds["grad"] * 1e13)}fTcm.csv')
fpath = os.path.join('csv', fname)
subj_crossval_df = pd.read_csv(fpath, index_col=0)
rval_df = subj_crossval_df.pivot(
    index=['subj', 'cond'], columns='hemi', values='rval'
    ).rename(columns=lambda x: f'rval_{x}')
epoch_df = trial_count_df.melt(
    id_vars=['subj', 'timepoint'],
    value_vars=['words', 'faces', 'cars', 'aliens'],
    var_name='cond', value_name='n_epochs'
    ).pivot(
        index=['subj', 'cond'], columns='timepoint', values='n_epochs'
    ).rename(columns=lambda x: f'n_epochs_{x}')
merged_df = rval_df.merge(
    epoch_df, left_index=True, right_index=True).reset_index()

# plot histogram of peak-to-peak epoch amplitudes
g = sns.FacetGrid(peak_to_peak_df, row='ch_type', sharex=False)
g.map(sns.histplot, 'ptp_ampl', stat='count')
for ch_type, ax in g.axes_dict.items():
    ax.axvline(x=thresholds[ch_type], color='C1', ls='--')
g.fig.savefig('peak-to-peak-hists-and-rejection-thresholds.png')

# scatterplot r-values in left and right hemispheres
fig, ax = plt.subplots()
sns.scatterplot(x='rval_lh', y='rval_rh', hue='cond', style='cond',
                data=merged_df, legend=False, ax=ax)
ax.grid(True)
fig.savefig('rval-scatterplot.png')

# dotplot of r-values for each subj
g = sns.PairGrid(data=merged_df.sort_values(['rval_lh', 'rval_rh']),
                 x_vars=['rval_lh', 'rval_rh'],
                 y_vars=['subj'],
                 hue='cond', height=10, aspect=0.3)
g.map(sns.stripplot, orient='h', jitter=False, linewidth=1, edgecolor='w')
for ax in g.axes.flat:
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.axvline(0, color='0.5', linewidth=1)
g.set(xlim=(-1, 1), ylabel="")
g.fig.subplots_adjust(left=0.2, wspace=0.4)
sns.despine(left=True, bottom=True)
g.fig.savefig('rval-dotplot.png')

# distribution of r-values across subjects
melted = merged_df.melt(id_vars=['subj'],
                        value_vars=['rval_lh', 'rval_rh'],
                        value_name='rval')
rval_distr = melted.groupby('subj').agg('mean').sort_values('rval')
ax = sns.histplot(rval_distr, x='rval', bins=int(round(len(subjects) // 3)))
ax.set_title('distribution of mean R-values across subjects')
ax.figure.savefig('rval-distribution.png')

# save
peak_to_peak_df.to_csv('unthresholded-epoch-peak-to-peak-amplitudes.csv',
                       index=False)
trial_count_df.to_csv('trial-counts-after-thresholding.csv', index=False)
merged_df.to_csv('trial-counts-and-r-values.csv', index=False)
# print(trial_count_df.query('cars<15 or words<15 or faces<15'))
