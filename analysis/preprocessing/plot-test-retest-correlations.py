#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Plot pre- vs post-camp evoked time course in early visual cortex (test/retest).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator
from analysis.aux_functions import (load_paths, load_params, yamload,
                                    load_inverse_params, PREPROCESS_JOINTLY)

# load params
*_, subjects, cohort = load_params(experiment='erp')

# inverse params
inverse_params = load_inverse_params()
method = inverse_params['method']
constr = inverse_params['orientation_constraint']
constr = f'-{constr}' if constr in ('free', 'fixed') else ''  # '' == loose
ori = inverse_params['estimate_type']
ori = ori if ori in ('vector', 'normal') else None  # None == 'magnitude'

# config paths
data_root, subjects_dir, _ = load_paths()
subfolder = 'combined' if PREPROCESS_JOINTLY else 'erp'
csvdir = 'csv'
os.makedirs(csvdir, exist_ok=True)

# config plot
colors = dict(words=('#EE99AA',   # light red
                     '#994455'),  # dark red
              faces=('#6699CC',   # light blue
                     '#004488'),  # dark blue
              cars=('#DDAA33',    # yellow (subbed for light yellow #EECC66)
                    '#997700'))   # dark yellow


# config other
conditions = ('words', 'faces', 'cars', 'aliens')
conditions_that_matter = conditions[:-1]
snr = 3.
lambda2 = 1. / snr ** 2
paramfile = os.path.join('..', 'preprocessing', 'mnefun_common_params.yaml')
with open(paramfile, 'r') as f:
    params = yamload(f)
lp_cut = params['preprocessing']['filtering']['lp_cut']
del params

# load rejection thresholds
with open('epoch-rejection-thresholds.yaml', 'r') as f:
    thresholds = yamload(f)

# load labels
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', hemi='both',
    regexp='Early Visual Cortex', subjects_dir=subjects_dir)


evoked_df = pd.DataFrame()

for s in subjects:
    # morph labels
    morphed_labels = mne.morph_labels(labels, subject_to=s.upper(),
                                      subject_from='fsaverage',
                                      subjects_dir=subjects_dir)
    # loop over pre/post measurement time
    for prepost in ('pre', 'post'):
        print(f'processing {s} {prepost}_camp')
        # paths for this subject / timepoint
        this_subj = os.path.join(data_root,
                                 f'{prepost}_camp', 'twa_hp', subfolder, s)
        inv_path = os.path.join(this_subj, 'inverse',
                                f'{s}-{lp_cut}-sss-meg{constr}-inv.fif')
        epo_path = os.path.join(this_subj, 'epochs',
                                f'All_{lp_cut}-sss_{s}-epo.fif')
        # load epochs
        epochs = mne.read_epochs(epo_path)
        epochs.drop_bad(thresholds)
        # equalize event counts
        epochs, dropped_indices = epochs.equalize_event_counts(
            event_ids=conditions_that_matter, method='mintime')
        # load inverse
        inv = read_inverse_operator(inv_path)
        # loop over conditions
        for cond in conditions_that_matter:
            # make evoked and STC
            evoked = epochs[cond].average()
            stc = apply_inverse(evoked, inv, lambda2, method=method,
                                pick_ori=ori)
            # extract time courses (one from each label)
            time_courses = mne.extract_label_time_course(
                stc, morphed_labels, inv['src'], mode='mean')
            n = len(evoked.times)
            hemis = [label.hemi for label in morphed_labels]
            for hemi, time_course in zip(hemis, time_courses):
                this_df = pd.DataFrame(
                    dict(subj=[s] * n, timept=[prepost] * n, cond=[cond] * n,
                         nave=[evoked.nave] * n, hemi=[hemi] * n,
                         time=evoked.times, value=time_course))
                evoked_df = pd.concat((evoked_df, this_df))

# load subject-level summary data
merged_df = pd.read_csv('trial-counts-and-r-values.csv')
melted = merged_df.melt(id_vars=['subj'],
                        value_vars=['rval_lh', 'rval_rh'],
                        value_name='rval')
rval_distr = melted.groupby('subj').agg('mean')
bad_subjs = rval_distr.query('rval < 0.4').index.values.astype(str)

# plot
timepoints = ('pre', 'post')
row_entries = [(cond, hemi) for cond in conditions_that_matter
               for hemi in ('lh', 'rh')]
fig = plt.figure(figsize=(2 * len(row_entries), 1.5 * len(subjects)))
axs = fig.subplots(len(subjects), len(row_entries), sharex=True, sharey=True)
for subj, row in zip(subjects, axs):
    for (cond, hemi), ax in zip(row_entries, row):
        query = f'subj == "{subj}" and cond == "{cond}" and hemi == "{hemi}"'
        data = evoked_df.query(query)
        sns.lineplot(
            data=data, x='time', y='value', hue='timept', style='timept',
            hue_order=timepoints, style_order=timepoints[::-1], ax=ax,
            palette=colors[cond], ci=None, legend=False)
        # prep for annotation
        data_pre = data.query('timept == "pre"')
        data_post = data.query('timept == "post"')
        nave_pre = data_pre["nave"].iat[0]
        nave_post = data_post["nave"].iat[0]
        rval = np.corrcoef(x=data_pre['value'], y=data_post['value'])[0, 1]
        # conditional formatting of bad values
        annot_kw = dict(x=0.05, transform=ax.transAxes, ha='left', va='top')
        bad_nave = np.any(np.array([nave_pre, nave_post]) < 15)
        bad_rval = rval < 0.4
        ax.text(y=0.96, s=f'nave: {nave_pre},{nave_post}',
                color='#AA4499' if bad_nave else '0.5',
                weight='bold' if bad_nave else 'normal',
                **annot_kw)
        ax.text(y=0.84, s=f'r = {rval:.03f}',
                color='#117733' if bad_rval else '0.5',
                weight='bold' if bad_rval else 'normal',
                **annot_kw)
        if subj in bad_subjs:
            ax.patch.set_color('0.9')
        # garnish
        spec = ax.get_subplotspec()
        if spec.is_first_row():
            ax.set_title(f'{hemi}: {cond}')
        if spec.is_first_col():
            ax.set_ylabel(subj)

fig.subplots_adjust(left=0.05, right=0.975, bottom=0.01, top=0.99)

# legend
legend_lines = (Line2D([], [], color='0.6', label='pre', linestyle='--',
                linewidth=1),
                Line2D([], [], color='k', label='post', linewidth=1))
axs[0, -1].legend(handles=legend_lines, ncol=2, loc='lower right',
                  frameon=True, bbox_to_anchor=(1.04, 1.2))

fig.suptitle('test-retest (pre-vs-post) in early visual cortex',
             x=0.05, y=0.998, ha='left', size='x-large')
fig.savefig('test-retest-label-timecourses.png')
