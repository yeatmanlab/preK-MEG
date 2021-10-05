#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Check pre/post P1 response correlation.
"""

import os
import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import apply_inverse, read_inverse_operator
from analysis.aux_functions import (load_paths, load_params, yamload,
                                    load_inverse_params, PREPROCESS_JOINTLY)

from joblib import Parallel, delayed

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

# load labels
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1_combined', hemi='both',
    regexp='Early Visual Cortex', subjects_dir=subjects_dir)


# function to get r-values from different epoch rejection thresholds
def get_rval(mag, grad):
    rval_df = pd.DataFrame()
    # loop over subjects
    for s in subjects:
        time_courses = dict()
        n_aves = dict()
        # morph labels
        morphed_labels = mne.morph_labels(labels, subject_to=s.upper(),
                                          subject_from='fsaverage',
                                          subjects_dir=subjects_dir)
        # loop over pre/post measurement time
        for prepost in ('pre', 'post'):
            print(f'processing {s} {prepost}_camp')
            time_courses[prepost] = dict()
            # paths for this subject / timepoint
            this_subj = os.path.join(data_root,
                                     f'{prepost}_camp', 'twa_hp', subfolder, s)
            inv_path = os.path.join(this_subj, 'inverse',
                                    f'{s}-{lp_cut}-sss-meg{constr}-inv.fif')
            epo_path = os.path.join(this_subj, 'epochs',
                                    f'All_{lp_cut}-sss_{s}-epo.fif')
            # load epochs
            epochs = mne.read_epochs(epo_path)
            epochs.drop_bad(dict(mag=mag, grad=grad))
            # make sure we have something to work with
            if not len(epochs):
                return -1
            lengths = [len(epochs[cond]) for cond in conditions_that_matter]
            if not np.all(lengths):
                return -1
            # equalize event counts
            meth = 'mintime' if np.all(np.array(lengths) > 1) else 'truncate'
            epochs, dropped_indices = epochs.equalize_event_counts(
                event_ids=conditions_that_matter, method=meth)
            # load inverse
            inv = read_inverse_operator(inv_path)
            # loop over conditions
            for cond in conditions_that_matter:
                # make evoked and STC
                evoked = epochs[cond].average()
                stc = apply_inverse(evoked, inv, lambda2, method=method,
                                    pick_ori=ori)
                # extract time courses (one from each label)
                time_courses[prepost][cond] = mne.extract_label_time_course(
                    stc, morphed_labels, inv['src'], mode='mean')
            # just store once; all conds same thanks to equalize_event_counts:
            n_aves[prepost] = evoked.nave
        # compute R values
        for cond in conditions_that_matter:
            for ix, hemi in enumerate(('lh', 'rh')):
                rval = np.corrcoef(
                    x=time_courses['pre'][cond][ix],
                    y=time_courses['post'][cond][ix])
                row = pd.DataFrame(dict(subj=[s], hemi=[hemi], cond=[cond],
                                        n_pre=n_aves['pre'],
                                        n_post=n_aves['post'],
                                        rval=[rval[0, 1]]))
                rval_df = pd.concat([rval_df, row], ignore_index=True)
    rval_df['absr'] = rval_df['rval'].apply(np.abs)
    # save the dataframe for later inspection
    fname = (f'pre-post-correlations-mag{int(mag * 1e15)}fT'
             f'-grad{int(grad * 1e13)}fTcm.csv')
    rval_df.to_csv(os.path.join(csvdir, fname))
    # print(rval_df['absr'].describe())
    return rval_df['absr'].mean()


# grid search setup
mags = np.linspace(3, 10, 15) * 1e-12  # 3k-10k fT, in 500 fT steps
grads = np.linspace(1, 4, 13) * 1e-10  # 1k-4k fT/cm, in 250 fT/cm steps
mags_, grads_ = np.meshgrid(mags, grads)
mags_ = mags_.ravel()
grads_ = grads_.ravel()

result_df = pd.DataFrame()
rvals = Parallel(n_jobs=6)(
    delayed(get_rval)(mag, grad) for mag, grad in zip(mags_, grads_))

result_df = pd.DataFrame(dict(mag=mags_, grad=grads_, rval=rvals))
result_df.to_csv('crossval-results.csv', index=False)
print(result_df.iloc[result_df['rval'].argmax()])
