#!/usr/bin/env python

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sswef_helpers.aux_functions import load_subjects

plt.ion()
seed = 15485863  # the one millionth prime
method = 'quantile'  # gaussian or quantile
plots = False

# load the data
metadata = pd.read_csv('preK_InterventionData.csv')

# remap values
mapping = dict(Letter='LetterIntervention', Language='LanguageIntervention')
metadata.replace(dict(group=mapping), inplace=True)
# load the experiment cohorts
cohort_map = {subj: cohort for cohort in ('original', 'replication')
              for subj in list(map(lambda x: int(x.lstrip('prek_')),
                                   load_subjects(cohort)))}
metadata['cohort'] = metadata['subID'].map(cohort_map)

# # # # # # # # # # # # # # # # # # # # # # # #
# FIND THE PRE-TEST LETTER KNOWLEDGE CUTPOINT #
# # # # # # # # # # # # # # # # # # # # # # # #

# base "prior knowledge" cutpoint on: original cohort pre-camp letter knowledge
precamp = metadata.query('cohort == "original" & visit == "pre"').copy()
cutvar = 'AlphabetKnowledge'
if plots:
    sns.distplot(precamp[cutvar], bins=np.linspace(0, 100, 101))

if method == 'gaussian':
    # find the trough in the bimodal
    X = np.atleast_2d(precamp[cutvar]).T
    model = GaussianMixture(n_components=2, covariance_type='spherical',
                            random_state=seed)
    model.fit(X)
    # map letter awareness values to groups
    cohorts = model.predict(X)
    precamp['pretest_cohort'] = cohorts
    precamp.sort_values(by=cutvar, axis=0, inplace=True)
    means = precamp.groupby('pretest_cohort')[cutvar].mean()
    cohort_names = {means.argmax(): 'UpperKnowledge',
                    means.argmin(): 'LowerKnowledge'}
    precamp['pretest_cohort'] = precamp['pretest_cohort'].map(cohort_names)
    # apply the cutpoint to the full dataset
    X = np.atleast_2d(metadata[cutvar]).T
    cohorts = model.predict(X)
    metadata['pretest_cohort'] = cohorts
    metadata['pretest_cohort'] = metadata['pretest_cohort'].map(cohort_names)
elif method == 'quantile':
    # use all data to determine endpoints (otherwise we end up with NaNs)
    bins = metadata[cutvar].quantile([0, 0.5, 1])
    # use pre-camp data to determine middle cutpoint:
    bins.at[0.5] = precamp[cutvar].quantile(0.5)
    metadata['pretest_cohort'] = pd.cut(
        metadata[cutvar], bins=bins, include_lowest=True, retbins=False,
        labels=('LowerKnowledge', 'UpperKnowledge'))


# # # # # # # # # # # # # # # #
# PLOT VARIABLE CORRELATIONS  #
# # # # # # # # # # # # # # # #
if plots:
    value_vars = ('AlphabetKnowledge', 'Decoding', 'PhonemeMatching',
                  'PhonemeSegmenting', 'EVTRaw', 'Retell', 'Grammar')
    g = sns.pairplot(metadata, vars=value_vars, kind='reg',
                     hue='pretest_cohort')


# # # # # # # # # # # # # # #
# EXTRACT GROUPINGS TO YAML #
# # # # # # # # # # # # # # #

cohorts = ('original', 'replication', 'pooled')
for cohort in cohorts:
    subjects = load_subjects(cohort)
    subjects = list(map(lambda x: int(x.lstrip('prek_')), subjects))
    _metadata = metadata[np.in1d(metadata['subID'], subjects)].copy()

    # extract grouping based on intervention received
    camp_groups = (_metadata
                   .set_index('group')
                   .groupby(level=0)['subID']
                   .unique()
                   .to_dict())

    # extract grouping based on pre-test scores
    pretest_groups = (_metadata
                      .groupby('visit')
                      .get_group('pre')
                      .set_index('pretest_cohort')
                      .groupby(level=0)['subID']
                      .unique()
                      .to_dict())

    # validate
    lower = set(pretest_groups['LowerKnowledge'])
    upper = set(pretest_groups['UpperKnowledge'])
    # cohorts are unique:
    assert len(lower.intersection(upper)) == 0
    # cohorts are exhaustive:
    assert len(set(_metadata['subID']) - lower.union(upper)) == 0

    # arrays → lists (for cleaner YAML writes)
    pretest_groups = {key: sorted(value.tolist()) for key, value in
                      pretest_groups.items()}
    camp_groups = {key: sorted(value.tolist()) for key, value in
                   camp_groups.items()}

    # save
    mode = 'w' if cohort == cohorts[0] else 'a'
    fnames = ('letter_knowledge_cohorts.yaml', 'intervention_cohorts.yaml')
    for fname, _dict in zip(fnames, (pretest_groups, camp_groups)):
        with open(os.path.join('..', '..', 'params', fname), mode) as f:
            yaml.dump({cohort: _dict}, stream=f)
