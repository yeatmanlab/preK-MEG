#!/usr/bin/env python

import os
import yaml
import numpy as np
import pandas as pd
from analysis.aux_functions import load_subjects

# load the data
metadata = pd.read_csv('preK_InterventionData.csv')

# remap values
mapping = dict(Letter='LetterIntervention', Language='LanguageIntervention')
metadata.replace(dict(group=mapping), inplace=True)

# load the experiment cohorts
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

    # group (upper/lower half) based on alphabet knowlege
    cutvar = 'AlphabetKnowledge'
    precamp = _metadata.groupby('visit').get_group('pre')

    # use pre-camp data to determine middle cutpoint; use all data to determine
    # endpoints (otherwise we end up with NaNs)
    bins = _metadata[cutvar].quantile([0, 0.5, 1])
    bins.at[0.5] = precamp[cutvar].quantile(0.5)
    _metadata['cohort'] = pd.cut(_metadata[cutvar], bins=bins,
                                 include_lowest=True, retbins=True,
                                 labels=('LowerKnowledge', 'UpperKnowledge')
                                 )[0]
    cohort_groups = (_metadata
                     .groupby('visit')
                     .get_group('pre')
                     .set_index('cohort')
                     .groupby(level=0)['subID']
                     .unique()
                     .to_dict())

    # validate
    lower = set(cohort_groups['LowerKnowledge'])
    upper = set(cohort_groups['UpperKnowledge'])
    # cohorts are unique:
    assert len(lower.intersection(upper)) == 0
    # cohorts are exhaustive:
    assert len(set(_metadata['subID']) - lower.union(upper)) == 0

    # arrays → lists (for cleaner YAML writes)
    cohort_groups = {key: value.tolist() for key, value in
                     cohort_groups.items()}
    camp_groups = {key: value.tolist() for key, value in camp_groups.items()}

    # save
    mode = 'w' if cohort == cohorts[0] else 'a'
    fnames = ('letter_knowledge_cohorts.yaml', 'intervention_cohorts.yaml')
    for fname, _dict in zip(fnames, (cohort_groups, camp_groups)):
        with open(os.path.join('..', '..', 'params', fname), mode) as f:
            yaml.dump({cohort: _dict}, stream=f)
