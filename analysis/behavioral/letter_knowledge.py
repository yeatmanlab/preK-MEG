#!/usr/bin/env python
"""
Some exploratory plots for the behavioral data.
"""


import yaml
from functools import partial
import numpy as np
from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import os


# load the data
metadata = pd.read_csv('preK_InterventionData.csv')

# add info about cohort (upper/lower half) based on alphabet knowlege
cohort_fname = 'letter_knowledge_cohorts.yaml'
with open(os.path.join('..', '..', 'params', cohort_fname), 'r') as f:
    cohort_groups = yaml.load(f, Loader=yaml.SafeLoader)
mapping = {subj: key for key, value in cohort_groups.items() for subj in value}
metadata['cohort'] = metadata['subID'].map(mapping)

# remove NaNs
nan_rows = metadata[metadata.isna().any(axis=1)]
metadata_nonans = metadata[np.logical_not(metadata.isna()).all(axis=1)]

# variable correlations
value_vars = ['AlphabetKnowledge', 'Decoding', 'PhonemeMatching',
              'PhonemeSegmenting', 'EVTRaw', 'Retell', 'Grammar']
g = sns.pairplot(metadata_nonans, vars=value_vars, kind='reg', hue='cohort')

# reshape & zscore
mdf = metadata_nonans.melt(id_vars=['subID', 'visit'], value_vars=value_vars)
mdf['zscore'] = mdf.groupby('variable')['value'].transform(partial(zscore,
                                                                   ddof=1))

# variable distributions (z-scores)
g = sns.FacetGrid(mdf, col='variable')
bins = np.linspace(-5, 5, 101)
g.map(sns.distplot, 'zscore', bins=bins)

sns.lmplot('AlphabetKnowledge', 'PhonemeMatching', data=metadata)
