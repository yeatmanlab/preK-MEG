#!/usr/bin/env python

from functools import partial
import numpy as np
from scipy.stats import zscore
import pandas as pd
import seaborn as sns


data = pd.read_csv('LetterKnowledge.csv')

df = data.melt(id_vars='subID', value_vars=['UpperName', 'LowerName',
                                            'UpperSound', 'LowerSound'])
df.set_index('subID', inplace=True)

# raw scores
g = sns.FacetGrid(df, col='variable')
bins = np.arange(27)
g.map(sns.distplot, 'value', bins=bins)

# z-scored
dfz = df.copy()
dfz['value'] = dfz.groupby('variable').transform(partial(zscore, ddof=1))
g = sns.FacetGrid(dfz, col='variable')
bins = np.linspace(-4, 4, 81)
g.map(sns.distplot, 'value', bins=bins)

# z-scored and averaged
dfm = dfz.reset_index().groupby('subID')['value'].agg('mean')
sns.distplot(dfm, bins=bins)

# put it back together
df = df.pivot(columns='variable', values='value')
data.set_index('subID', inplace=True)
data['newmean'] = dfm
