#!/usr/bin/env python

import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h5io import write_hdf5
from mne.stats import permutation_cluster_test

rng = np.random.default_rng(seed=15485863)  # the one millionth prime

# load raw data
fname = 'roi-MPM_IOS_IOG_pOTS_lh-timeseries-long.csv'
dtypes = dict(
    time=float,
    value=float,
    roi=pd.StringDtype(),
    subj=pd.StringDtype(),
    method=pd.StringDtype(),
    pretest=pd.CategoricalDtype(categories=['lower', 'upper'], ordered=True),
    timepoint=pd.CategoricalDtype(categories=['pre', 'post'], ordered=True),
    condition=pd.CategoricalDtype(categories=['words', 'faces', 'cars']),
    intervention=pd.CategoricalDtype(categories=['letter', 'language']),
)
rawdata = pd.read_csv(fname, index_col=0, dtype=dtypes)

# filter to just what we want
modeldata = rawdata.loc[
    (rawdata['roi'] == 'MPM_IOS_IOG_pOTS_lh') &
    (rawdata['method'] == 'dSPM') &  # drop "MNE"
    (rawdata['condition'].notna())  # drops "aliens"
]

# simplify to look at the interactions we want
modeldata = modeldata.pivot(
    columns='timepoint', values='value',
    index=['time', 'condition', 'subj', 'intervention']
)
modeldata['post_minus_pre'] = modeldata['post'] - modeldata['pre']
modeldata = (modeldata.drop(['pre', 'post'], axis='columns')
                      .reset_index('condition')
                      .pivot(columns='condition', values='post_minus_pre')
                      .rename_axis(None, axis='columns'))
modeldata['words_minus_faces'] = modeldata['words'] - modeldata['faces']
modeldata['words_minus_cars'] = modeldata['words'] - modeldata['cars']
modeldata = (modeldata.drop(['words', 'faces', 'cars'], axis='columns')
                      .reset_index()
                      .sort_values(by=['intervention', 'subj', 'time'])
                      .reset_index(drop=True))

# extract the arrays
arrays = dict()
contrasts = ('cars', 'faces')
for contrast in contrasts:
    keep = f'words_minus_{contrast}'
    drop = f'words_minus_{list(set(contrasts) - set([contrast]))[0]}'
    arrays[keep] = dict()
    for intervention in ('letter', 'language'):
        arrays[keep][intervention] = dict()
        df = (modeldata[modeldata['intervention'] == intervention]
              .drop(['intervention', drop], axis='columns')
              .pivot(columns='time', values=keep, index='subj')
              .rename_axis(None, axis='columns'))
        arrays[keep][intervention]['subjs'] = df.index.to_list()
        # add extra dimension for "vertex/channel"
        arrays[keep][intervention]['data'] = df.to_numpy()[..., np.newaxis]
times = df.columns.to_numpy()

# clustering setup
n_jobs = 6
n_permutations = 10000
threshold = dict(start=0.02, step=0.02)
exclude = np.logical_or(times < 0, times > 0.5)[..., np.newaxis]

cluster_results = dict()
for contrast, _dict in arrays.items():
    X = [_dict['letter']['data'], _dict['language']['data']]
    f_obs, clusters, pvals, H0 = permutation_cluster_test(
        X, threshold=threshold, n_permutations=n_permutations, n_jobs=n_jobs,
        seed=rng, out_type='mask', step_down_p=0.05, exclude=exclude)
    cluster_results[contrast] = dict(
        f_obs=f_obs,
        clusters=clusters,
        pvals=pvals,
        H0=H0,
        subjs=dict(letter=_dict['letter']['subjs'],
                   language=_dict['language']['subjs'])
    )
    # save significant time span(s)
    mask = pvals < 0.05
    if mask.any():
        start_stop = np.flatnonzero(np.diff(mask)) + 1
        if mask[0]:
            start_stop = np.concatenate([0], start_stop)
        if mask[-1]:
            start_stop = np.concatenate(start_stop, [mask.size])
        start_stop = start_stop.reshape(-1, 2)
        signif_spans = [times[x].tolist() for x in start_stop]
        with open(f'{contrast}-signif-spans.yml', 'w') as f:
            yaml.dump(signif_spans, stream=f)

# save all results
write_hdf5('cluster-results.h5', cluster_results, overwrite=True)

# quick plot of the results
pad = 0.05
span_kw = dict(color='C1', alpha=0.1)
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
for contrast, ax in zip(cluster_results, axs):
    ax.plot(times, -np.log10(cluster_results[contrast]['pvals']))
    ax.set_title(' '.join(contrast.split('_')))
    ax.set_xlim(times[0], times[-1])
    # excluded regions
    ax.axvspan(xmin=times[0], xmax=times[~exclude[:, 0]][0], **span_kw)
    ax.axvspan(xmin=times[~exclude[:, 0]][-1], xmax=times[-1], **span_kw)
    ax.annotate(
        'excluded regions', xy=(0.99, 0.97), xycoords='axes fraction',
        va='top', ha='right', alpha=0.5, fontweight='black', color='C1')
    # pval cutoff
    for pval in (0.05,):
        neglogp = -np.log10(0.05)
        ax.axhline(neglogp, ls='--', lw=0.5, color='k')
        ax.text(times[-1], neglogp - pad, f'p={pval}', va='top', ha='right')
    # significant region
    mask = cluster_results[contrast]['pvals'] < 0.05
    if mask.any():
        signif_span = times[mask][[0, -1]]
        ax.axvspan(*signif_span, color='C0', alpha=0.1)
        ann = ('signif.\nregion\n' +
               '-\n'.join(map(str, (signif_span * 1e3).astype(int))) + ' ms')
        ax.annotate(
            ann, xy=(signif_span.mean(), 0.03), va='bottom', ha='center',
            xycoords=ax.get_xaxis_transform(), alpha=0.5, fontweight='black',
            color='C0')

ax.set_xlabel('time (s)')
fig.supylabel('\n-log10(pvalue)\n(post minus pre, cluster-corrected)',
              ha='center')
fig.subplots_adjust(left=0.15, right=0.95, hspace=0.4, top=0.92)
fig.savefig('cluster-results.png')
