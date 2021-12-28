#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
from functools import partial
import numpy as np
from mne import read_source_spaces, add_source_space_distances

paramdir = os.path.join('..', '..', 'params')
yamload = partial(yaml.load, Loader=yaml.FullLoader)

with open(os.path.join(paramdir, 'current_cohort.yaml'), 'r') as f:
    cohort = yamload(f)

PREPROCESS_JOINTLY = False  # controls folder path


def load_params(skip=True, experiment=None):
    """Load experiment parameters from YAML files."""
    with open(os.path.join(paramdir, 'brain_plot_params.yaml'), 'r') as f:
        brain_plot_kwargs = yamload(f)
    with open(os.path.join(paramdir, 'movie_params.yaml'), 'r') as f:
        movie_kwargs = yamload(f)
    subjects = load_subjects(cohort, experiment, skip)
    return brain_plot_kwargs, movie_kwargs, subjects, cohort


def load_subjects(cohort, experiment=None, skip=True):
    with open(os.path.join(paramdir, 'subjects.yaml'), 'r') as f:
        subjects_dict = yamload(f)
    if cohort == 'pooled':
        subjects = sum(subjects_dict.values(), [])
    else:
        subjects = list(subjects_dict[cohort])
    # skip bad subjects
    if skip:
        skips = _get_skips(experiment)
        subjects = sorted(set(subjects) - skips)
    return subjects


def load_fsaverage_src():
    """Load fsaverage source space."""
    _, subjects_dir, _ = load_paths()
    fsaverage_src_path = os.path.join(subjects_dir, 'fsaverage', 'bem',
                                      'fsaverage-ico-5-src.fif')
    fsaverage_src = read_source_spaces(fsaverage_src_path)
    fsaverage_src = add_source_space_distances(fsaverage_src, dist_limit=0)
    return fsaverage_src


def _get_skips(experiment):
    all_experiments = ('erp', 'pskt')
    if experiment in all_experiments:
        experiment = (experiment,)
    elif experiment == 'combined':
        experiment = all_experiments
    elif experiment is None:
        raise ValueError('must pass "experiment" when skipping subjects')
    skips = set()
    for exp in experiment:
        fname = f'skip_subjects_{exp}.yaml'
        with open(os.path.join(paramdir, fname), 'r') as f:
            new_skips = yamload(f)
            skips = skips.union(new_skips)
    return skips


def load_psd_params():
    """Load experiment parameters from YAML files."""
    with open(os.path.join(paramdir, 'psd_params.yaml'), 'r') as f:
        psd_params = yamload(f)
    return psd_params


def load_inverse_params():
    """Load inverse parameters from YAML file."""
    with open(os.path.join(paramdir, 'inverse_params.yaml'), 'r') as f:
        inverse_params = yamload(f)
    return inverse_params


def load_cohorts(experiment=None):
    """load intervention and knowledge groups."""
    with open(os.path.join(paramdir, 'intervention_cohorts.yaml'), 'r') as f:
        intervention_groups = yamload(f)[cohort]
    with open(os.path.join(paramdir, 'letter_knowledge_cohorts.yaml'),
              'r') as f:
        letter_knowledge_groups = yamload(f)[cohort]
    skips = _get_skips(experiment)
    for _dict in (intervention_groups, letter_knowledge_groups):
        for _group in _dict:
            # convert 1103 → 'prek_1103'
            _dict[_group] = [f'prek_{num}' for num in _dict[_group]]
            # remove anyone in skip_subjects.yaml from the cohorts
            _dict[_group] = sorted(set(_dict[_group]) - skips)
    return intervention_groups, letter_knowledge_groups


def load_paths():
    """Load necessary filesystem paths."""
    with open(os.path.join(paramdir, 'paths.yaml'), 'r') as f:
        paths = yamload(f)
    with open(os.path.join(paramdir, 'results-path-identifier.yaml'), 'r'
              ) as f:
        results_path_id = yamload(f)
    if len(results_path_id):
        results_path_id = f'-{results_path_id}'
    paths['results_dir'] = os.path.join(
        paths['results_dir'], f'{cohort}{results_path_id}')
    return paths['data_root'], paths['subjects_dir'], paths['results_dir']


def prep_cluster_stats(cluster_results):
    (tvals, clusters, cluster_pvals, hzero) = cluster_results
    stats = dict(n_clusters=len(clusters),
                 clusters=clusters,
                 tvals=tvals,
                 pvals=cluster_pvals,
                 # this is multicomparison-corrected already:
                 good_cluster_idxs=np.where(cluster_pvals < 0.05),
                 hzero=hzero)
    return stats


def define_labels(region, action, hemi, subjects_dir=None):
    from mne import read_labels_from_annot
    if action not in ('include', 'exclude'):
        raise ValueError('action must be "include" or "exclude".')

    region_dict = {'medial-wall': ('G_and_S_paracentral',      # 3
                                   'G_and_S_cingul-Ant',       # 6
                                   'G_and_S_cingul-Mid-Ant',   # 7
                                   'G_and_S_cingul-Mid-Post',  # 8
                                   'G_cingul-Post-dorsal',     # 9
                                   'G_cingul-Post-ventral',    # 10
                                   'G_front_sup',              # 16
                                   'G_oc-temp_med-Parahip',    # 23
                                   'G_precuneus',              # 30
                                   'G_rectus',                 # 31
                                   'G_subcallosal',            # 32
                                   'S_cingul-Marginalis',      # 46
                                   'S_pericallosal',           # 66
                                   'S_suborbital',             # 70
                                   'S_subparietal',            # 71
                                   'Unknown',
                                   ),
                   'VOTC': ('G_and_S_occipital_inf',      # 2
                            'G_oc-temp_lat-fusifor',      # 21
                            'G_temporal_inf',             # 37
                            'S_collat_transv_ant',        # 50
                            'S_collat_transv_post',       # 51
                            'S_occipital_ant',            # 59
                            'S_oc-temp_lat',              # 60
                            'S_oc-temp_med_and_Lingual',  # 61
                            'S_temporal_inf',             # 72
                            )
                   }

    valid_regions = tuple(region_dict)
    if region not in valid_regions + (None,):
        raise ValueError(f'region must be "{", ".join(valid_regions)}", or '
                         'None.')

    label_names = region_dict[region]
    regexp = r'|'.join(label_names)
    # DON'T DO IT THIS WAY (PARCELLATION NOT GUARANTEED TO BE EXHAUSTIVE):
    # if action == 'include':
    #     # select for exclusion all labels *except* those given
    #     regexp = f'(?!{regexp})'
    labels = read_labels_from_annot(subject='fsaverage',
                                    parc='aparc.a2009s',
                                    hemi=hemi,
                                    subjects_dir=subjects_dir,
                                    regexp=regexp)
    n_hemis = 2 if hemi == 'both' else 1
    assert len(labels) == n_hemis * len(label_names)
    # merge the labels using the sum(..., start) hack
    merged_label = sum(labels[1:], labels[0])
    assert len(merged_label.name.split('+')) == n_hemis * len(label_names)
    return merged_label


def get_stc_from_conditions(method, timepoint, condition, subject):
    """Load an STC file for the given experimental conditions.

    Parameters
    ----------

    method : 'dSPM' | 'sLORETA'

    timepoint : 'pre' | 'post'

    condition : 'words' | 'faces' | 'cars' | 'aliens' | 'ps' | 'kt' | 'all'
        Note that "all" means "ps" + "kt".

    subject : str
        Can be a subject identifier ('prek_1103'), a group name ('language',
        'letter', 'upper', or 'lower'), or `None` (to get grand average of all
        subjects).
    """
    from mne import read_source_estimate
    data_root, _, results_dir = load_paths()
    # allow both groups and subjects as the "subject" argument
    group_map = {None: 'GrandAvg',
                 'language': 'LanguageInterventionN24',
                 'letter': 'LetterInterventionN24',
                 'upper': 'UpperKnowledgeN24',
                 'lower': 'LowerKnowledgeN24'}
    # if "subject" is not in group_map, use it as-is
    subject = group_map.get(subject, subject)
    # filename pattern
    fname = f'{subject}FSAverage_{timepoint}Camp_{method}_{condition}'
    if subject in group_map.values():
        folder = os.path.join(results_dir, 'group_averages')
    elif condition in ('words', 'faces', 'cars', 'aliens'):
        folder = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp', 'erp',
                              subject, 'stc')
    else:
        fname = f'{subject}FSAverage-{timepoint}_camp-pskt-{condition}-fft-stc'
        chosen_constraints = ('{orientation_constraint}-{estimate_type}'
                              ).format_map(load_inverse_params())
        folder = os.path.join(results_dir, 'pskt', 'stc',
                              'morphed-to-fsaverage', chosen_constraints)
    stc_path = os.path.join(folder, fname)
    stc = read_source_estimate(stc_path)
    if method == 'snr':
        stc.data = div_by_adj_bins(np.abs(stc.data))
    return stc


def get_dataframe_from_label(label, src, methods=('dSPM', 'MNE'),
                             timepoints=('pre', 'post'),
                             conditions=('words', 'faces', 'cars', 'aliens'),
                             subjects=None, unit='time',
                             experiment=None):
    """Get average timecourse within label across all subjects."""
    from pandas import DataFrame, concat, melt
    from mne import Label, extract_label_time_course
    # allow looping over single Label
    if isinstance(label, Label):
        label = [label]
    # load subjects list
    if subjects is None:
        _, _, subjects, _ = load_params(experiment=experiment)
    elif isinstance(subjects, str):
        subjects = [subjects]
    # load cohort information
    intervention_group, letter_knowledge_group = load_cohorts(experiment)
    intervention_map = {subj: group.lower()[:-12]
                        for group, members in intervention_group.items()
                        for subj in members}
    knowledge_map = {subj: group.lower()[:-9]
                     for group, members in letter_knowledge_group.items()
                     for subj in members}

    time_courses = dict()
    # loop over source localization algorithms
    for method in methods:
        time_courses[method] = dict()
        # loop over pre/post measurement time
        for timept in timepoints:
            time_courses[method][timept] = dict()
            # loop over conditions
            for cond in conditions:
                time_courses[method][timept][cond] = dict()
                # loop over subjects
                for subj in subjects:
                    # load STC
                    stc = get_stc_from_conditions(method, timept, cond, subj)
                    # extract label time course
                    time_courses[method][timept][cond][subj] = np.squeeze(
                        extract_label_time_course(stc, label, src=src,
                                                  mode='mean'))
                # convert dict of each subj's time series to DataFrame
                df = DataFrame(time_courses[method][timept][cond],
                               index=range(len(stc.times)))
                df[unit] = stc.times
                time_courses[method][timept][cond] = df
                # store current "condition" in a column before exiting loop
                time_courses[method][timept][cond]['condition'] = cond
            # combine DataFrames across conditions
            dfs = (time_courses[method][timept][c] for c in conditions)
            time_courses[method][timept] = concat(dfs)
            # store current "timepoint" in a column before exiting loop
            time_courses[method][timept]['timepoint'] = timept
        # combine DataFrames across timepoints
        dfs = (time_courses[method][t] for t in timepoints)
        time_courses[method] = concat(dfs)
        # store current "method" in a column before exiting loop
        time_courses[method]['method'] = method
    # combine DataFrames across methods
    dfs = (time_courses[m] for m in methods)
    time_courses = concat(dfs)

    # reshape DataFrame
    all_cols = time_courses.columns.values
    subj_cols = time_courses.columns.str.startswith('prek')
    id_vars = all_cols[np.logical_not(subj_cols)]
    time_courses = melt(time_courses, id_vars=id_vars, var_name='subj')
    # add columns for intervention cohort and pretest letter knowledge
    time_courses['intervention'] = time_courses['subj'].map(intervention_map)
    time_courses['pretest'] = time_courses['subj'].map(knowledge_map)
    return time_courses


def set_brain_view_distance(brain, views, hemi, distance):
    # zoom out so the brains aren't cut off or overlapping
    if hemi == 'split':
        views = np.tile(views, (2, 1)).T
    else:
        views = np.atleast_2d(views).T
    for rix, row in enumerate(views):
        for cix, view in enumerate(row):
            brain.show_view(view, row=rix, col=cix, distance=distance)


def plot_label(label, img_path, alpha=1., **kwargs):
    from mne.viz import Brain
    defaults = dict(surf='inflated')
    defaults.update(kwargs)
    brain = Brain('fsaverage', **defaults)
    brain.add_label(label, alpha=alpha)
    brain.save_image(img_path)
    return img_path


def plot_label_and_timeseries(label, img_path, df, method, groups, timepoints,
                              conditions, all_timepoints, all_conditions,
                              cluster=None, lineplot_kwargs=None, unit='time'):
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import seaborn as sns

    # defaults
    lineplot_kwargs = dict() if lineplot_kwargs is None else lineplot_kwargs

    # triage groups
    all_interventions = ('letter', 'language')
    all_pretest_cohorts = ('lower', 'upper')
    if groups[0] in all_interventions:
        all_groups = all_interventions
    elif groups[0] in all_pretest_cohorts:
        all_groups = all_pretest_cohorts
    else:
        all_groups = ['grandavg']

    # plot setup
    sns.set(style='whitegrid', font_scale=0.8)
    grey_vals = ['0.75', '0.55', '0.35']
    color_vals = ['#004488', '#bb5566', '#ddaa33']
    gridspec_kw = dict(height_ratios=[4] + [1] * len(all_groups))
    fig, axs = plt.subplots(len(all_groups) + 1, 1, figsize=(9, 13),
                            gridspec_kw=gridspec_kw)
    title_dict = dict(language='Language Intervention cohort',
                      letter='Letter Intervention cohort',
                      grandavg='All participants',
                      lower='Pre-test lower half of participants',
                      upper='Pre-test upper half of participants')

    # draw the brain/cluster image into first axes
    cluster_image = imread(img_path)
    axs[0].imshow(cluster_image)
    axs[0].set_axis_off()
    axs[0].set_title(os.path.split(img_path)[-1])

    # plot
    for group, ax in zip(all_groups, axs[1:]):
        # plot cluster-relevant lines in color, others gray (unless
        # we're plotting the non-cluster-relevant group → all gray)
        colors = [color_vals[i]
                  if c in conditions and group in groups else
                  grey_vals[i]
                  for i, c in enumerate(all_conditions)]
        # get just the data for this group
        if group in all_interventions:
            group_column = df['intervention']
        elif group in all_pretest_cohorts:
            group_column = df['pretest']
        else:
            group_column = np.full(df.shape[:1], 'grandavg')
        data = df.loc[(group_column == group) &
                      np.in1d(df['timepoint'], timepoints) &
                      np.in1d(df['condition'], all_conditions)]
        # draw
        with sns.color_palette(colors):
            sns.lineplot(x=unit, y='value', data=data, ax=ax,
                         **lineplot_kwargs)
        # indicate temporal span of cluster signif. difference
        if group in groups and cluster is not None:
            temporal_idxs, _ = cluster
            times = np.sort(df[unit].unique())
            xmin = times[temporal_idxs.min()]
            xmax = times[temporal_idxs.max()]
            ax.fill_betweenx((0, 4), xmin, xmax, color='k', alpha=0.1)
        # garnish
        ymax = dict(MNE=1e-10, dSPM=4, sLORETA=2, fft=300, snr=8)[method]
        ax.set_ylim(0, ymax)
        ax.set_title(title_dict[group])
        # suppress x-axis label on upper panel
        if ax == axs[-2]:
            ax.set_xlabel('')
        # force legend to right edge, by removing and re-adding it
        ax.legend_.remove()
        ax.legend(loc='right')
    # save plot (overwrites the cluster image PNG)
    sns.despine()
    fig.savefig(img_path)
    plt.close(fig)


def subdivide_epochs(epochs, divisions):
    """Reshape epochs data to get different numbers of epochs."""
    from mne import EpochsArray
    if epochs.times.size != 1000:
        # cut off last sample
        epochs.crop(None, epochs.times[-2])
        assert epochs.times.size == 1000
    # reshape epochs data to get different numbers of epochs
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    assert n_times % divisions == 0
    new_n_times = n_times // divisions
    new_shape = (n_epochs, n_channels, divisions, new_n_times)
    data = np.reshape(data, new_shape)
    data = data.transpose(0, 2, 1, 3)
    data = np.reshape(data, (divisions * n_epochs, n_channels, new_n_times))
    recut_epochs = EpochsArray(data, epochs.info)
    return recut_epochs


def div_by_adj_bins(data, n_bins=2, method='mean', return_noise=False):
    """
    data : np.ndarray
        the data to enhance
    n_bins : int
        number of bins on either side to include.
    method : 'mean' | 'sum'
        whether to divide by the sum or average of adjacent bins.
    """
    from scipy.ndimage import convolve1d
    assert data.dtype == np.float64
    weights = np.ones(2 * n_bins + 1)
    weights[n_bins] = 0  # don't divide target bin by itself
    if method == 'mean':
        weights /= 2 * n_bins
    noise = convolve1d(data, mode='constant', weights=weights)
    return noise if return_noise else data / noise


def nice_ticklabels(ticks, n=2):
    return list(
        map(str, [int(t) if t == int(t) else round(t, n) for t in ticks]))
