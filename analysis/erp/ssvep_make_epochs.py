#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract SSVEP epochs, downsample, and save to disk.
"""

import os
import numpy as np
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
epo_dir = os.path.join(results_dir, 'pskt', 'epochs')
fig_dir = os.path.join(results_dir, 'pskt', 'spectra', 'figs')
psd_dir = os.path.join(results_dir, 'pskt', 'spectra', 'psds')
for _dir in (epo_dir, fig_dir, psd_dir):
    os.makedirs(_dir, exist_ok=True)

# load params
_, _, subjects = load_params()

# config other
timepoints = ('pre', 'post')
runs = (1, 2)
trial_dur = 20  # seconds
# subdivide_epochs should be False or an integer number of seconds that evenly
# divides into trial_dur
subdivide_epochs = 5

# loop over subjects
for s in subjects:
    for timepoint in timepoints:
        this_subj = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                 'pskt', s)
        raws = list()
        events_list = list()
        first_samps = list()
        last_samps = list()
        # extract the events from the original raw (STIM channels are dropped
        # from the preprocessed raw) TODO: this will prob. change at some point
        for run in runs:
            this_fname = f'{s}_pskt_{run:02}_{timepoint}_raw.fif'
            raw_path = os.path.join(this_subj, 'raw_fif', this_fname)
            raw = mne.io.read_raw_fif(raw_path, allow_maxshield=True)
            eve = mne.find_events(raw, stim_channel='STI001')
            eve[3:, 2] = 7  # fix events to code for PS vs KT differently
            first_samps.append(raw.first_samp)
            last_samps.append(raw.last_samp)
            events_list.append(eve)
        events = mne.concatenate_events(events_list, first_samps, last_samps)
        # now process the clean files
        for run in runs:
            this_fname = f'{s}_pskt_{run:02}_{timepoint}_allclean_fil80_raw_sss.fif'  # noqa E501
            raw_path = os.path.join(this_subj, 'sss_pca_fif', this_fname)
            raw = mne.io.read_raw_fif(raw_path)
            raws.append(raw)
        # combine runs
        raw, events = mne.concatenate_raws(raws, events_list=events_list)
        # subdivide
        if subdivide_epochs:
            assert trial_dur % subdivide_epochs == 0
            n_new_events = trial_dur // subdivide_epochs
            t_offsets = np.arange(n_new_events, dtype=float) * subdivide_epochs
            initial_times = (events[:, 0] - raw.first_samp) / raw.info['sfreq']
            new_times = np.ravel(initial_times[:, np.newaxis] +
                                 t_offsets[np.newaxis, :])
            new_indices = (raw.time_as_index(new_times, use_rounding=True) +
                           raw.first_samp)
            # make sure the original event sample numbers didn't change
            assert len(set(events[:, 0]) - set(new_indices)) == 0
            events = np.column_stack((new_indices,
                                      np.zeros_like(new_indices),
                                      np.repeat(events[:, -1], n_new_events)))
        # downsample
        raw, events = raw.resample(sfreq=50, events=events, n_jobs='cuda')
        # clean up
        del raws, first_samps, last_samps, events_list
        # epoch
        event_dict = dict(ps=5, kt=7)
        epochs = mne.Epochs(raw, events, event_dict, tmin=0, tmax=trial_dur,
                            baseline=None, proj=True,
                            reject_by_annotation=False, preload=True)
        # save epochs
        subdiv = f'-{subdivide_epochs}_sec' if subdivide_epochs else ''
        fname = f'{s}-{timepoint}_camp-pskt{subdiv}-epo.fif'
        epochs.save(os.path.join(epo_dir, fname), fmt='double')
