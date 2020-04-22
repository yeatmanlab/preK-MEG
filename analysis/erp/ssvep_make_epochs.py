#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract SSVEP epochs, downsample, and save to disk.
"""

import os
import mne
from aux_functions import load_paths, load_params

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True
n_jobs = 10

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
        # downsample
        raw, events = raw.resample(sfreq=50, events=events, n_jobs='cuda')
        # clean up
        del raws, first_samps, last_samps, events_list
        # epoch
        event_dict = dict(ps=5, kt=7)
        epochs = mne.Epochs(raw, events, event_dict, tmin=0, tmax=trial_dur,
                            baseline=None, proj=True,  # reject=reject_dict,
                            reject_by_annotation=False, preload=True)
        # save epochs
        fname = f'{s}-{timepoint}_camp-pskt-epo.fif'
        epochs.save(os.path.join(epo_dir, fname), fmt='double')
        # PSD settings
        psd_kwargs = dict(fmin=0, fmax=20, bandwidth=0.1, adaptive=False,
                          n_jobs=n_jobs)
