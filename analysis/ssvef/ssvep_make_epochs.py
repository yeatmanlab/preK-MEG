#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Extract SSVEP epochs, downsample, and save to disk.
"""

import os
import numpy as np
import mne
from sswef_helpers.aux_functions import (load_paths, load_params,
                                    PREPROCESS_JOINTLY, yamload)

mne.cuda.init_cuda()

# flags
compute_psds = True
plot_psds = True
plot_topomaps = True

# config paths
data_root, subjects_dir, results_dir = load_paths()
epo_dir = os.path.join(results_dir, 'pskt', 'epochs')
os.makedirs(epo_dir, exist_ok=True)
subfolder = 'combined' if PREPROCESS_JOINTLY else 'pskt'

# load params
*_, subjects, cohort = load_params(experiment='pskt')
paramfile = os.path.join('..', 'preprocessing', 'mnefun_common_params.yaml')
with open(paramfile, 'r') as f:
    params = yamload(f)
lp_cut = params['preprocessing']['filtering']['lp_cut']
resamp_sfreq = np.rint(2.1 * lp_cut).astype(int)

# config other
timepoints = ('pre', 'post')
runs = (1, 2)
tmax = 5  # seconds. Orig trial dur was 20 s but the score func subdivided it
event_dict = dict(ps=60, kt=70)

# loop over subjects
for s in subjects:
    for timepoint in timepoints:
        this_subj = os.path.join(data_root, f'{timepoint}_camp', 'twa_hp',
                                 subfolder, s)
        epochs_list = list()
        for run in runs:
            # read events from file made by the score func during preprocessing
            this_fname = f'ALL_{s}_pskt_{run:02}_{timepoint}-eve.lst'
            eve_path = os.path.join(this_subj, 'lists', this_fname)
            events = mne.read_events(eve_path)
            # load the raw file
            this_fname = f'{s}_pskt_{run:02}_{timepoint}_allclean_fil{lp_cut}_raw_sss.fif'  # noqa E501
            raw_path = os.path.join(this_subj, 'sss_pca_fif', this_fname)
            raw = mne.io.read_raw_fif(raw_path, preload=True)
            # downsample
            raw, events = raw.resample(sfreq=resamp_sfreq, events=events,
                                       n_jobs='cuda')
            # remove the `BAD_EOG_MANUAL` annotations (we don't want to reject
            # based on those, but do want to reject on, e.g., `BAD_ACQ_SKIP`)
            ann_to_del = list()
            for idx, ann in enumerate(raw.annotations):
                if ann['description'] == 'BAD_EOG_MANUAL':
                    ann_to_del.append(idx)
            raw.annotations.delete(ann_to_del)
            # epoch
            epo = mne.Epochs(raw, events, event_dict, tmin=0, tmax=tmax,
                             baseline=None, proj=True,
                             reject_by_annotation=True, preload=True)
            # trim last samp from epochs so our FFT bins come out nicely spaced
            epo.crop(tmax=tmax, include_tmax=False)
            assert len(epo.times) % 10 == 0
            epochs_list.append(epo)
        # combine runs (if there are multiple)
        epochs = mne.concatenate_epochs(epochs_list)
        # save epochs
        fname = f'{s}-{timepoint}_camp-pskt-epo.fif'
        epochs.save(os.path.join(epo_dir, fname), fmt='double', overwrite=True)
