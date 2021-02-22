#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import mne
from mnefun._paths import (get_raw_fnames, get_event_fnames)
import expyfun

# INCOMING EVENT CODES
# ====================
# words = 1
# faces = 2
# cars = 3 (channels 1 and 2)
# alien = 4
# PS = 1  } first 3 trials are always PS
# KT = 1  } last 3 trials are always KT


def prek_score(p, subjects):
    for si, subject in enumerate(subjects):
        fnames = get_raw_fnames(p, subject, which='raw', erm=False,
                                add_splits=False, run_indices=None)
        event_fnames = get_event_fnames(p, subject, run_indices=None)
        for fname, event_fname in zip(fnames, event_fnames):
            raw = mne.io.read_raw_fif(fname, allow_maxshield=True)
            sfreq = raw.info['sfreq']

            # original trials 20 s; add events to later split into 5 s epochs
            if 'pskt' in fname:
                events = mne.find_events(raw, shortest_event=1, mask=1)
                assert events.shape[0] == 6
                events[2, :3] = 60
                events[2, 3:] = 70
                new_events = list()
                for row in events:
                    new_events.append(row)
                    if row[2] in (60, 70):
                        offsets = np.round(np.arange(1, 4) * 5 * sfreq
                                           ).astype(int)
                        for offset in offsets:
                            new_row = row + np.array([offset, 0, 0])
                            new_events.append(new_row)
                events = np.array(new_events)
            else:
                # split events for behavioral scoring
                presses = mne.find_events(raw, shortest_event=1, mask=240)
                alien = mne.find_events(raw, shortest_event=1, mask=4)
                # a mask of 3 will include events 1 & 2 also, and masks of 1 or
                # 2 will each include event 3 also, so we need to segregate
                wordsfacescars = mne.find_events(raw, shortest_event=1, mask=3)
                words = wordsfacescars[wordsfacescars[:, 2] == 1]
                faces = wordsfacescars[wordsfacescars[:, 2] == 2]
                cars = wordsfacescars[wordsfacescars[:, 2] == 3]
                # ensure the timestamps are distinct
                assert not np.in1d(words[:, 0], cars[:, 0]).any()
                assert not np.in1d(words[:, 0], faces[:, 0]).any()
                assert not np.in1d(cars[:, 0], faces[:, 0]).any()
                assert not np.in1d(alien[:, 0], cars[:, 0]).any()
                assert not np.in1d(alien[:, 0], faces[:, 0]).any()
                assert not np.in1d(alien[:, 0], words[:, 0]).any()
                # recode
                presses[:, 2] = 5
                events = np.concatenate((words, cars, faces, alien, presses))
                events[:, 2] *= 10
                events = events[events[:, 0].argsort()]  # sort chronologically
            # write events, and bail early if we don't need to analyze presses
            mne.write_events(event_fname, events)
            if 'pskt' in fname:
                return
            # if ERP, write the behavioral data
            hits = []
            misses = []
            correct_rejections = 0
            images = np.concatenate((words, cars, faces))
            all_events = mne.find_events(raw, shortest_event=1)

            for event in all_events:
                if event[0] in presses:
                    event[2] = 5
            all_events[:, 2] *= 10

            for i, event in enumerate(all_events):
                if np.in1d(event[0], alien):  # for each alien image...
                    # if the alien image is the very last event, its a miss
                    if event[0] == all_events[-1, 0]:
                        print('Miss', event, i)
                        misses.append(event)
                    # if the next event isn't a button press, it's a miss.
                    elif all_events[i+1][2] != 50:
                        print('Miss', event, i)
                        misses.append(event)
                    else:   # add the next event to hits if it's a button press
                        if all_events[i+1][2] == 50:
                            hits.append(all_events[i + 1])
                            print('hit: %s' % all_events[i+1], i+1)
                        else:
                            continue

                if np.in1d(event[0], images):  # for each non-alien image
                    try:
                        # if the kid doesn't press a button next...
                        if all_events[i+1][2] != 50:
                            correct_rejections += 1  # ... it's a correct rej.
                    except IndexError:
                        # if image is last event, there was no press after it
                        correct_rejections += 1

            extras = [x for x in all_events if x[2] == 50 and not
                      np.in1d(x[0], hits)]

            hits = len(hits)
            misses = len(misses)
            false_alarms = len(extras)

            d_prime = expyfun.analyze.dprime([hits, misses, false_alarms,
                                              correct_rejections])

            results_line = (f'{len(presses)}, {hits}, {misses}, '
                            f'{false_alarms}, {correct_rejections}, {d_prime}')
            with open(os.path.join(p.work_dir, subject,
                                   f'{subject}_behavioral.txt'), 'wb') as fid:
                fid.write('total_presses, hits, misses, false_alarms, '
                          'correct_rejections, d_prime: \n'.encode())
                fid.write(results_line.encode())

            with open(os.path.join(p.work_dir, 'pre_behavioral.txt'), 'ab') as fid:  # noqa E501
                print('Adding a line to the global behavioral file.')
                fid.write(f'{subject}, {results_line} \n'.encode())


def pick_cov_events_prek(events):
    # we only want visual events for baseline, not button presses
    events = [x for x in events if x[2] != 5]
    return events
