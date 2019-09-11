#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os.path as op
import mne
import numpy as np
from mnefun._paths import (get_raw_fnames, get_event_fnames)
import expyfun


# words = 1
# faces = 2
# cars = 3 (channels 1 and 2)
# alien = 4


def prek_score(p, subjects):
    for si, subject in enumerate(subjects):
        fnames = get_raw_fnames(p, subject, which='raw', erm=False, add_splits=False, run_indices=None)
        event_fnames = get_event_fnames(p, subject, run_indices=None)
        for fi, fname in enumerate(fnames):
            raw = mne.io.read_raw_fif(fname, allow_maxshield=True)

            # events are split for behavioral scoring
            words = mne.find_events(raw, shortest_event=2, mask=1)
            faces = mne.find_events(raw, shortest_event=2, mask=2)
            cars = mne.find_events(raw, shortest_event=2, mask=3)
            alien = mne.find_events(raw, shortest_event=2, mask=4)

            cars = [x for x in cars if x[2] == 3]
            words = [x for x in words if not np.in1d(x[0], cars)]
            faces = [x for x in faces if not np.in1d(x[0], cars)]

            words = np.array(words)
            faces = np.array(faces)
            cars = np.array(cars)
            alien = np.array(alien)

            # check that these events have distinct timestamps
            assert not np.in1d(words[:, 0], cars[:, 0]).any()
            assert not np.in1d(words[:, 0], faces[:, 0]).any()
            assert not np.in1d(cars[:, 0], faces[:, 0]).any()
            assert not np.in1d(alien[:, 0], cars[:, 0]).any()
            assert not np.in1d(alien[:, 0], faces[:, 0]).any()
            assert not np.in1d(alien[:, 0], words[:, 0]).any()

            # find button presses and turn them all into events with a value of 5
            presses = mne.find_events(raw, shortest_event=2, mask=240)
            presses[:, 2] = 5

            # return all events
            events = np.concatenate((words, cars, faces, alien, presses))
            events[:, 2] *= 10
            events = events[events[:, 0].argsort()]  # sort chronologically
            mne.write_events(event_fnames[fi], events)

            # write the behavioral data
            hits = []
            misses = []
            correct_rejections = 0
            images = np.concatenate((words, cars, faces))
            all_events = mne.find_events(raw, shortest_event=2)

            for event in all_events:
                if event[0] in presses:
                    event[2] = 5
            all_events[:, 2] *= 10

            for i, event in enumerate(all_events):
                if np.in1d(event[0], alien): # for each alien image
                    if event[0] == all_events[-1, 0]:  # if the alien image is the very last event, its a miss
                        print('Miss', event, i)
                        misses.append(event)
                    elif all_events[i+1][2] != 50:  # if the next event isn't a button press, it's a miss.
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
                        if all_events[i+1][2] != 50:   # if the kid doesn't press a button next
                            correct_rejections += 1           # then it's a correct rejection
                    except IndexError:
                        correct_rejections += 1   # if an image is the last event, then there's no button press following

            extras = [x for x in all_events if x[2] == 50 and not np.in1d(x[0], hits)]

            hits = len(hits)
            misses = len(misses)
            false_alarms = len(extras)

            d_prime = expyfun.analyze.dprime([hits, misses, false_alarms, correct_rejections])

            with open(op.join(p.work_dir, subject, '%s_behavioral.txt' % subject), 'wb') as fid:
                print('writing the behavioral csv at %s' % op.join(subject, '%s_behavioral.txt' % subject))
                fid.write('total_presses, hits, misses, false_alarms, correct_rejections, d_prime: \n'.encode())
                fid.write((('%s, %s, %s, %s, %s, %s' % (len(presses), hits, misses, false_alarms, correct_rejections, d_prime)).encode()))

            with open(op.join(p.work_dir,'pre_behavioral.txt'), 'ab') as fid:
                print('Adding a line to the global behavioral file.')
                fid.write((('%s, %s, %s, %s, %s, %s, %s \n' % (subject, len(presses), hits, misses, false_alarms, correct_rejections, d_prime)).encode()))



def pick_cov_events_prek(events):
    events = [x for x in events if x[2] != 5] # we only want visual events for baseline, not button presses
    return events