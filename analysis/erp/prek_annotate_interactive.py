#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Daniel McCloy

Opens and plots MNE raw files for (re)annotation, and runs blink detection
before and after annotation, to check if the level of annotation is adequate
to allow the blink detection algorithm to perform reasonably well.

This script should be run from the command line, or (preferably) via the
shell-script wrapper `prek_annotate_interactive.sh`
"""

import sys
import mne
from mne.preprocessing import find_eog_events

# parse args
rawpath = sys.argv[1]
annpath = sys.argv[2]
rerun = bool(int(sys.argv[3]))
save = bool(int(sys.argv[4]))

# load raw & annotations
raw = mne.io.read_raw_fif(rawpath, allow_maxshield=True, verbose=False)
if rerun:
    ann = mne.annotations.read_annotations(annpath)
    raw.set_annotations(ann)

# see how well blink algorithm works before annotation
old_blink_events = find_eog_events(raw, reject_by_annotation=True,
                                   verbose=False)
old = old_blink_events.shape[0]
print('#############################')
print(f'old: {old}')
print('#############################')

# interactive annotation: mark bad channels & transient noise
order = list(range(len(raw.ch_names)))[::-1]
raw.plot(duration=30, events=old_blink_events, block=True,
         order=order, scalings=dict(mag=1e15, grad=1e13),
         n_channels=32)

accepted = False
while not accepted:
    # compare blink algorithm performance after annotation
    new_blink_events = find_eog_events(raw, reject_by_annotation=True,
                                       verbose=False)
    new = new_blink_events.shape[0]
    print('#############################')
    print(f'old: {old}')
    print(f'new: {new}')
    print('#############################')

    raw.plot(duration=30, events=new_blink_events, block=True,
             order=order, scalings=dict(mag=1e15, grad=1e13),
             n_channels=32)

    old = new
    satisfied = input('Satisfied? y/n  ')
    accepted = satisfied.lower().startswith('y')

# save annotations
if save and len(raw.annotations):
    raw.annotations.save(annpath, overwrite=True)
