# Analysis pipeline for evoked responses

1. Annotate the raw recordings to exclude noisy spans on the EOG channel (so
   that SSP projectors will actually reflect blink events, not noise). This is
   done with `prek_annotate_interactive.sh`, which is a wrapper for
   `prek_annotate_interactive.py`. Saves the annotations alongside the raw
   recordings, in `*-custom-annot.fif` files

2. Run `prek_mnefun.py` (semi-supervised preprocessing). This requires:
    - The `-custom-annot.fif` files (see above)
    - `prek_score.py` (scoring function, called by `prek_mnefun.py`)
    - `prek_setup_source.py` (makes hi-res brain model for each subject,
      creates a source space, plots the source space for quality check)

3. Preliminary analysis: `prek_make_stcs.py` (makes STC for each subj, morphs
   it to fsaverage, makes subj avg for each experimental condition)

4. Planned analyses:
    - `prek_make_group_averages.py`
    - `prek_contrast_conds.py`
