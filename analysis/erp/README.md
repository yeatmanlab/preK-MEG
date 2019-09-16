# Analysis pipeline for evoked responses

- `prek_mnefun.py` (semi-supervised preprocessing)
    - `prek_score.py` (scoring function, called by above)
    - `prek_setup_source.py` (makes hi-res brain model for each subject, create source space, plot source space for quality check)
- `prek_make_stcs.py` (makes STC for each subj, morphs to fsaverage, makes subj avg for each condition)

