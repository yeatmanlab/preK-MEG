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

3. Preliminary aggregation:
    - `prek_make_stcs.py` (makes STC for each subj, morphs it to fsaverage,
      makes subj avg for each experimental condition). Results are saved in
      `/mnt/scratch/prek/<PRE_OR_POST_CAMP>/twa_hp/erp/<SUBJ>/stc`.
    - `prek_make_group_averages.py` (makes average STC for each experimental
      condition, across all subjects, separately for pre- and post-intervention
      recordings). Writes group-average STCs and movies to
      `/mnt/scratch/prek/results/group_averages` and `.../movies`
      (respectively).

4. Preliminary analyses:
    - `prek_contrast_conds.py` makes across-subject-average STCs (separately
      for pre- and post-camp recordings) for each pairwise condition contrast.

    - `prek_contrast_prepost.py` makes post-camp-minus-pre-camp,
      across-subject-average STCs for each condition and for each pairwise
      condition contrast.

5. Planned analyses:
    - `prek_contrast_letter_knowledge.py` makes group comparison STCs between
      the top and bottom half of the participant cohort, as determined by
      letter-name and letter-sound awareness of both uppercase and lowercase
      glyphs (this data stored in `../behavioral/LetterKnowledge.csv`). This
      group comparison is done only for the pre-camp MEG recordings.

    - `script-not-yet-written.py` makes group comparison STCs depending on
      which treatment group the subject was assigned to ("language"
      intervention or "literacy" intervention; cohort information is stored in
      `../behavioral/LetterKnowledge.csv`). This group comparison is done only
      for the post-camp MEG recordings and the post-camp-minus-pre-camp
      differences.


## Filename conventions

Source-space files follow the convention:

```python
f'{subject}_{timepoint}_{method}_{condition}.fileExtension'
```

- for group averages in source space, activity is morphed to a template brain
  and hence `subject` will always include `FSAverage` (referring to the
  FreeSurfer average template brain). File names for group averages also
  include the number of participants averaged, so an average based on all 48
  participants will be called `GrandAvgN48FSAverage`. Averages of more
  restricted subsets of the participant population will not say "GrandAvg" but
  instead use some descriptor of the group selection criteria (e.g.,
  "UpperQuartileN12FSAverage")
- `timepoint` takes values `preCamp`, `postCamp`, or `PostCampMinusPreCamp`,
  and indicates whether the data were acquired before or after the
  intervention.
- `method` refers to the source localization method (e.g., `dSPM`, `sLORETA`,
  etc)
- `condition` refers to the experimental conditions, e.g., `faces`, `words`,
  `cars`, and `aliens`. Files showing differences between conditions use the
  convention `ConditionOneMinusConditionTwo`, e.g., `FacesMinusCars`.
