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
      recordings). Also makes group averages for each intervention cohort and
      for the top/bottom half on the letter awareness pre-tests.
      Writes group-average STCs to `/mnt/scratch/prek/results/group_averages`

4. Contrasts:
    - The script `prek_do_contrasts.py`:
        - makes across-subject-average STCs (separately for pre- and post-camp
          recordings) for each pairwise condition contrast.
        - makes post-camp-minus-pre-camp, across-subject-average STCs for each
          condition and for each pairwise condition contrast.
        - makes group comparison STCs between the top and bottom half of the
          participant cohort, as determined by letter-name and letter-sound
          awareness of both uppercase and lowercase glyphs (this data stored in
          `../../params/letter_knowledge_cohorts.yaml`). This group comparison
          is done only for the pre-camp MEG recordings.
        - makes group comparison STCs depending on which treatment group the
          subject was assigned to ("language" intervention or "letter"
          intervention; cohort information is stored in
          `../../params/intervention_cohorts.yaml`). This group comparison is
          done only for the post-camp-minus-pre-camp differences.
    - STCs are saved to `/mnt/scratch/prek/results/group_averages`
    - Movies are made of every STC, and are saved to 
      `/mnt/scratch/prek/results/movies`

5. ROI analysis:
    - `../ROIs/create_ventral_band_labels.py` will generate the ROI labels
    - `prek_extract_ROI_time_courses.py` will extract average time course in
      each ROI label for each subject in each condition.

6. Clustering:
    - `prek_clustering.py` runs the clustering
    - `prek_plot_clusters_quick_and_dirty.py` generates cortical images of the 
      clusters, alongside plots of average time series extracted from the
      cluster-defined regions.

# Analysis pipeline for SSVEP

1. `ssvep_make_epochs.py`
    - optional: `ssvep_plot_phases.py`
    - optional: `ssvep_plot_sensor_psds.py`

2. `ssvep_epochs_to_stc.py` applies the FFT to the epoch data and then converts
   to source space (keeping the complex values).
    - `ssvep_plot_stcs.py` plots each subject at 2, 4, 6, 12 Hz, both
      "magnitude" and "SNR" versions.

3. `ssvep_group_level_aggregate_stcs.py` aggregates individual subject STCs
   into group-level averages, in "magnitude", "SNR", and "log(SNR)" versions.
    - `ssvep_group_level_plot_stcs.py` plots group-average STCs at 2, 4, 6,
      and 12 Hz.

4. `ssvep_stats.py` computes uncorrected t-value maps and optionally runs
   clustering / TFCE.
    - `ssvep_plot_tvals.py` plots the t-value maps
    - `ssvep_plot_clusters.py` plots the clusters


## Filename conventions

Source-space files and movies follow the convention:

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
  `LanguageInterventionN24FSAverage`)
- `timepoint` takes values `preCamp`, `postCamp`, or `PostCampMinusPreCamp`,
  and indicates whether the data were acquired before or after the
  intervention.
- `method` refers to the source localization method (e.g., `dSPM`, `sLORETA`,
  etc)
- `condition` refers to the experimental conditions, e.g., `faces`, `words`,
  `cars`, and `aliens`. Files showing differences between conditions use the
  convention `ConditionOneMinusConditionTwo`, e.g., `FacesMinusCars`.
