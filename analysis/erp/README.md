# Analysis pipeline for evoked responses

1. Annotate the raw recordings to exclude noisy spans on the EOG channel (so
   that SSP projectors will actually reflect blink events, not noise). This is
   done with `prek_annotate_interactive.sh`, which is a wrapper for
   `prek_annotate_interactive.py`. Saves the annotations alongside the raw
   recordings, in `*-custom-annot.fif` files

2. Run semi-supervised preprocessing. This requires:
    - The `-custom-annot.fif` files (see above)
    - `prek_score.py` (scoring function, called by `prek_mnefun.py`)
    - `prek_setup_source.py` (makes hi-res brain model for each subject,
      creates a source space, plots the source space for quality check)
    - see `../preprocessing/README.md` for more info.

3. Preliminary aggregation:
    - `prek_make_stcs.py` (makes STC for each subj, morphs it to fsaverage,
      makes subj avg for each experimental condition). Results are saved in
      `/mnt/scratch/prek/<PRE_OR_POST_CAMP>/twa_hp/erp/<SUBJ>/stc`.
    - `prek_make_group_averages.py` (makes average STC for each experimental
      condition, across all subjects, separately for pre- and post-intervention
      recordings). Also makes group averages for each intervention cohort and
      for the top/bottom half on the letter awareness pre-tests.
      Writes group-average STCs to `/mnt/scratch/prek/results/<COHORT>/group_averages`

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
    - STCs are saved to `/mnt/scratch/prek/results/<COHORT>/group_averages`
    - Movies are made of every STC, and are saved to 
      `/mnt/scratch/prek/results/<COHORT>/movies`

5. ROI analysis:
    - `../ROIs/create_ventral_band_labels.py` will generate the ROI labels
    - `prek_extract_ROI_time_courses.py` will extract average time course in
      each ROI label for each subject in each condition.

6. Clustering:
    - `prek_clustering.py` runs the clustering
    - `prek_plot_clusters_quick_and_dirty.py` generates cortical images of the 
      clusters, alongside plots of average time series extracted from the
      cluster-defined regions.
