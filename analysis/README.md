To run the analysis scripts/notebooks, it is necessary to "install" this
repository as a package using `pip` to make shared helper functions
available to all the scripts. The shared helper functions live in
`../sswef_helpers/aux_functions.py`.

**Before running any of these scripts, execute:** `pip install .` from the
root directory of the repo.

## Filename conventions

In general, source-space files and movies follow the convention:

```python
f'{subject}_{timepoint}_{method}_{condition}.fileExtension'
```

Here are the general guidelines (these may deviate somewhat between the SSVEP
and the ERP scripts, but this should give an idea of what to expect):

- for group averages in source space, activity is morphed to a template brain
  and hence `subject` will always include `FSAverage` (referring to the
  FreeSurfer average template brain). File names for group averages may also
  include the number of participants averaged, so an average based on all 48
  participants will be called `GrandAvgN48FSAverage`. Averages of more
  restricted subsets of the participant population will not say "GrandAvg" but
  instead use some descriptor of the group selection criteria (e.g.,
  `LanguageInterventionN24FSAverage`).
- `timepoint` takes values `preCamp`, `postCamp`, or `PostCampMinusPreCamp`,
  and indicates whether the data were acquired before or after the
  intervention.
- `method` refers to the source localization method (e.g., `dSPM`, `sLORETA`,
  etc)
- `condition` refers to the experimental conditions, e.g., `faces`, `words`,
  `cars`, and `aliens` in the "ERP" experiment. Files showing differences
  between conditions use the convention `ConditionOneMinusConditionTwo`, e.g.,
  `FacesMinusCars`.

Other analysis parameters (e.g., what frequency bin is being displayed) are
often added to the end of this template, especially in the names of figures.
