The main use of files in this folder is for defining the upper/lower grouping
based on pre-intervention letter awareness test scores. A secondary use is to
generate a couple exploratory plots of the behavioral data.

- `BehavioralCutpoint.m` is an old file that is no longer used. It reads in a
  raw report from RedCap (not under version control, for privacy reasons) and
  writes out a much-reduced CSV file `LetterKnowledge.csv`
- `preK_InterventionData.csv` is the main data file
- `extract-cohorts.py` is the script that determines the upper/lower groupings.
  It uses either a gaussian mixture model or a quantile-based cutoff, which
  yield similar (but not quite identical) groupings. The resulting groupings
  are written to YAML files in the `../../params` directory.
