#!/bin/bash

# loop over raw files
# run separately for /mnt/scratch/prek/pre_camp/twa_hp/*/*/raw_fif/*_pre_raw.fif
#                and /mnt/scratch/prek/post_camp/twa_hp/*/*/raw_fif/*_post_raw.fif
for raw in $(ls /data/prek/pre_camp/twa_hp/*/*/raw_fif/*_pre_raw.fif); do
    # set name of annotation file
    ann=${raw%.fif}"-custom-annot.fif"
    # get the subject ID
    bname=$(basename $raw .fif)
    subj=${bname%_pre_raw}
    echo "processing subject $subj..."
    # what to do if already exists?
    # note: last 2 args of python script are booleans: rerun & save
    if [ -f $ann ]; then
        echo "Annotation file for $subj already exists. Your choices:"
        select reann in \
        "Reannotate that file & overwrite when done" \
        "Open that file for viewing; don't save when done" \
        "Annotate original RAW & overwrite existing annotated file" \
        "Skip this subject"; do
            case $REPLY in
            1) python prek_annotate_interactive.py $raw $ann 1 1; break ;;
            2) python prek_annotate_interactive.py $raw $ann 1 0; break ;;
            3) python prek_annotate_interactive.py $raw $ann 0 1; break ;;
            4) echo "skipping $subj"; break ;;
            *) echo "Invalid choice. Enter 1-4 (or Ctrl+C to abort)." ;;
            esac
        done </dev/tty
    else
        python prek_annotate_interactive.py $raw $ann 0 1
    fi
done
