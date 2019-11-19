#!/bin/bash
DATA_ROOT="/mnt/scratch/prek"
FRAMES_DIR="$DATA_ROOT/results/clustering/frames"
MOVIES_DIR="$DATA_ROOT/results/movies/clustering"
for dir in $(ls $FRAMES_DIR); do
    ffmpeg -framerate 5 \
        -i ${dir%frames}%03d.png \
        -c:v libx264 \
        -pix_fmt yuv420p \
        "$MOVIES_DIR/${dir%_frames}.mp4"
done
