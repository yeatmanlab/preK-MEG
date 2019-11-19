#!/bin/bash
data_root="/mnt/scratch/prek"
cluster_root="$data_root/results/clustering"
read -r cluster_dir<"$cluster_root/most-recent-clustering.txt"
frames_dir="$cluster_dir/frames"
movies_dir="$data_root/results/movies/clustering"
for dir in $(ls $frames_dir); do
    ffmpeg -framerate 5 \
        -i ${dir%frames}%03d.png \
        -c:v libx264 \
        -pix_fmt yuv420p \
        "$movies_dir/${dir%_frames}.mp4"
done
