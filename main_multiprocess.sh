#!/bin/bash

num_runs=16
bias_length=4

parallel_runs=$num_runs/$bias_length

start_time=$(date +"%Y-%m-%d %H:%M:%S.%3N")

for (( i=0; i<parallel_runs; i++ ))
do
    bias=$i
    echo "Running process with bias: $bias"

    ./set_seeds.sh "$bias" "$bias_length" &
done

wait
echo "All tasks completed."

current_time=$(date +"%Y-%m-%d %H:%M:%S.%3N")
echo "Start time: $start_time"
echo "Current time: $current_time"

