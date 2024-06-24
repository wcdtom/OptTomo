#!/bin/bash

num_runs=64
bias_length=16

parallel_runs=$num_runs/$bias_length

for (( i=0; i<parallel_runs; i++ ))
do
    bias=$i
    echo "Running process with bias: $bias"

    ./set_seeds.sh "$bias" "$bias_length" &
done

wait
echo "All tasks completed."
