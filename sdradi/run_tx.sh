#!/bin/bash
# run_tx.sh - Run Transmitter on Main Host (RTX5090)
# Fixes GLIBCXX error by preloading system libstdc++

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "Starting Transmitter (AntSDR @ 192.168.1.10)..."
python sdr_video_comm.py \
    --mode tx \
    --device antsdr \
    --ip ip:192.168.1.10 \
    --num_bits 50000
