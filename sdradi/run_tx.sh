#!/bin/bash
# run_tx.sh - Run Transmitter on Main Host (RTX5090)
# Fixes GLIBCXX error by preloading system libstdc++

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

echo "Starting Transmitter (Pluto @ 192.168.3.2)..."
python sdr_video_comm.py \
    --mode tx \
    --device pluto \
    --ip ip:192.168.3.2 \
    --fc 915e6 \
    --num_bits 50000
