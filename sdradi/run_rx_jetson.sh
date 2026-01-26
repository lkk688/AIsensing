#!/bin/bash
# run_rx_jetson.sh - Run Receiver on Jetson
# Assumes 'py310' venv is already active or available

echo "=== Environment Check ==="
# Check/Install missing dependencies
pip install scipy matplotlib pyadi-iio pylibiio "numpy<2.0" || true

echo "=== Starting Receiver (Pluto @ Local) ==="
# Use localhost or specific IP depending on Jetson networking
# If Docker: 192.168.2.1
# If Native: 192.168.2.2
TARGET_IP="ip:192.168.2.2"

# Check if 192.168.2.2 is visible, else try 127.0.0.1
if ping -c 1 192.168.2.2 &> /dev/null; then
    TARGET_IP="ip:192.168.2.2"
else
    echo "Pluto IP (192.168.2.1) not reachable. Trying 127.0.0.1..."
    TARGET_IP="ip:127.0.0.1"
fi

python sdr_video_comm.py \
    --mode rx \
    --device pluto \
    --ip $TARGET_IP \
    --num_bits 50000
