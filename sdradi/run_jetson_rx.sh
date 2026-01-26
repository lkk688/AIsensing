#!/bin/bash
# run_jetson_rx.sh
# Runs the SDR Receiver inside the 'jetson-dev' container where PyTorch is installed.

CONTAINER="jetson-dev"
WORK_DIR="/Developer/AIsensing/sdradi"
PLUTO_IP="ip:192.168.2.1"  # Default IP of Pluto over USB

echo "=== Setup: Checking Container Environment ==="
# Ensure dependencies are installed inside the container
# We use '|| true' to assume success if already installed or to ignore trivial apt errors
echo "Installing libiio and pip packages..."
docker exec $CONTAINER bash -c "apt-get update && apt-get install -y libiio-utils libiio-dev || true"
docker exec $CONTAINER bash -c "pip install pyadi-iio 'numpy<2.0' scipy || true"

echo "=== Running RX Node ==="
echo "Mode: RX"
echo "Device URI: $PLUTO_IP"
echo "Container: $CONTAINER"

# Run the python script
docker exec -it -w $WORK_DIR $CONTAINER python3 sdr_video_comm.py \
    --mode rx \
    --device pluto \
    --ip $PLUTO_IP \
    --num_bits 10000

# Note: If 192.168.2.1 is not reachable, ensure the container was started with --net=host
# or map the USB device directly.
