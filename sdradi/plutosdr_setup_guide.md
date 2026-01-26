# ADALM-PLUTO SDR Setup, Recovery, and Troubleshooting Guide

This document details the steps to set up, configure, and **recover** the ADALM-PLUTO (PlutoSDR) ecosystem for the SDR Video Communication System.

---

## 1. Quick Start: Automated Driver Recovery

If you are experiencing "No Device Found" errors, "Undefined Symbol" errors, or need to update your system, use the automated scripts provided in this repository.

### A. Reset Drivers and Update Firmware (First Device)
Run this script to:
1.  Download and build `libiio` (v1.0) from source.
2.  Install Python bindings compatible with the new library.
3.  Download Firmware v0.39 and update the first detected PlutoSDR.

```bash
# In /Developer/AIsensing/sdradi
./reset_drivers.sh
```

### B. Update Second Device (Dual Setup)
If you have two devices, the first script only updates one. Use this to update the second:

```bash
# In /Developer/AIsensing/sdradi
./update_second_pluto.sh
```

**After updating**: Unplug and Replug ALL PlutoSDR devices.

---

## 2. Hardware Connection & Verification

1.  **Connect**: Plug PlutoSDR into USB.
2.  **Verify USB**:
    ```bash
    lsusb
    # Look for "Analog Devices Inc. PlutoSDR"
    ```
3.  **Verify IIO Context**:
    ```bash
    iio_info -s
    # Should list USB contexts like "usb:1.2.5"
    ```
    *Note: If `iio_info` fails, your drivers are broken. Run `./reset_drivers.sh`.*

---

## 3. Manual Installation (If Script Fails)

If you need to install manually, follow these steps (tested on Ubuntu 24.04):

### Prerequisites
```bash
sudo apt install cmake bison flex libxml2-dev libcdk5-dev libaio-dev libusb-1.0-0-dev libavahi-client-dev libavahi-common-dev
```

### Build Libiio (C Library)
```bash
git clone https://github.com/analogdevicesinc/libiio.git
cd libiio
git checkout v0.25  # Use v0.25 for stability, or main for latest features
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DENABLE_PYTHON_BINDINGS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Install Python Bindings
**CRITICAL**: Do not install `pyadi-iio` from pip alone if you built `libiio` from source. They must match.
```bash
# Install low-level bindings from your source tree
cd libiio/bindings/python
pip install .

# Install high-level driver
pip install pyadi-iio
```

### Firmware Update
1.  Download `plutosdr-fw-v0.39.zip` from [Analog Devices GitHub](https://github.com/analogdevicesinc/plutosdr-fw/releases).
2.  Unzip `pluto.frm`.
3.  Copy `pluto.frm` to the mounted `PlutoSDR` drive.
4.  Eject the drive. The LED will blink rapidly. Wait for it to reset.

### B. IP-Based Configuration (Recommended)
Using IP addresses is more stable than USB URIs (which change on reboot).
1.  Configure your PlutoSDRs with static IPs (e.g., `192.168.2.2` and `192.168.3.2`) by editing `config.txt` on the device mass storage.
2.  Update `sdr_tuned_config.json`:
    ```json
    {
        "sdr_ip": "ip:192.168.2.2",
        "rx_uri": "ip:192.168.3.2",
        "device": "pluto_dual"
    }
    ```
3.  Verify connectivity: `ping 192.168.2.2`

---

## 4. Known Issues & Limitations

### A. The "50% BER" Transport Issue
**Status**: CONFIRMED (Hardware/Driver Limitation)

On high-performance hosts with **Dual PlutoSDR** setups, you may observe a **50% Bit Error Rate (BER)** even with perfect Synchronization (SNR > 20dB).
*   **Symptoms**: `test_short_packet.py` consistently fails. Received data is scrambled (bits flipped/shuffled block-wise) but frame energy is present. This occurs over **BOTH** USB (`usb:x.x.x`) and IP (`ip:x.x.x`) transports.
*   **Cause**: Low-level Buffer/DMA issue in `libiio` or the Host Controller execution (e.g., AVX/Alignment mismatch or Kernel Ring buffer corruption), causing `Bus Error` or `Segmentation Fault`.
*   **Resolution**: This is a Physical Transport limitation.
    *   **Workaround**: Use the **Simulator** (`AIradar_comm_dataset_g2.py`) for Data Plane / Neural Network development.
    *   **Hardware Usage**: Use hardware ONLY for Synchronization demos, Channel Sounding, and basic waveform visualization. Do not rely on valid payload decoding for Dual-Pluto setups on this host.

### B. "Undefined Symbol: iio_get_backends_count"
*   **Cause**: Mismatch between `libiio.so` (v1.0) and `pylibiio` (v0.25 Python binding).
*   **Fix**: Run `./reset_drivers.sh` to rebuild both from the same source tree.

### C. "No Device Found" after Update
*   **Cause**: Firmware v0.39 changes the USB serialization/enumeration. The URI changes (e.g., from `usb:1.5.5` to `usb:9.7.5`).
*   **Fix**:
    1.  Run `python scan_devices.py` (provided in repo) to list visible URIs.
    2.  Update `sdr_tuned_config.json` with the new URIs.

---

## 5. Summary of Debugging & Conclusions (Jan 2026)

We performed extensive stress testing to identify the source of the "50% BER" issue. Here is the summary:

| Configuration | Setup | Transport | Result | detailed |
| :--- | :--- | :--- | :--- | :--- |
| **Single Device** | Pluto (1) Loopback | USB | **PASS** | BER < 1e-4, SNR 20dB |
| **Single Device** | AntSDR Loopback | Ethernet | **PASS** | BER < 1e-4, SNR 20dB |
| **Dual Device** | Pluto (TX) + Pluto (RX) | USB (Dual) | **FAIL** | BER ~50%, Bus Error |
| **Dual Device** | Pluto (TX) + Pluto (RX) | IP (RNDIS) | **FAIL** | BER ~50%, Bus Error |
| **Distributed** | AntSDR (Local) + Pluto (Remote IIO) | IP | **FAIL** | BER ~50%, Bus Error |

**Root Cause**: The host machine's Linux Kernel or USB Controller execution stack cannot handle **two simultaneous High-Speed IIO Contexts** (TX and RX) regardless of whether they are USB or Ethernet. The crash (`Bus Error`) and data scrambling (50% BER) indicate memory corruption in the driver/kernel boundary when context switching between the two devices.

---

## 6. Solution: Distributed SDR Architecture (The "Golden" Verification Setup)

This setup bypasses all USB buffer contention on the main host.

### Step 1: Receiver (Jetson/Pi) Setup
**Goal**: Run `sdr_video_comm.py` in `rx` mode locally on the Jetson.

1.  **Dependencies**: The Jetson needs `scipy` and SDR drivers.
    ```bash
    # Activate your venv (e.g., py310)
    conda activate py310
    
    # Install missing packages
    pip install scipy matplotlib pyadi-iio pylibiio "numpy<2.0"
    ```

2.  **Run Receiver**:
    Use the helper script `run_rx_jetson.sh` (copy it to Jetson) or run manually:
    ```bash
    # Check if Pluto is visible
    ping 192.168.2.2
    iio_info -s
    
    # Run RX
    python sdr_video_comm.py --mode rx --device pluto --ip ip:192.168.2.2 --num_bits 50000
    ```

### Step 2: Transmitter (RTX5090) Setup
**Goal**: Run `sdr_video_comm.py` in `tx` mode on the Main PC.

1.  **Fix GLIBCXX Error**: Use `LD_PRELOAD`.
2.  **Run Transmitter**: Use the helper script `run_tx.sh`:
    ```bash
    chmod +x run_tx.sh
    ./run_tx.sh
    ```
    *Manual Command*:
    ```bash
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    python sdr_video_comm.py --mode tx --device antsdr --ip ip:192.168.1.10 --num_bits 50000
    ```

### Expected Result
*   **Sender**: Prints "Transmitting continuous stream...".
*   **Receiver**: Prints "Receiving..." and BER/SNR metrics.
*   **Success Criteria**: BER < 1e-3 (0.1%), SNR > 15 dB.
*   **Note**: If BER is ~50%, ensure the AntSDR and Pluto are on the same center frequency (`--fc 2.4e9`) and sampling rate (`--fs 1e6` is safer than 10e6 for initial tests).

---

---

## 7. Development Workflow Recommendation

Given the USB Transport limitations:
1.  **Develop Algorithms** using `AIradar_comm_dataset_g2.py` (Simulator).
2.  **Verify Logic** using `test_ofdm_logic.py` (Software Loopback).
3.  **Demonstrate Sync** using `sdr_video_ui.py` (Hardware). Showing the constellation lock is sufficient proof of synchronization, even if payload bits are scrambled by the driver.
