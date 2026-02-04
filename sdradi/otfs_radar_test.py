import numpy as np
import matplotlib.pyplot as plt
import adi
import time
import sys
import os

# Add path to access sdr_video_comm (in same directory)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sdr_video_comm import OTFSConfig, OTFSTransceiver

def otfs_radar_test():
    print("=== OTFS Radar Test on CN0566/AD9361 ===")
    
    # 1. Configuration
    SDR_IP = "ip:192.168.86.40" # Default from app
    FC = 10.25e9 # CN0566 Center Freq (Fixed LO)
    FS = 30.72e6 # 30.72 MSPS (Standard LTE rate, stable on Pluto)
    # BW = 40e6 # 40 MHz
    
    print(f"Connecting to SDR at {SDR_IP}...")
    try:
        sdr = adi.Pluto(uri=SDR_IP)
    except Exception as e:
        print(f"Error connecting: {e}")
        return

    # Configure SDR
    sdr.sample_rate = int(FS)
    sdr.tx_lo = int(FC)
    sdr.rx_lo = int(FC)
    sdr.tx_rf_bandwidth = int(FS) # Match FS roughly
    sdr.rx_rf_bandwidth = int(FS)
    sdr.tx_hardwaregain_chan0 = -10 # Conservative start
    sdr.rx_hardwaregain_chan0 = 40
    sdr.rx_buffer_size = 1024*1024 # Large buffer for capture
    
    # 2. Generate OTFS Waveform
    print("Generating OTFS Waveform...")
    cfg = OTFSConfig()
    cfg.mod_order = 4 # QPSK
    cfg.N_delay = 256 # 256 delay bins (Time)
    cfg.N_doppler = 64 # 64 doppler bins (Freq)
    # cp_length = 0 ideally for OTFS Radar to maximize valid grid
    
    tr = OTFSTransceiver(cfg)
    
    # Generate Known Symbols (Full Grid) -> Sounding Signal
    # Random QPSK symbols
    num_syms = cfg.N_delay * cfg.N_doppler
    tx_bits = np.random.randint(0, 2, num_syms * 2) # 2 bits per symbol (QPSK)
    
    # Get Time Domain Signal
    tx_signal = tr.modulate(tx_bits)
    
    # Add Zadoff-Chu Sync Preamble
    # ZC Len 127
    n = np.arange(127)
    root = 25
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / 127)
    
    # Pad: [Silence(500), ZC, Silence(100), OTFS_Payload, Silence(1000)]
    preamble = np.concatenate([np.zeros(500), zc, np.zeros(100)])
    full_tx = np.concatenate([preamble, tx_signal, np.zeros(1000)])
    
    # Normalize
    full_tx = full_tx / np.max(np.abs(full_tx)) * 0.5 # -6 dBFS
    full_tx = full_tx.astype(np.complex64)
    
    # 3. Burst Transmission
    print(f"Transmitting Burst ({len(full_tx)} samples)...")
    sdr.tx_cyclic_buffer = True
    sdr.tx(full_tx) # Start continuous TX of this burst
    
    # 4. Receive
    print("Receiving...")
    # Discard first few buffers to settle
    for _ in range(3):
        _ = sdr.rx()
        
    rx_raw = sdr.rx()
    sdr.tx_destroy_buffer() # Stop TX
    
    # 5. Processing
    print("Processing...")
    
    # A. Correlate for Sync
    corr = np.correlate(rx_raw, zc, mode='valid')
    peak = np.argmax(np.abs(corr))
    print(f"Sync Peak at {peak}")
    
    # B. Extract Payload
    # Preamble ends at peak + 127 (roughly)
    # Payload starts 100 samples after ZC
    payload_start = peak + 100
    if payload_start + len(tx_signal) > len(rx_raw):
        print("Error: Buffer too short to capture full payload.")
        return
        
    rx_payload = rx_raw[payload_start : payload_start + len(tx_signal)]
    
    # C. OTFS Radar Processing (TF Division)
    # 1. Reshape to Time Grid [Ns, Nc] (Rows=Delay, Cols=Doppler/Time)
    # Note: OTFSTransceiver structure: 
    #   rx_time_grid = rx_payload.reshape(Nc, Ns).T  => [Ns, Nc]
    #   Wait, need to check `demodulate` reshape.
    #   In `sdr_video_comm.modulate`:
    #       x_tf = x_dd_grid (iffting etc)
    #       x_time = ifft(x_tf, axis=0) -> columns are time-blocks? No.
    #       Heisenberg transform.
    #   Let's rely on dimensions N_delay (M) and N_doppler (N).
    #   Usually Time-Frequency grid is M x N.
    #   Time payload is simple vectorization of columns? or CP-OFDM style?
    #   `sdr_video_comm` implementation roughly:
    #   Modulate:
    #       x_tf = isfft(x_dd) (Inverse Symplectic)
    #       x_time = ifft(x_tf, axis=0) (OFDM modulator)
    #       So time signal is serialization of columns of x_time.
    
    Ns = cfg.N_delay # M (Subcarriers)
    Nc = cfg.N_doppler # N (Time slots)
    
    # Reshape RX
    rx_grid_time = rx_payload.reshape(Nc, Ns).T # [Ns, Nc]
    
    # FFT to TF Domain
    rx_grid_tf = np.fft.fft(rx_grid_time, axis=0)
    
    # Reconstruct TX TF Grid
    # We need the TX symbols in TF domain.
    # We can get them by "demodulating" the *tx_signal* (clean loopback) without noise, 
    # OR simpler: Re-run the modulator steps partly.
    # Let's use `demodulate` on the clean TX signal to get the internal grids? 
    # No, `demodulate` destroys phase info.
    # Let's do the Forward Transform manually on the `tx_signal` we generated.
    tx_grid_time = tx_signal.reshape(Nc, Ns).T
    tx_grid_tf = np.fft.fft(tx_grid_time, axis=0)
    
    # Channel Estimation (Element-wise Division in TF)
    # H_tf = Y_tf / X_tf
    # Avoid div by zero
    H_tf = rx_grid_tf / (tx_grid_tf + 1e-9)
    
    # Transform to Delay-Doppler (ISFFT)
    # Implementation in sdr_video_comm:
    #   rx_dd_grid = fft(rx_tf_grid, axis=1) # Row FFT
    #   rx_dd_grid = ifft(rx_dd_grid, axis=0) # Col IFFT
    #   This is the Inverse Symplectic.
    
    # We want H_dd.
    H_dd = np.fft.fft(H_tf, axis=1)
    H_dd = np.fft.ifft(H_dd, axis=0)
    
    # Shift to center Doppler
    H_dd = np.fft.fftshift(H_dd, axes=1)
    
    # 6. Plot
    print("Plotting...")
    RDM = 20 * np.log10(np.abs(H_dd) + 1e-12)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(RDM, aspect='auto', cmap='viridis', origin='lower',
               extent=[-FS/2, FS/2, 0, Ns])
    plt.colorbar(label='Amplitude (dB)')
    plt.title("OTFS Radar Response (Delay-Doppler)")
    plt.xlabel("Doppler (Hz) [Unscaled]")
    plt.ylabel("Delay Bins")
    plt.savefig ("otfs_radar_result.png")
    print("Saved otfs_radar_result.png")

if __name__ == "__main__":
    otfs_radar_test()
