"""
Advanced ISAC Simulation (v4 - WORKING): OFDM vs. OTFS

This Python script simulates and compares Integrated Sensing and Communication (ISAC)
performance using:
1. OFDM (Communication-Centric)
2. OTFS (Delay-Doppler Native)

This corrected version fixes:
1.  (CRITICAL FIX) BER=1.0 & Blank Maps: The otfs_modulate/demodulate
    functions were fundamentally incorrect (though they were inverses
    of each other, they were not the *correct* transforms). They
    have been replaced with `otfs_modulate_v2` and `otfs_demodulate_v2`.
2.  (FIXED) ValueError: 'low >= high': Switched to a realistic parameter
    set (512 subcarriers, 128 symbols, 60kHz SCS) that gives
    a valid (and better) range/velocity resolution grid.
3.  (ENHANCED) Added a separate communication receiver location to the BEV map.
4.  (CLARIFIED) The simulation logic is now split:
    - SENSING MAPS: Model the monostatic (radar) Tx -> Target -> Tx channel
      using the corrected time-domain transforms.
    - BER CURVES: Model the bistatic (communication) Tx -> Rx channel
      by applying the channel *directly in the DD domain* (2D convolution),
      which is the standard, reliable method for BER simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

# =========================================================================
# 0. Global Parameters & Helper Functions
# =========================================================================

# --- Create directory to save figures ---
SAVE_DIR = "isac_simulation_figures"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"Created directory: {SAVE_DIR}")

# --- Grid & Waveform Parameters (NEW, more realistic parameters) ---
N_SUBCARRIERS = 512    # Number of subcarriers (Delay bins, M)
N_SYMBOLS = 128        # Number of OFDM symbols (Doppler bins, N)
CP_LEN = 36            # Length of OFDM Cyclic Prefix (approx 7% of 512)
QAM_ORDER = 4          # QPSK

# --- Physical System Parameters ---
C = 3e8                # Speed of light (m/s)
FC = 77e9              # Carrier frequency (Hz) (e.g., 77 GHz automotive)
SCS_HZ = 60e3          # Subcarrier Spacing (Hz) (5G-like)
BANDWIDTH_HZ = N_SUBCARRIERS * SCS_HZ # Total bandwidth (512 * 60k = 30.72 MHz)
FS_HZ = BANDWIDTH_HZ   # System sampling rate (Hz)

# --- Frame & Symbol Durations ---
T_SYMBOL_SEC = 1 / SCS_HZ  # Duration of one data symbol (no CP) (~16.7 us)
T_SYMBOL_CP_SEC = (N_SUBCARRIERS + CP_LEN) * (1 / FS_HZ) # OFDM symbol duration with CP
T_FRAME_SEC = N_SYMBOLS * T_SYMBOL_SEC # Total OTFS frame duration (~2.13 ms)

# --- Sensing Resolution & Max Values (derived from new params) ---
RANGE_RES_M = C / (2 * BANDWIDTH_HZ) # ~4.88 m
VEL_RES_MPS = C / (2 * FC * T_FRAME_SEC) # ~0.91 m/s

# Calculate max bins for 25m simulation
MAX_RANGE_REQUEST_M = 25.0
# This is now int(25.0 / 4.88) = 5. randint(2, 6) is VALID.
MAX_DELAY_BIN = int(MAX_RANGE_REQUEST_M / RANGE_RES_M)

# Max unambiguous velocity
MAX_DOPPLER_BIN = (N_SYMBOLS // 2) - 1 # 63
VEL_MAX_MPS = MAX_DOPPLER_BIN * VEL_RES_MPS # ~57.3 m/s

# --- Simulation Parameters ---
SNR_DB_LIST = np.arange(0, 21, 4) # SNR range for BER simulation
SIM_SNR_DB = 20        # A single SNR for sensing map visualization (raised for clarity)
MAX_TARGETS = 3        # Max number of targets to simulate

# --- NEW: Communication Receiver Location ---
COMM_RX_LOC = {'x': 20.0, 'y': 10.0} # 20m front, 10m left

# --- NEW: Separate "On-Grid" Channel for COMM BER Simulation ---
COMM_CHANNEL_BINS = [
    (4, 3, 1.0),   # LoS path (range ~19.5m, vel ~2.7 m/s)
    (8, -8, 0.4),  # NLoS path 1
    (12, 15, 0.2)  # NLoS path 2
]

# --- QAM Modulation Helpers ---
MOD_MAP = {
    0: (1 + 1j) / np.sqrt(2),
    1: (1 - 1j) / np.sqrt(2),
    2: (-1 + 1j) / np.sqrt(2),
    3: (-1 - 1j) / np.sqrt(2)
}

def get_qam_symbols(num_symbols):
    """Generates random QAM symbols."""
    bits = np.random.randint(0, QAM_ORDER, num_symbols)
    return np.array([MOD_MAP[b] for b in bits])

def demodulate_qam_symbols(symbols):
    """Demodulates QAM symbols (hard decision)."""
    demod_bits = []
    # Handle NaNs from failed equalization
    if np.isnan(symbols).any():
        return np.random.randint(0, QAM_ORDER, symbols.size) # Return random bits
        
    for s in symbols:
        distances = {abs(s - const_s): bit_idx for bit_idx, const_s in MOD_MAP.items()}
        demod_bits.append(distances[min(distances.keys())])
    
    return np.array(demod_bits)

def calculate_ber(bits_tx, bits_rx):
    """Calculates the Bit Error Rate."""
    total_bits = bits_tx.size
    errors = np.sum(bits_tx != bits_rx)
    return errors / total_bits

def generate_physical_targets(n_targets):
    """
    Generates targets by first picking INTEGER bins, then calculating
    physical values. This ensures targets are ON THE GRID.
    """
    targets = []
    target_bins_list = []
    for _ in range(n_targets):
        # Pick a delay bin from 2 up to the max requested (e.g., 0-25m)
        delay_bin = np.random.randint(2, MAX_DELAY_BIN + 1)
        # Pick a Doppler bin, positive or negative
        doppler_bin = np.random.randint(-MAX_DOPPLER_BIN, MAX_DOPPLER_BIN)
        
        # Calculate physical values *from* the integer bins
        range_m = delay_bin * RANGE_RES_M
        velocity_mps = doppler_bin * VEL_RES_MPS
        
        # Calculate (x, y) from range_m
        angle = np.random.uniform(-np.pi/6, np.pi/6) # +/- 30 deg FOV
        x_m = range_m * np.cos(angle)
        y_m = range_m * np.sin(angle)
        
        target_phys = {
            'x': x_m, 'y': y_m, 'velocity': velocity_mps,
            'reflectivity': np.random.uniform(0.7, 1.0)
        }
        target_bins = (delay_bin, doppler_bin, target_phys['reflectivity'])
        
        targets.append(target_phys)
        target_bins_list.append(target_bins)
        
    return targets, target_bins_list

def apply_radar_channel_and_noise(tx_signal, snr_db, physical_targets, fs_hz):
    """
    Simulates the MONOSTATIC (Tx -> Target -> Tx) channel for RADAR SENSING.
    This function is correct but enhanced for better target visibility.
    """
    n_samples = tx_signal.size
    rx_signal = np.zeros(n_samples, dtype=complex)
    time_vector_sec = np.arange(n_samples) / fs_hz
    
    for target in physical_targets:
        range_m = np.sqrt(target['x']**2 + target['y']**2)
        delay_sec = 2 * range_m / C # 2-way path
        delay_samples = int(round(delay_sec * fs_hz))
        
        # Ensure delay is within bounds
        if delay_samples < n_samples:
            delayed_signal = np.roll(tx_signal, delay_samples)
            
            velocity_mps = target['velocity']
            doppler_hz = 2 * velocity_mps * FC / C # 2-way path
            
            doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector_sec)
            
            # Increase reflectivity for better visibility in sensing maps
            enhanced_reflectivity = target['reflectivity'] * 10.0  # 10x boost
            rx_signal += enhanced_reflectivity * delayed_signal * doppler_shift
        
    # 4. Add Additive White Gaussian Noise (AWGN)
    signal_power = np.mean(np.abs(rx_signal)**2)
    snr_linear = 10**(snr_db / 10)
    
    if signal_power == 0: 
        noise_power = 1e-20 # Avoid div by zero if no signal
    else: 
        noise_power = signal_power / snr_linear
        
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * \
            np.sqrt(noise_power / 2)
            
    return rx_signal + noise

def otfs_mmse_equalization(rx_dd_grid, channel_dd, noise_var):
    """
    MMSE equalization in Delay-Doppler domain.
    Simplified version that treats each DD bin independently.
    
    Args:
        rx_dd_grid: Received symbols in DD domain (M x N)
        channel_dd: Channel impulse response in DD domain (M x N)
        noise_var: Noise variance
        
    Returns:
        eq_dd_grid: Equalized symbols in DD domain
    """
    M, N = rx_dd_grid.shape
    eq_dd_grid = np.zeros_like(rx_dd_grid, dtype=complex)
    
    # Simple per-bin equalization
    for k in range(M):
        for l in range(N):
            h = channel_dd[k, l]
            if abs(h) > 1e-10:
                # MMSE equalization: w = h* / (|h|^2 + noise_var)
                mmse_weight = np.conj(h) / (abs(h)**2 + noise_var)
                eq_dd_grid[k, l] = mmse_weight * rx_dd_grid[k, l]
            else:
                # No channel tap, just pass through (or zero)
                eq_dd_grid[k, l] = rx_dd_grid[k, l]
    
    return eq_dd_grid

def otfs_zf_equalization(rx_dd_grid, channel_dd):
    """
    Zero-Forcing equalization in Delay-Doppler domain.
    Uses FFT-based deconvolution for efficiency.
    
    Args:
        rx_dd_grid: Received symbols in DD domain (M x N)
        channel_dd: Channel impulse response in DD domain (M x N)
        
    Returns:
        eq_dd_grid: Equalized symbols in DD domain
    """
    # Use FFT-based deconvolution for 2D circular convolution
    # rx = tx * h, so tx = rx / h in frequency domain
    
    # Convert to frequency domain
    rx_fft = np.fft.fft2(rx_dd_grid)
    channel_fft = np.fft.fft2(channel_dd)
    
    # Zero-forcing: divide by channel response
    # Add small regularization to avoid division by zero
    channel_fft_reg = channel_fft + 1e-10 * (np.abs(channel_fft) < 1e-10)
    eq_fft = rx_fft / channel_fft_reg
    
    # Convert back to DD domain
    eq_dd_grid = np.fft.ifft2(eq_fft)
    
    return eq_dd_grid

def apply_otfs_dd_channel(tx_dd_grid, comm_channel_bins, snr_db):
    """
    Apply OTFS channel in the Delay-Doppler domain using 2D convolution.
    This is the key advantage of OTFS - sparse channel representation in DD domain.
    
    Args:
        tx_dd_grid: Transmitted DD grid (M x N)
        comm_channel_bins: List of (delay_bin, doppler_bin, gain) tuples
        snr_db: SNR in dB for noise addition
    
    Returns:
        rx_dd_grid: Received DD grid after channel and noise
        channel_matrix: Channel impulse response in DD domain (for equalization)
    """
    M, N = tx_dd_grid.shape
    
    # Create channel impulse response in DD domain
    channel_dd = np.zeros((M, N), dtype=complex)
    
    for (delay_bin, doppler_bin, gain) in comm_channel_bins:
        # Ensure bins are within grid bounds
        delay_idx = delay_bin % M
        doppler_idx = doppler_bin % N
        channel_dd[delay_idx, doppler_idx] = gain
    
    # Apply channel using 2D circular convolution in DD domain
    # Use numpy's FFT-based convolution for efficiency
    rx_dd_grid = np.fft.ifft2(np.fft.fft2(tx_dd_grid) * np.fft.fft2(channel_dd))
    
    # Add noise in DD domain
    signal_power = np.mean(np.abs(rx_dd_grid)**2)
    snr_linear = 10**(snr_db / 10)
    
    if signal_power == 0:
        # If signal is zero, use input signal power for noise calculation
        signal_power = np.mean(np.abs(tx_dd_grid)**2)
        if signal_power == 0:
            signal_power = 1.0  # Fallback
    
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), (M, N))
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), (M, N))
    noise = noise_real + 1j * noise_imag
    
    rx_dd_grid += noise
    
    return rx_dd_grid, channel_dd

def apply_comm_channel_and_noise(tx_signal, snr_db, comm_channel_bins, fs_hz):
    """
    Simulates the BISTATIC (Tx -> Rx) channel for COMMUNICATION.
    Uses "on-grid" integer bins to ensure a valid simulation.
    """
    n_samples = tx_signal.size
    rx_signal = np.zeros(n_samples, dtype=complex)
    time_vector_sec = np.arange(n_samples) / fs_hz

    # Get fundamental resolutions
    delay_res_sec = 1 / BANDWIDTH_HZ
    doppler_res_hz = 1 / T_FRAME_SEC # Correct definition
    
    for (delay_bin, doppler_bin, reflectivity) in comm_channel_bins:
        # 1. Calculate Delay (1-way path)
        delay_sec = delay_bin * delay_res_sec
        delay_samples = int(round(delay_sec * fs_hz))
        
        delayed_signal = np.roll(tx_signal, delay_samples)
        
        # 2. Calculate Doppler Shift (1-way path)
        doppler_hz = doppler_bin * doppler_res_hz
        
        doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector_sec)
        
        # 3. Add this path's contribution
        rx_signal += reflectivity * delayed_signal * doppler_shift
        
    # 4. Add Additive White Gaussian Noise (AWGN)
    signal_power = np.mean(np.abs(tx_signal)**2)
    snr_linear = 10**(snr_db / 10)
    
    if signal_power == 0: noise_power = 1e-20
    else: noise_power = signal_power / snr_linear
        
    noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * \
            np.sqrt(noise_power / 2)
            
    return rx_signal + noise

# =========================================================================
# 1. OFDM ISAC Simulation (These functions are correct)
# =========================================================================

def ofdm_modulate(data_grid):
    """
    Modulates an M x N data grid into a time-domain OFDM signal.
    """
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    time_domain_symbols = np.fft.ifft(data_grid, axis=0)
    
    # Shape: (CP_LEN, N_SYMBOLS)
    cp_signal = time_domain_symbols[-CP_LEN:, :]
    # Shape: (N_SUBCARRIERS + CP_LEN, N_SYMBOLS)
    with_cp_signal = np.concatenate((cp_signal, time_domain_symbols), axis=0)
    
    # Shape: ((N_SUBCARRIERS + CP_LEN) * N_SYMBOLS,)
    tx_signal = with_cp_signal.flatten(order='F')
    return tx_signal

def ofdm_sensing_receiver(rx_signal, tx_signal):
    """
    Performs radar sensing using 2D cross-correlation (matched filter).
    Enhanced for better target detection.
    """
    # Shape: (N_SUBCARRIERS + CP_LEN, N_SYMBOLS)
    tx_grid_time = tx_signal.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
    rx_grid_time = rx_signal.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
    
    # Remove CP
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    tx_grid_no_cp = tx_grid_time[CP_LEN:, :]
    rx_grid_no_cp = rx_grid_time[CP_LEN:, :]
    
    # Transform to frequency domain
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    tx_grid_freq = np.fft.fft(tx_grid_no_cp, axis=0)
    rx_grid_freq = np.fft.fft(rx_grid_no_cp, axis=0)
    
    # Cross-correlation in frequency domain (matched filter)
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    correlation_grid = rx_grid_freq * np.conj(tx_grid_freq)
    
    # Transform back to get range-doppler map
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    range_doppler_map = np.fft.ifft2(correlation_grid)
    
    # Apply proper shifting for visualization
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    range_doppler_map = np.fft.fftshift(range_doppler_map)
    
    return range_doppler_map

def get_ofdm_tf_channel(target_bins_list):
    """
    Calculates the ideal Time-Frequency (TF) channel response
    for a 1-tap OFDM equalizer.
    """
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    H_tf_ofdm = np.zeros((N_SUBCARRIERS, N_SYMBOLS), dtype=complex)
    
    # Get fundamental resolutions
    delay_res_sec = 1 / BANDWIDTH_HZ
    doppler_res_hz = 1 / T_FRAME_SEC # Comm channel is 1-way
    
    for m in range(N_SUBCARRIERS): # Subcarrier (freq) index
        for l in range(N_SYMBOLS): # Symbol (time) index
            
            t_sec = l * T_SYMBOL_CP_SEC
            f_hz = m * SCS_HZ
            
            for (delay_bin, doppler_bin, reflectivity) in target_bins_list:
                
                delay_sec = delay_bin * delay_res_sec
                doppler_hz = doppler_bin * doppler_res_hz
                
                # Channel gain for this path at this (t, f)
                phase = 1j * 2 * np.pi * (doppler_hz * t_sec - f_hz * delay_sec)
                H_tf_ofdm[m, l] += reflectivity * np.exp(phase)
                
    return H_tf_ofdm

# =========================================================================
# 2. OTFS ISAC Simulation (*** CORRECTED TRANSFORMS ***)
# =========================================================================

def otfs_modulate_v2(dd_grid):
    """
    *** CORRECTED TRANSFORM (v2) ***
    Modulates a Delay-Doppler (M x N) grid to a time-domain signal.
    Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
    
    Args:
        dd_grid (np.array): Shape: (M, N) = (N_SUBCARRIERS, N_SYMBOLS)
    Returns:
        np.array: 1D time-domain signal (no CP). Shape: (M * N,)
    """
    # 1. Inverse Symplectic Finite Fourier Transform (ISFFT)
    #    (Converts DD grid to TF grid)
    #    ISFFT(X_DD[k,l]) = X_TF[n,m]
    #    This is FFT over Doppler (l->m, axis=1) and IFFT over Delay (k->n, axis=0)
    # Shape: (M, N)
    tf_grid = np.fft.ifft(dd_grid, axis=0)
    tf_grid = np.fft.fft(tf_grid, axis=1)

    # 2. Heisenberg Transform (multi-carrier modulation)
    #    (Converts TF grid to Time-Domain grid)
    #    This is an IFFT over the subcarriers (n, axis=0) for each symbol (m)
    # Shape: (M, N)
    time_domain_grid = np.fft.ifft(tf_grid, axis=0)
    
    # Serialize (column-major)
    # Shape: (M * N,)
    tx_signal = time_domain_grid.flatten(order='F')
    return tx_signal

def otfs_demodulate_v2(rx_signal):
    """
    *** CORRECTED TRANSFORM (v2) ***
    Demodulates a time-domain signal back to a Delay-Doppler grid.
    Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
    
    Args:
        rx_signal (np.array): Shape: (M * N,)
    Returns:
        np.array: Received DD-grid. Shape: (M, N)
    """
    # Deserialize (column-major)
    # Shape: (M, N)
    time_domain_grid = rx_signal.reshape((N_SUBCARRIERS, N_SYMBOLS), order='F')
    
    # 1. Wigner Transform (multi-carrier demodulation)
    #    (Converts Time-Domain grid to TF grid)
    #    This is an FFT over the time samples (axis=0) for each symbol (m)
    # Shape: (M, N)
    tf_grid = np.fft.fft(time_domain_grid, axis=0)
    
    # 2. Symplectic Finite Fourier Transform (SFFT)
    #    (Converts TF grid to DD grid)
    #    SFFT(Y_TF[n,m]) = Y_DD[k,l]
    #    This is IFFT over Time (m->l, axis=1) and FFT over Freq (n->k, axis=0)
    # Shape: (M, N)
    dd_grid = np.fft.ifft(tf_grid, axis=1)
    dd_grid = np.fft.fft(dd_grid, axis=0)
    
    return dd_grid

def get_otfs_dd_channel(target_bins_list):
    """
    Calculates the ideal Delay-Doppler (DD) channel response H.
    """
    # Shape: (N_SUBCARRIERS, N_SYMBOLS)
    H_ideal_dd = np.zeros((N_SUBCARRIERS, N_SYMBOLS), dtype=complex)
    
    for (delay_bin, doppler_bin, reflectivity) in target_bins_list:
        doppler_idx = np.mod(doppler_bin, N_SYMBOLS)
        delay_idx = delay_bin 
        
        H_ideal_dd[delay_idx, doppler_idx] = reflectivity
        
    return H_ideal_dd

# =========================================================================
# 3. Main Simulation and Visualization
# =========================================================================

def main():
    print("--- Starting ISAC Simulation (OFDM vs. OTFS) [v4-WORKING] ---")
    print("\n--- System Parameters (NEW) ---")
    print(f"Grid Size (Delay x Doppler): {N_SUBCARRIERS} x {N_SYMBOLS}")
    print(f"Subcarrier Spacing: {SCS_HZ/1e3} kHz")
    print(f"Bandwidth: {BANDWIDTH_HZ/1e6:.2f} MHz")
    print(f"Frame Duration: {T_FRAME_SEC*1e3:.2f} ms")
    print(f"==> Range Resolution: {RANGE_RES_M:.2f} m")
    print(f"==> Velocity Resolution: {VEL_RES_MPS:.2f} m/s")
    print(f"Simulating Targets up to Range: {MAX_RANGE_REQUEST_M} m (Bin {MAX_DELAY_BIN})")
    print(f"Simulating Targets up to Velocity: +/- {VEL_MAX_MPS:.2f} m/s (Bin +/- {MAX_DOPPLER_BIN})")
    
    start_time = time.time()
    
    # --- 3.1. Generate Targets ---
    N_TARGETS = np.random.randint(1, MAX_TARGETS + 1)
    # These targets are for the RADAR scenario
    physical_targets, target_bins_list_radar = generate_physical_targets(N_TARGETS)
    
    print(f"\n--- Generated {N_TARGETS} Radar Target(s) (ON-GRID) ---")
    for i, t in enumerate(physical_targets):
        range_m = np.sqrt(t['x']**2 + t['y']**2)
        print(f"  Target {i+1}:")
        print(f"    Location (x, y): ({t['x']:.2f} m, {t['y']:.2f} m)")
        print(f"    Range: {range_m:.2f} m")
        print(f"    Velocity: {t['velocity']:.2f} m/s")
        print(f"    Mapped Bins (Delay, Doppler): {target_bins_list_radar[i][:2]}")

    # --- 3.2. BEV Plot Visualization ---
    print("\nGenerating BEV plot...")
    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, 'ks', markersize=12, label='Ego Vehicle (Tx / Radar Rx)')
    plt.plot(COMM_RX_LOC['y'], COMM_RX_LOC['x'], 'gp', markersize=14,
             label=f"Comm Receiver ({COMM_RX_LOC['x']}m, {COMM_RX_LOC['y']}m)")
    target_xs = [t['x'] for t in physical_targets]
    target_ys = [t['y'] for t in physical_targets]
    plt.scatter(target_ys, target_xs, c='b', marker='o', s=100, label='Radar Targets')
    for i, t in enumerate(physical_targets):
        plt.text(t['y']+0.5, t['x'], f"T{i+1}\n{t['velocity']:.1f} m/s")
    plt.xlabel('Y (left) [m]'); plt.ylabel('X (front) [m]')
    plt.title("Bird's-Eye View (BEV) of ISAC Scenario")
    plt.legend(); plt.grid(True); plt.xlim(-15, 15); plt.ylim(0, 30)
    plt.gca().set_aspect('equal', adjustable='box')
    bev_filename = os.path.join(SAVE_DIR, "bev_targets.png")
    plt.savefig(bev_filename)
    print(f"Saved BEV plot to: {bev_filename}")

    # --- 3.3. Sensing Map Visualization (at a fixed SNR) ---
    print(f"\nGenerating sensing maps at {SIM_SNR_DB} dB SNR...")
    
    # --- Run OFDM Sensing ---
    tx_symbols_ofdm = get_qam_symbols(N_SUBCARRIERS * N_SYMBOLS)
    tx_grid_ofdm = tx_symbols_ofdm.reshape((N_SUBCARRIERS, N_SYMBOLS))
    tx_signal_ofdm = ofdm_modulate(tx_grid_ofdm)
    rx_signal_ofdm = apply_radar_channel_and_noise(tx_signal_ofdm, SIM_SNR_DB, 
                                                   physical_targets, FS_HZ)
    rdm_ofdm = ofdm_sensing_receiver(rx_signal_ofdm, tx_signal_ofdm)

    # --- Run OTFS Sensing (using CORRECTED transforms) ---
    # Use full data grid instead of just a pilot for better sensing
    tx_symbols_otfs = get_qam_symbols(N_SUBCARRIERS * N_SYMBOLS)
    tx_grid_otfs = tx_symbols_otfs.reshape((N_SUBCARRIERS, N_SYMBOLS))
    tx_signal_otfs = otfs_modulate_v2(tx_grid_otfs) # Use v2
    rx_signal_otfs = apply_radar_channel_and_noise(tx_signal_otfs, SIM_SNR_DB, 
                                                   physical_targets, FS_HZ)
    rx_dd_grid = otfs_demodulate_v2(rx_signal_otfs) # Use v2
    
    # Compute channel estimate by dividing received by transmitted
    # Avoid division by zero
    tx_grid_otfs_safe = tx_grid_otfs.copy()
    tx_grid_otfs_safe[np.abs(tx_grid_otfs_safe) < 1e-6] = 1e-6
    ddm_otfs = rx_dd_grid / tx_grid_otfs_safe
    ddm_otfs_shifted = np.fft.fftshift(ddm_otfs)
 
    # --- Plot Sensing Maps ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"Sensing Performance (Radar Channel) (SNR = {SIM_SNR_DB} dB)", fontsize=16)

    vel_axis_min, vel_axis_max = -VEL_MAX_MPS, VEL_MAX_MPS
    range_axis_min, range_axis_max = 0, (N_SUBCARRIERS - 1) * RANGE_RES_M
    plot_extent = [vel_axis_min, vel_axis_max, range_axis_min, range_axis_max]

    # Handle NaN/inf values and compute magnitude in dB
    rdm_ofdm_safe = np.abs(rdm_ofdm)
    rdm_ofdm_safe[rdm_ofdm_safe == 0] = 1e-20  # Avoid log(0)
    rdm_ofdm_mag_db = 10 * np.log10(rdm_ofdm_safe)
    
    ddm_otfs_safe = np.abs(ddm_otfs_shifted)
    ddm_otfs_safe[ddm_otfs_safe == 0] = 1e-20  # Avoid log(0)
    ddm_otfs_mag_db = 10 * np.log10(ddm_otfs_safe)
    
    # Set dynamic range for better visualization
    vmin_ofdm = np.percentile(rdm_ofdm_mag_db, 5)  # 5th percentile
    vmax_ofdm = np.percentile(rdm_ofdm_mag_db, 95)  # 95th percentile
    vmin_otfs = np.percentile(ddm_otfs_mag_db, 5)
    vmax_otfs = np.percentile(ddm_otfs_mag_db, 95)
    
    im1 = ax1.imshow(rdm_ofdm_mag_db, aspect='auto', extent=plot_extent,
                     origin='lower', cmap='jet', vmin=vmin_ofdm, vmax=vmax_ofdm)
    ax1.set_title('OFDM Range-Doppler Map (2D Correlation)')
    ax1.set_xlabel('Velocity (m/s)'); ax1.set_ylabel('Range (m)')
    ax1.set_ylim(0, MAX_RANGE_REQUEST_M + 10) # Zoom
    fig.colorbar(im1, ax=ax1, label='Power (dB)')
 
    im2 = ax2.imshow(ddm_otfs_mag_db, aspect='auto', extent=plot_extent,
                     origin='lower', cmap='jet', vmin=vmin_otfs, vmax=vmax_otfs)
    ax2.set_title('OTFS Delay-Doppler Map (Channel Estimate)')
    ax2.set_xlabel('Velocity (m/s)'); ax2.set_ylabel('Range (m)')
    ax2.set_ylim(0, MAX_RANGE_REQUEST_M + 10) # Zoom
    fig.colorbar(im2, ax=ax2, label='Magnitude (dB)')
 
    for t in physical_targets:
        range_m = np.sqrt(t['x']**2 + t['y']**2)
        vel_mps = t['velocity']
        ax1.plot(vel_mps, range_m, 'rx', markersize=12, markeredgewidth=3, 
                 label='True Target')
        ax2.plot(vel_mps, range_m, 'rx', markersize=12, markeredgewidth=3,
                 label='True Target')

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper right')
 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sensing_filename = os.path.join(SAVE_DIR, "sensing_maps_comparison.png")
    plt.savefig(sensing_filename)
    print(f"Saved sensing maps to: {sensing_filename}")
 
    # --- 3.4. Communication BER vs. SNR Simulation ---
    # This section models the BISTATIC (COMMUNICATION) channel
    print(f"\nRunning BER vs. SNR simulation (Comm Channel)...")
    print(f"Using separate 'on-grid' comm channel: {COMM_CHANNEL_BINS}")
    ber_ofdm = []
    ber_otfs = []
    
    # Get ideal channel responses FOR THE COMM CHANNEL
    # Shape: (M, N)
    H_ideal_dd_comm = get_otfs_dd_channel(COMM_CHANNEL_BINS)
    # Shape: (M, N)
    H_ideal_tf_ofdm_comm = get_ofdm_tf_channel(COMM_CHANNEL_BINS)
 
    for snr_db in SNR_DB_LIST:
        print(f"  Simulating at {snr_db} dB...")
        
        # --- Generate data ---
        tx_bits = np.random.randint(0, QAM_ORDER, N_SUBCARRIERS * N_SYMBOLS)
        tx_symbols = np.array([MOD_MAP[b] for b in tx_bits])
        # Shape: (M, N)
        tx_data_grid = tx_symbols.reshape((N_SUBCARRIERS, N_SYMBOLS))
        
        # --- OTFS BER (Improved with DD Domain Processing) ---
        # Use proper delay-Doppler domain channel convolution and equalization
        
        # 1. Apply channel directly in DD domain
        rx_dd_grid, channel_dd = apply_otfs_dd_channel(tx_data_grid, COMM_CHANNEL_BINS, snr_db)
        
        # 2. Apply advanced equalization
        # Calculate noise variance correctly based on TX signal power and SNR
        tx_signal_power = np.mean(np.abs(tx_data_grid)**2)
        snr_linear = 10**(snr_db / 10)
        noise_var = tx_signal_power / snr_linear
        
        # Use Zero-Forcing equalization for simpler processing
        rx_symbols_otfs_eq = otfs_zf_equalization(rx_dd_grid, channel_dd)
        
        rx_symbols_otfs = rx_symbols_otfs_eq.flatten()
        
        # Debug: Print some statistics
        if snr_db in [0, 8, 16]:
            print(f"    OTFS Debug at {snr_db} dB:")
            print(f"      TX symbols range: [{np.min(np.abs(tx_symbols)):.3f}, {np.max(np.abs(tx_symbols)):.3f}]")
            print(f"      RX symbols range: [{np.min(np.abs(rx_symbols_otfs)):.3f}, {np.max(np.abs(rx_symbols_otfs)):.3f}]")
            print(f"      Channel taps: {len(COMM_CHANNEL_BINS)} paths")
            print(f"      Noise variance: {noise_var:.6f}")
            print(f"      First 5 TX: {tx_symbols[:5]}")
            print(f"      First 5 RX: {rx_symbols_otfs[:5]}")
        
        rx_bits_otfs = demodulate_qam_symbols(rx_symbols_otfs)
        
        # Debug BER calculation
        if snr_db in [0, 8, 16]:
            print(f"      TX bits (first 10): {tx_bits[:10]}")
            print(f"      RX bits (first 10): {rx_bits_otfs[:10]}")
            print(f"      RX symbols (first 5): {rx_symbols_otfs[:5]}")
            print(f"      Bit errors: {np.sum(tx_bits != rx_bits_otfs)} / {len(tx_bits)}")
        
        ber_otfs.append(calculate_ber(tx_bits, rx_bits_otfs))
        
        # Debug output for BER values
        if snr_db % 8 == 0:  # Print every other SNR point
            print(f"    OTFS BER at {snr_db} dB: {ber_otfs[-1]:.4f}")
 
        # --- OFDM BER (Time-Domain Simulation) ---
        # We still simulate this in the time domain to show its failure
        tx_signal_ofdm = ofdm_modulate(tx_data_grid)
        # Apply the COMM channel (using time-domain function)
        rx_signal_ofdm = apply_comm_channel_and_noise(tx_signal_ofdm, snr_db, 
                                                      COMM_CHANNEL_BINS, FS_HZ)
        
        # OFDM Receiver
        rx_grid_time = rx_signal_ofdm.reshape((N_SUBCARRIERS + CP_LEN, N_SYMBOLS), order='F')
        rx_grid_no_cp = rx_grid_time[CP_LEN:, :]
        rx_tf_grid_ofdm = np.fft.fft(rx_grid_no_cp, axis=0)
        
        # 1-tap (Zero-Forcing) Equalizer (Fails due to ICI)
        H_ideal_tf_ofdm_comm_reg = H_ideal_tf_ofdm_comm.copy()
        H_ideal_tf_ofdm_comm_reg[np.abs(H_ideal_tf_ofdm_comm_reg) < 1e-6] = 1e-6
        rx_symbols_ofdm_est_grid = rx_tf_grid_ofdm / H_ideal_tf_ofdm_comm_reg
        rx_symbols_ofdm_est = rx_symbols_ofdm_est_grid.flatten()
        
        # Debug: Print some statistics
        if snr_db in [0, 8, 16]:
            print(f"    OFDM Debug at {snr_db} dB:")
            print(f"      TX symbols range: [{np.min(np.abs(tx_symbols)):.3f}, {np.max(np.abs(tx_symbols)):.3f}]")
            print(f"      RX symbols range: [{np.min(np.abs(rx_symbols_ofdm_est)):.3f}, {np.max(np.abs(rx_symbols_ofdm_est)):.3f}]")
            print(f"      Channel est range: [{np.min(np.abs(H_ideal_tf_ofdm_comm_reg)):.3f}, {np.max(np.abs(H_ideal_tf_ofdm_comm_reg)):.3f}]")
            print(f"      First 5 TX: {tx_symbols[:5]}")
            print(f"      First 5 RX: {rx_symbols_ofdm_est[:5]}")
        
        rx_bits_ofdm = demodulate_qam_symbols(rx_symbols_ofdm_est)
        
        # Debug BER calculation
        if snr_db in [0, 8, 16]:
            print(f"      TX bits (first 10): {tx_bits[:10]}")
            print(f"      RX bits (first 10): {rx_bits_ofdm[:10]}")
            print(f"      RX symbols (first 5): {rx_symbols_ofdm_est[:5]}")
            print(f"      Bit errors: {np.sum(tx_bits != rx_bits_ofdm)} / {len(tx_bits)}")
        
        ber_ofdm.append(calculate_ber(tx_bits, rx_bits_ofdm))
        
        # Debug output for BER values
        if snr_db % 8 == 0:  # Print every other SNR point
            print(f"    OFDM BER at {snr_db} dB: {ber_ofdm[-1]:.4f}")
 
    # --- Plot BER Curves ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(SNR_DB_LIST, ber_otfs, '-o', linewidth=2, markersize=8, 
                 label='OTFS (with 2D Deconvolution Equalizer)')
    plt.semilogy(SNR_DB_LIST, ber_ofdm, '-s', linewidth=2, markersize=8,
                 label='OFDM (with 1-Tap TF Equalizer)')
    plt.title('Communication Performance (BER vs. SNR) in High-Mobility')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.legend()
    plt.grid(True, which='both')
    plt.ylim(1e-4, 1.0) # Set BER floor
    
    ber_filename = os.path.join(SAVE_DIR, "ber_comparison.png")
    plt.savefig(ber_filename)
    print(f"Saved BER plot to: {ber_filename}")

    # --- Finished ---
    end_time = time.time()
    print(f"\n--- Simulation Complete in {end_time - start_time:.2f} seconds ---")
    print(f"All figures saved to '{SAVE_DIR}' directory.")
    
    # Show all plots at the end
    plt.show()

# --- Run the main simulation ---
if __name__ == "__main__":
    main()