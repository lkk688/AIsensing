#!/usr/bin/env python3
"""
RF Link Test - Step 3: RX (OFDM Demodulation)

Receives and demodulates OFDM signal:
1. Estimates CFO from pilot tone
2. Applies CFO correction
3. Detects STF for timing sync
4. Uses LTF for channel estimation
5. Demodulates OFDM data symbols with pilot-based phase tracking

Run on the local RX device.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# OFDM Parameters (must match TX)
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)
SYMBOL_LEN = N_FFT + N_CP


def estimate_cfo_from_tone(rx_samples, fs, expected_freq=100e3):
    """Estimate CFO from pilot tone using FFT peak detection."""
    N = len(rx_samples)
    fft = np.fft.fft(rx_samples)
    fft_mag = np.abs(fft)

    # Remove DC and near-DC
    fft_mag[0] = 0
    fft_mag[1:30] = 0
    fft_mag[-30:] = 0

    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1/fs)

    # Parabolic interpolation
    if 1 < peak_idx < N-1:
        alpha = np.log(fft_mag[peak_idx-1] + 1e-10)
        beta = np.log(fft_mag[peak_idx] + 1e-10)
        gamma = np.log(fft_mag[peak_idx+1] + 1e-10)
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        detected_freq = freq_bins[peak_idx] + p * (freq_bins[1] - freq_bins[0])
    else:
        detected_freq = freq_bins[peak_idx]

    cfo = detected_freq - expected_freq
    return cfo


def apply_cfo_correction(samples, cfo, fs):
    """Apply CFO correction."""
    n = np.arange(len(samples))
    return samples * np.exp(-1j * 2 * np.pi * cfo * n / fs).astype(np.complex64)


def create_stf_ref(N=64):
    """Create STF reference (must match TX, including CP)."""
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0])
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs))
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, 2).astype(np.complex64)
    # Add CP to match TX
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp


def create_ltf_ref(N=64):
    """Create LTF reference (must match TX)."""
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if i % 2 == 0 else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, 2).astype(np.complex64)
    return ltf, X


def detect_stf(rx, stf_ref, search_range=50000):
    """Detect STF using cross-correlation."""
    search_len = min(search_range, len(rx) - len(stf_ref))
    if search_len <= 0:
        return -1, 0.0

    corr = np.abs(np.correlate(rx[:search_len], stf_ref, mode='valid'))
    stf_energy = np.sqrt(np.sum(np.abs(stf_ref)**2))
    corr = corr / stf_energy

    peak_idx = np.argmax(corr)
    peak_val = corr[peak_idx]

    return peak_idx, peak_val, corr


def extract_ofdm_symbol(rx, start_idx):
    """Extract one OFDM symbol, removing CP."""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start_idx + N_CP:start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(sym))


def channel_estimate_from_ltf(rx, ltf_start, ltf_freq_ref):
    """Estimate channel from LTF."""
    # LTF has two symbols
    Y0 = extract_ofdm_symbol(rx, ltf_start)
    Y1 = extract_ofdm_symbol(rx, ltf_start + SYMBOL_LEN)

    if Y0 is None or Y1 is None:
        return None

    # Average
    Y_avg = (Y0 + Y1) / 2

    # Channel estimate on used subcarriers
    H = np.ones(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for k in used:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 0.1:
            H[idx] = Y_avg[idx] / ltf_freq_ref[idx]

    return H, Y0, Y1


def qpsk_demap(symbols):
    """Demap QPSK symbols to bits."""
    bits = []
    for s in symbols:
        re = np.real(s) >= 0
        im = np.imag(s) >= 0
        if re and im:
            bits.extend([0, 0])
        elif not re and im:
            bits.extend([0, 1])
        elif not re and not im:
            bits.extend([1, 1])
        else:
            bits.extend([1, 0])
    return np.array(bits, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 3)")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--num_ofdm_syms", type=int, default=20, help="Expected OFDM symbols")
    ap.add_argument("--num_captures", type=int, default=5, help="Number of captures")
    ap.add_argument("--output_dir", default="rf_link_step3_results", help="Output directory")
    ap.add_argument("--kp", type=float, default=0.1, help="Phase tracking Kp")
    ap.add_argument("--ki", type=float, default=0.01, help="Phase tracking Ki")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Create reference signals
    stf_ref = create_stf_ref(N_FFT)
    ltf_ref, ltf_freq_ref = create_ltf_ref(N_FFT)

    # Known TX data for BER calculation
    rng = np.random.default_rng(12345)
    tx_bits = rng.integers(0, 2, size=args.num_ofdm_syms * N_DATA * 2)

    print(f"RF Link Test - Step 3: RX (OFDM Demodulation)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Expected OFDM symbols: {args.num_ofdm_syms}")
    print()

    # Configure SDR
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = args.buf_size

    # Flush
    for _ in range(3):
        sdr.rx()

    print("Starting captures...")
    print("-" * 60)

    results = []

    for cap_idx in range(args.num_captures):
        rx_raw = sdr.rx()
        rx = rx_raw.astype(np.complex64) / (2**14)
        rx = rx - np.mean(rx)  # Remove DC

        # Step 1: CFO estimation from pilot tone (use first part of buffer)
        # Tone is at the beginning of the frame
        tone_search_len = min(30000, len(rx))
        cfo = estimate_cfo_from_tone(rx[:tone_search_len], args.fs, 100e3)

        # Step 2: CFO correction - MUST be done before STF detection!
        rx_cfo = apply_cfo_correction(rx, cfo, args.fs)

        # Step 3: Detect STF on CFO-corrected signal
        stf_idx, stf_peak, stf_corr = detect_stf(rx_cfo, stf_ref)

        # Debug: show more info about STF detection
        rx_power_db = 10 * np.log10(np.mean(np.abs(rx)**2) + 1e-10)
        print(f"[{cap_idx+1}] RX Power: {rx_power_db:.1f} dB, CFO: {cfo:.1f} Hz, STF peak: {stf_peak:.4f} at {stf_idx}")

        if stf_idx < 0 or stf_peak < 0.05:  # Lower threshold for debugging
            print(f"    STF detection weak - trying anyway")
            if stf_peak < 0.02:
                continue

        # STF reference already includes CP, so LTF starts right after
        ltf_start = stf_idx + len(stf_ref)

        # Step 4: Channel estimation from LTF
        ch_result = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref)
        if ch_result is None:
            print(f"[{cap_idx+1}] LTF extraction failed")
            continue

        H, Y_ltf0, Y_ltf1 = ch_result

        # Step 5: Demodulate OFDM symbols with phase tracking
        payload_start = ltf_start + 2 * SYMBOL_LEN  # After two LTF symbols

        pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
        pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS])
        data_idx = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS])

        all_data_syms = []
        phase_errors = []
        pilot_powers = []
        phase_acc = 0.0
        freq_acc = 0.0

        for sym_idx in range(args.num_ofdm_syms):
            sym_start = payload_start + sym_idx * SYMBOL_LEN
            Y = extract_ofdm_symbol(rx_cfo, sym_start)

            if Y is None:
                break

            # Equalize
            Y_eq = np.zeros_like(Y)
            for k in range(-26, 27):
                if k == 0:
                    continue
                idx = (k + N_FFT) % N_FFT
                if np.abs(H[idx]) > 1e-6:
                    Y_eq[idx] = Y[idx] / H[idx]

            # Pilot-based phase tracking
            pilot_sign = 1 if sym_idx % 2 == 0 else -1
            expected_pilots = pilot_sign * pilot_pattern

            rx_pilots = Y_eq[pilot_idx]
            phase_err = np.angle(np.sum(rx_pilots * np.conj(expected_pilots)))

            # PI loop
            freq_acc += args.ki * phase_err
            phase_acc += freq_acc + args.kp * phase_err

            # Apply phase correction
            Y_eq *= np.exp(-1j * phase_acc)

            # Extract data
            data_syms = Y_eq[data_idx]
            all_data_syms.append(data_syms)

            phase_errors.append(phase_err)
            pilot_powers.append(np.mean(np.abs(rx_pilots)**2))

        if len(all_data_syms) == 0:
            print(f"[{cap_idx+1}] No OFDM symbols decoded")
            continue

        # Compute BER
        all_data_syms = np.concatenate(all_data_syms)
        rx_bits = qpsk_demap(all_data_syms)

        num_bits = min(len(rx_bits), len(tx_bits))
        bit_errors = np.sum(rx_bits[:num_bits] != tx_bits[:num_bits])
        ber = bit_errors / num_bits

        # EVM calculation
        # Reconstruct expected symbols
        expected_syms = []
        for sym_idx in range(len(all_data_syms) // N_DATA):
            start_bit = sym_idx * N_DATA * 2
            end_bit = start_bit + N_DATA * 2
            if end_bit <= len(tx_bits):
                bits = tx_bits[start_bit:end_bit]
                syms = np.zeros(N_DATA, dtype=np.complex64)
                for i in range(N_DATA):
                    b0, b1 = bits[2*i], bits[2*i+1]
                    if b0 == 0 and b1 == 0:
                        syms[i] = 1 + 1j
                    elif b0 == 0 and b1 == 1:
                        syms[i] = -1 + 1j
                    elif b0 == 1 and b1 == 1:
                        syms[i] = -1 - 1j
                    else:
                        syms[i] = 1 - 1j
                expected_syms.append(syms / np.sqrt(2))

        if len(expected_syms) > 0:
            expected_flat = np.concatenate(expected_syms)
            n_compare = min(len(all_data_syms), len(expected_flat))
            evm = np.sqrt(np.mean(np.abs(all_data_syms[:n_compare] - expected_flat[:n_compare])**2))
        else:
            evm = float('inf')

        result = {
            'capture': cap_idx + 1,
            'cfo': cfo,
            'stf_peak': stf_peak,
            'ber': ber,
            'evm': evm,
            'num_syms': len(all_data_syms) // N_DATA,
            'bit_errors': bit_errors,
        }
        results.append(result)

        print(f"[{cap_idx+1}] CFO={cfo:.1f}Hz STF={stf_peak:.4f} BER={ber:.4f} ({bit_errors}/{num_bits}) EVM={evm:.4f}")

        # Create diagnostic plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. STF correlation
        ax = axes[0, 0]
        disp_len = min(30000, len(stf_corr))
        ax.plot(stf_corr[:disp_len])
        ax.axvline(stf_idx, color='r', linestyle='--', label=f'STF at {stf_idx}')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Correlation')
        ax.set_title(f'STF Detection (peak={stf_peak:.4f})')
        ax.legend()
        ax.grid(True)

        # 2. Channel magnitude
        ax = axes[0, 1]
        H_mag = np.abs(np.fft.fftshift(H))
        ax.plot(np.arange(-N_FFT//2, N_FFT//2), H_mag)
        ax.set_xlabel('Subcarrier')
        ax.set_ylabel('|H|')
        ax.set_title('Channel Estimate')
        ax.grid(True)

        # 3. Constellation
        ax = axes[0, 2]
        ax.scatter(np.real(all_data_syms), np.imag(all_data_syms), s=2, alpha=0.5)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'QPSK Constellation (BER={ber:.4f})')
        ax.axis('equal')
        ax.grid(True)

        # 4. Phase error over symbols
        ax = axes[1, 0]
        ax.plot(phase_errors)
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Phase Error (rad)')
        ax.set_title('Pilot Phase Errors')
        ax.grid(True)

        # 5. Pilot power
        ax = axes[1, 1]
        ax.plot(pilot_powers)
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Power')
        ax.set_title('Pilot Power per Symbol')
        ax.grid(True)

        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""
RF Link Test - Step 3 - Capture {cap_idx+1}

CFO Estimation:
  Detected CFO: {cfo:.1f} Hz

Synchronization:
  STF peak: {stf_peak:.4f}
  STF index: {stf_idx}

Demodulation:
  OFDM symbols decoded: {len(all_data_syms)//N_DATA}
  BER: {ber:.4f} ({bit_errors}/{num_bits} errors)
  EVM: {evm:.4f}

Phase Tracking:
  Kp={args.kp}, Ki={args.ki}
"""
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')

        fig.suptitle(f'RF Link Step 3 - Capture {cap_idx+1}', fontsize=14)
        fig.tight_layout()

        plot_path = os.path.join(args.output_dir, f'capture_{cap_idx+1:02d}.png')
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if len(results) > 0:
        bers = [r['ber'] for r in results]
        evms = [r['evm'] for r in results]
        cfos = [r['cfo'] for r in results]

        print(f"CFO: mean={np.mean(cfos):.1f} Hz, std={np.std(cfos):.1f} Hz")
        print(f"BER: mean={np.mean(bers):.4f}, min={np.min(bers):.4f}, max={np.max(bers):.4f}")
        print(f"EVM: mean={np.mean(evms):.4f}")

        if np.mean(bers) < 0.01:
            print("\n✅ OFDM demodulation successful! BER < 1%")
        elif np.mean(bers) < 0.1:
            print("\n⚠️  OFDM works but BER is high - check SNR")
        else:
            print("\n❌ OFDM demodulation has issues - BER too high")
    else:
        print("No successful captures")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except:
        pass


if __name__ == "__main__":
    main()
