#!/usr/bin/env python3
"""
RF Link Test - Step 4: RX (Full Packet with CRC)

Receives and verifies complete packets:
1. Estimates and corrects CFO
2. Detects preamble and synchronizes
3. Demodulates OFDM symbols
4. Applies majority voting if repetition was used
5. Verifies CRC and extracts payload

Run on the local RX device.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import zlib

# OFDM Parameters (must match TX)
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS])
N_DATA = len(DATA_SUBCARRIERS)
SYMBOL_LEN = N_FFT + N_CP
BITS_PER_OFDM_SYM = N_DATA * 2

MAGIC = b"AIS1"


def estimate_cfo_from_tone(rx_samples, fs, expected_freq=100e3):
    """Estimate CFO from pilot tone."""
    N = len(rx_samples)
    fft = np.fft.fft(rx_samples)
    fft_mag = np.abs(fft)
    fft_mag[0] = 0
    fft_mag[1:30] = 0
    fft_mag[-30:] = 0
    peak_idx = np.argmax(fft_mag)
    freq_bins = np.fft.fftfreq(N, 1/fs)
    if 1 < peak_idx < N-1:
        alpha = np.log(fft_mag[peak_idx-1] + 1e-10)
        beta = np.log(fft_mag[peak_idx] + 1e-10)
        gamma = np.log(fft_mag[peak_idx+1] + 1e-10)
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        detected_freq = freq_bins[peak_idx] + p * (freq_bins[1] - freq_bins[0])
    else:
        detected_freq = freq_bins[peak_idx]
    return detected_freq - expected_freq


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
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp


def create_ltf_ref(N=64):
    """Create LTF reference."""
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
        return -1, 0.0, np.array([])
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
    Y0 = extract_ofdm_symbol(rx, ltf_start)
    Y1 = extract_ofdm_symbol(rx, ltf_start + SYMBOL_LEN)
    if Y0 is None or Y1 is None:
        return None
    Y_avg = (Y0 + Y1) / 2
    H = np.ones(N_FFT, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0])
    for k in used:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 0.1:
            H[idx] = Y_avg[idx] / ltf_freq_ref[idx]
    return H


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


def majority_vote(bits, repeat):
    """Apply majority voting for repeated bits."""
    if repeat <= 1:
        return bits
    L = (len(bits) // repeat) * repeat
    bits = bits[:L].reshape(-1, repeat)
    return (np.sum(bits, axis=1) >= (repeat / 2)).astype(np.uint8)


def parse_packet(bits_bytes: bytes):
    """
    Parse received packet.

    Returns: (success, payload, crc_rx, crc_calc)
    """
    if len(bits_bytes) < 10:  # MAGIC(4) + LEN(2) + CRC(4)
        return False, b"", 0, 0

    if bits_bytes[:4] != MAGIC:
        return False, b"", 0, 0

    plen = int.from_bytes(bits_bytes[4:6], "little")
    need = 6 + plen + 4

    if len(bits_bytes) < need:
        return False, b"", 0, 0

    payload = bits_bytes[6:6+plen]
    crc_rx = int.from_bytes(bits_bytes[6+plen:6+plen+4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF

    return (crc_rx == crc_calc), payload, crc_rx, crc_calc


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 4)")
    ap.add_argument("--uri", default="ip:192.168.2.2", help="Pluto URI")
    ap.add_argument("--fc", type=float, default=915e6, help="Center frequency (Hz)")
    ap.add_argument("--fs", type=float, default=2e6, help="Sample rate (Hz)")
    ap.add_argument("--rx_gain", type=float, default=40, help="RX gain (dB)")
    ap.add_argument("--buf_size", type=int, default=2**17, help="RX buffer size")
    ap.add_argument("--max_ofdm_syms", type=int, default=200, help="Max OFDM symbols to try")
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4], help="Expected bit repetition")
    ap.add_argument("--tries", type=int, default=10, help="Number of capture attempts")
    ap.add_argument("--output_dir", default="rf_link_step4_results", help="Output directory")
    ap.add_argument("--outfile", default="recovered.bin", help="Output file for recovered data")
    ap.add_argument("--kp", type=float, default=0.1, help="Phase tracking Kp")
    ap.add_argument("--ki", type=float, default=0.01, help="Phase tracking Ki")
    args = ap.parse_args()

    import adi

    os.makedirs(args.output_dir, exist_ok=True)

    # Create reference signals
    stf_ref = create_stf_ref(N_FFT)
    ltf_ref, ltf_freq_ref = create_ltf_ref(N_FFT)

    print(f"RF Link Test - Step 4: RX (Full Packet)")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Expected repetition: {args.repeat}x")
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

    print(f"Attempting up to {args.tries} captures...")
    print("-" * 60)

    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)
    pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS])
    data_idx = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS])

    for t in range(args.tries):
        rx_raw = sdr.rx()
        rx = rx_raw.astype(np.complex64) / (2**14)
        rx = rx - np.mean(rx)

        # CFO estimation and correction
        cfo = estimate_cfo_from_tone(rx[:30000], args.fs, 100e3)
        rx_cfo = apply_cfo_correction(rx, cfo, args.fs)

        # STF detection
        stf_idx, stf_peak, stf_corr = detect_stf(rx_cfo, stf_ref)

        rx_power_db = 10 * np.log10(np.mean(np.abs(rx)**2) + 1e-10)

        if stf_idx < 0 or stf_peak < 0.02:
            print(f"[{t+1:02d}] No signal (STF peak={stf_peak:.4f})")
            continue

        # Channel estimation
        ltf_start = stf_idx + len(stf_ref)
        H = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref)
        if H is None:
            print(f"[{t+1:02d}] LTF failed")
            continue

        # Demodulate OFDM symbols with phase tracking
        payload_start = ltf_start + 2 * SYMBOL_LEN
        all_data_syms = []
        phase_acc = 0.0
        freq_acc = 0.0

        for sym_idx in range(args.max_ofdm_syms):
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

            freq_acc += args.ki * phase_err
            phase_acc += freq_acc + args.kp * phase_err
            Y_eq *= np.exp(-1j * phase_acc)

            data_syms = Y_eq[data_idx]
            all_data_syms.append(data_syms)

        if len(all_data_syms) == 0:
            print(f"[{t+1:02d}] No OFDM symbols")
            continue

        # Convert to bits
        all_data_syms = np.concatenate(all_data_syms)
        bits_raw = qpsk_demap(all_data_syms)

        # Apply majority voting
        bits = majority_vote(bits_raw, args.repeat)

        # Convert to bytes
        bits_bytes = np.packbits(bits).tobytes()

        # Try to parse packet
        success, payload, crc_rx, crc_calc = parse_packet(bits_bytes)

        if success:
            # Save payload
            with open(args.outfile, "wb") as f:
                f.write(payload)

            print(f"[{t+1:02d}] SUCCESS! Payload: {len(payload)} bytes, CRC: 0x{crc_calc:08X}")
            print(f"      CFO: {cfo:.1f} Hz, STF peak: {stf_peak:.4f}")
            print(f"      Saved to: {args.outfile}")

            # Show payload preview
            if len(payload) <= 64:
                try:
                    print(f"      Data: {payload}")
                except:
                    print(f"      Data (hex): {payload.hex()}")

            # Create success plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            ax = axes[0, 0]
            ax.plot(stf_corr[:min(40000, len(stf_corr))])
            ax.axvline(stf_idx, color='r', linestyle='--')
            ax.set_title(f'STF Detection (peak={stf_peak:.4f})')
            ax.grid(True)

            ax = axes[0, 1]
            ax.scatter(np.real(all_data_syms), np.imag(all_data_syms), s=2, alpha=0.5)
            ax.set_title('QPSK Constellation')
            ax.axis('equal')
            ax.grid(True)

            ax = axes[1, 0]
            H_mag = np.abs(np.fft.fftshift(H))
            ax.plot(np.arange(-N_FFT//2, N_FFT//2), H_mag)
            ax.set_title('Channel Estimate |H|')
            ax.grid(True)

            ax = axes[1, 1]
            ax.axis('off')
            summary = f"""
PACKET RECEIVED SUCCESSFULLY!

CFO: {cfo:.1f} Hz
STF peak: {stf_peak:.4f}
OFDM symbols: {len(all_data_syms)//N_DATA}

Payload length: {len(payload)} bytes
CRC32: 0x{crc_calc:08X}

Output file: {args.outfile}
"""
            ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace')

            fig.suptitle('RF Link Step 4 - Packet Received!', fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(args.output_dir, 'success.png'), dpi=120)
            plt.close(fig)

            break
        else:
            # CRC failed
            if bits_bytes[:4] == MAGIC:
                print(f"[{t+1:02d}] CRC FAIL: rx=0x{crc_rx:08X} calc=0x{crc_calc:08X}")
            else:
                print(f"[{t+1:02d}] Magic not found (CFO={cfo:.1f}Hz, STF={stf_peak:.4f})")

    else:
        print("\nNo valid packet recovered after all attempts.")

    # Cleanup
    try:
        sdr.rx_destroy_buffer()
    except:
        pass


if __name__ == "__main__":
    main()
