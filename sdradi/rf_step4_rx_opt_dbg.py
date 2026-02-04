#!/usr/bin/env python3
"""
RF Link Test - Step 4: RX (Full Packet with CRC) - OPT/DBG version
- Keep the same on-air assumptions as your working RX.
- No new protocol features.
- Heavily instrumented: saves one big debug figure per try, success or fail.

What gets plotted (per capture):
  1) Time-domain magnitude with key boundaries (tone/STF/LTF/payload start)
  2) Tone FFT spectrum + detected peak + CFO estimate
  3) STF correlation curve + detected index
  4) Channel estimate |H| and angle(H) on all bins
  5) Pilot power per symbol
  6) Phase error per symbol (unwrapped) + phase_acc/freq_acc
  7) Constellation: raw-eq (before CPE) vs after CPE
  8) EVM per symbol (rough)
  9) Byte parsing preview: first 64 bytes, MAGIC check, CRC summary
"""

import argparse
import os
import zlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# OFDM Parameters (must match TX)
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)
SYMBOL_LEN = N_FFT + N_CP
BITS_PER_OFDM_SYM = N_DATA * 2

MAGIC = b"AIS1"

# Keep your original indexing convention for consistency:
# idx = (k + N) % N   while using fftshift(fft()) / ifft(ifftshift())
pilot_idx = np.array([(k + N_FFT) % N_FFT for k in PILOT_SUBCARRIERS], dtype=int)
data_idx  = np.array([(k + N_FFT) % N_FFT for k in DATA_SUBCARRIERS], dtype=int)
used_subs = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
used_idx  = np.array([(k + N_FFT) % N_FFT for k in used_subs], dtype=int)


def estimate_cfo_from_tone(rx_samples: np.ndarray, fs: float, expected_freq=100e3,
                           search_bw=250e3):
    """
    CFO estimate from tone by FFT peak search near expected frequency.
    Optimization: only search around expected_freq +/- search_bw to avoid being distracted by OFDM bins.
    """
    x = rx_samples.astype(np.complex64)
    N = len(x)
    if N < 1024:
        return 0.0, 0.0, None, None  # not enough data

    win = np.hanning(N).astype(np.float32)
    X = np.fft.fftshift(np.fft.fft(x * win))
    mag = np.abs(X)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1/fs))

    # mask search region
    mask = (freqs > (expected_freq - search_bw)) & (freqs < (expected_freq + search_bw))
    if not np.any(mask):
        return 0.0, 0.0, freqs, mag

    # remove DC neighborhood just in case
    mask &= (np.abs(freqs) > 2e3)

    mag2 = mag.copy()
    mag2[~mask] = 0.0

    peak_idx = int(np.argmax(mag2))
    peak_mag = float(mag[peak_idx])
    f_peak = float(freqs[peak_idx])

    cfo = f_peak - expected_freq
    return float(cfo), peak_mag, freqs, mag


def apply_cfo_correction(samples: np.ndarray, cfo: float, fs: float):
    n = np.arange(len(samples), dtype=np.float64)
    return (samples * np.exp(-1j * 2*np.pi * cfo * n / fs)).astype(np.complex64)


def create_stf_ref(N=64):
    rng = np.random.default_rng(42)
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i] + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, 2).astype(np.complex64)  # 2N
    stf_with_cp = np.concatenate([stf[-N_CP:], stf]).astype(np.complex64)
    return stf_with_cp


def create_ltf_ref(N=64):
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N) % N] = (1.0 if i % 2 == 0 else -1.0) + 0j
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x]).astype(np.complex64)
    ltf = np.tile(ltf_sym, 2).astype(np.complex64)

    # Frequency reference (in the same bin indexing convention)
    # We'll return X as "ltf_freq_ref" to use Y/X per bin.
    return ltf, X.astype(np.complex64)


def detect_stf(rx: np.ndarray, stf_ref: np.ndarray, search_range=50000):
    """
    Cross-correlation detection, normalized by reference energy only (kept same spirit as your version).
    Enhancement: also return raw corr for plotting.
    """
    L = len(stf_ref)
    search_len = min(search_range, len(rx) - L)
    if search_len <= 0:
        return -1, 0.0, np.array([], dtype=np.float32)

    corr = np.abs(np.correlate(rx[:search_len], stf_ref, mode="valid")).astype(np.float32)
    stf_energy = float(np.sqrt(np.sum(np.abs(stf_ref)**2)) + 1e-12)
    corr_norm = corr / stf_energy

    peak_idx = int(np.argmax(corr_norm))
    peak_val = float(corr_norm[peak_idx])
    return peak_idx, peak_val, corr_norm


def extract_ofdm_symbol_fd(rx: np.ndarray, start_idx: int):
    """Extract (CP removed) then FFTSHIFT."""
    if start_idx + SYMBOL_LEN > len(rx):
        return None
    sym = rx[start_idx + N_CP : start_idx + SYMBOL_LEN]
    return np.fft.fftshift(np.fft.fft(sym)).astype(np.complex64)


def channel_estimate_from_ltf(rx: np.ndarray, ltf_start: int, ltf_freq_ref: np.ndarray):
    Y0 = extract_ofdm_symbol_fd(rx, ltf_start)
    Y1 = extract_ofdm_symbol_fd(rx, ltf_start + SYMBOL_LEN)
    if Y0 is None or Y1 is None:
        return None, None, None

    Y = (Y0 + Y1) * 0.5
    H = np.ones(N_FFT, dtype=np.complex64)
    eps = 1e-9
    # Only on used bins
    for k in used_subs:
        idx = (k + N_FFT) % N_FFT
        if np.abs(ltf_freq_ref[idx]) > 0.1:
            H[idx] = Y[idx] / (ltf_freq_ref[idx] + eps)
    return H, Y, (Y0, Y1)


def qpsk_demap_gray_vec(symbols: np.ndarray) -> np.ndarray:
    """
    Vectorized QPSK demap matching the TX Gray map:
      quadrant (re>=0, im>=0) -> 00
      (re<0, im>=0) -> 01
      (re<0, im<0) -> 11
      (re>=0, im<0) -> 10
    """
    re = np.real(symbols) >= 0
    im = np.imag(symbols) >= 0

    bits = np.empty((len(symbols), 2), dtype=np.uint8)
    # 00
    m = re & im
    bits[m, 0] = 0; bits[m, 1] = 0
    # 01
    m = (~re) & im
    bits[m, 0] = 0; bits[m, 1] = 1
    # 11
    m = (~re) & (~im)
    bits[m, 0] = 1; bits[m, 1] = 1
    # 10
    m = re & (~im)
    bits[m, 0] = 1; bits[m, 1] = 0

    return bits.reshape(-1)


def majority_vote(bits: np.ndarray, repeat: int) -> np.ndarray:
    if repeat <= 1:
        return bits.astype(np.uint8)
    L = (len(bits) // repeat) * repeat
    bits2 = bits[:L].reshape(-1, repeat)
    return (np.sum(bits2, axis=1) >= (repeat / 2)).astype(np.uint8)


def parse_packet(bits_bytes: bytes):
    if len(bits_bytes) < 10:
        return False, b"", 0, 0, 0
    if bits_bytes[:4] != MAGIC:
        return False, b"", 0, 0, 0

    plen = int.from_bytes(bits_bytes[4:6], "little")
    need = 6 + plen + 4
    if len(bits_bytes) < need:
        return False, b"", plen, 0, 0

    payload = bits_bytes[6:6+plen]
    crc_rx = int.from_bytes(bits_bytes[6+plen:6+plen+4], "little")
    crc_calc = zlib.crc32(payload) & 0xFFFFFFFF
    return (crc_rx == crc_calc), payload, plen, crc_rx, crc_calc


def nearest_qpsk(sym: np.ndarray) -> np.ndarray:
    """Nearest ideal QPSK points for rough EVM."""
    # Ideal points: (+/-1 +/-1)/sqrt(2)
    s = sym.copy()
    re = np.where(np.real(s) >= 0, 1.0, -1.0)
    im = np.where(np.imag(s) >= 0, 1.0, -1.0)
    ideal = (re + 1j*im) / np.sqrt(2)
    # fix Gray mapping doesn't matter for EVM; ideal set is same
    return ideal.astype(np.complex64)


def save_try_figure(outp: str, title: str,
                    rx: np.ndarray, fs: float,
                    tone_freqs, tone_mag, cfo: float, tone_peak_mag: float,
                    stf_corr: np.ndarray, stf_idx: int, stf_peak: float,
                    H: np.ndarray,
                    pilot_pow: np.ndarray,
                    phase_err: np.ndarray, phase_acc: np.ndarray, freq_acc: np.ndarray,
                    const_pre: np.ndarray, const_post: np.ndarray,
                    evm_sym: np.ndarray,
                    bytes_preview: bytes,
                    parse_info: str):
    fig = plt.figure(figsize=(20, 12))

    # 1) Time |rx|
    ax1 = fig.add_subplot(3, 3, 1)
    m = np.abs(rx)
    ax1.plot(m[:min(len(m), 80000)])
    ax1.axvline(stf_idx, color="r", ls="--", label="STF idx")
    ax1.set_title("Time |rx| (first 80k) + STF idx")
    ax1.grid(True)
    ax1.legend()

    # 2) Tone spectrum
    ax2 = fig.add_subplot(3, 3, 2)
    if tone_freqs is not None and tone_mag is not None:
        # show +/- 300k around expected
        f = tone_freqs
        mag = tone_mag
        mask = (f > -400e3) & (f < 400e3)
        ax2.plot(f[mask]/1e3, 20*np.log10(mag[mask] + 1e-12))
        ax2.set_title(f"Tone FFT (dB) | CFO={cfo:+.1f} Hz | peak_mag={tone_peak_mag:.3g}")
        ax2.set_xlabel("kHz")
        ax2.grid(True)
    else:
        ax2.text(0.1, 0.5, "Tone FFT not available", transform=ax2.transAxes)
        ax2.axis("off")

    # 3) STF corr
    ax3 = fig.add_subplot(3, 3, 3)
    if stf_corr.size > 0:
        ax3.plot(stf_corr[:min(len(stf_corr), 60000)])
        ax3.axvline(stf_idx, color="r", ls="--")
        ax3.set_title(f"STF corr (peak={stf_peak:.4f}, idx={stf_idx})")
        ax3.grid(True)
    else:
        ax3.text(0.1, 0.5, "No STF corr", transform=ax3.transAxes)
        ax3.axis("off")

    # 4) |H|
    ax4 = fig.add_subplot(3, 3, 4)
    if H is not None:
        ax4.plot(np.abs(np.fft.fftshift(H)))
        ax4.set_title("|H| (fftshift view)")
        ax4.grid(True)
    else:
        ax4.text(0.1, 0.5, "No H", transform=ax4.transAxes)
        ax4.axis("off")

    # 5) angle(H)
    ax5 = fig.add_subplot(3, 3, 5)
    if H is not None:
        ang = np.unwrap(np.angle(np.fft.fftshift(H)))
        ax5.plot(ang)
        ax5.set_title("angle(H) unwrap (fftshift view)")
        ax5.grid(True)
    else:
        ax5.text(0.1, 0.5, "No H", transform=ax5.transAxes)
        ax5.axis("off")

    # 6) Pilot power
    ax6 = fig.add_subplot(3, 3, 6)
    if pilot_pow.size > 0:
        ax6.plot(pilot_pow)
        ax6.set_title("Pilot power per OFDM symbol")
        ax6.grid(True)
    else:
        ax6.text(0.1, 0.5, "No pilot power", transform=ax6.transAxes)
        ax6.axis("off")

    # 7) Phase loop
    ax7 = fig.add_subplot(3, 3, 7)
    if phase_err.size > 0:
        ax7.plot(np.unwrap(phase_err), label="phase_err (unwrap)")
        ax7.plot(phase_acc, label="phase_acc")
        ax7.plot(freq_acc, label="freq_acc")
        ax7.set_title("Pilot loop traces")
        ax7.grid(True)
        ax7.legend()
    else:
        ax7.text(0.1, 0.5, "No phase loop traces", transform=ax7.transAxes)
        ax7.axis("off")

    # 8) Constellation pre/post
    ax8 = fig.add_subplot(3, 3, 8)
    if const_pre.size > 0:
        ax8.scatter(np.real(const_pre[:8000]), np.imag(const_pre[:8000]), s=3, alpha=0.25, label="pre-CPE")
    if const_post.size > 0:
        ax8.scatter(np.real(const_post[:8000]), np.imag(const_post[:8000]), s=3, alpha=0.25, label="post-CPE")
    ax8.set_title("Constellation (data bins)")
    ax8.axis("equal")
    ax8.grid(True)
    ax8.legend()

    # 9) EVM + bytes preview
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis("off")
    if evm_sym.size > 0:
        evm_txt = f"EVM(sym) mean={np.mean(evm_sym):.3f}, p90={np.quantile(evm_sym, 0.9):.3f}\n"
    else:
        evm_txt = "EVM(sym) N/A\n"

    pv = bytes_preview[:64]
    pv_hex = pv.hex()
    txt = (
        f"{parse_info}\n"
        f"{evm_txt}"
        f"Bytes[0:64] hex:\n{pv_hex}\n"
    )
    ax9.text(0.02, 0.98, txt, va="top", family="monospace")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outp, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="RF Link Test - RX (Step 4) OPT/DBG")
    ap.add_argument("--uri", default="ip:192.168.2.2")
    ap.add_argument("--fc", type=float, default=915e6)
    ap.add_argument("--fs", type=float, default=2e6)
    ap.add_argument("--rx_gain", type=float, default=40)
    ap.add_argument("--buf_size", type=int, default=2**17)
    ap.add_argument("--max_ofdm_syms", type=int, default=200)
    ap.add_argument("--repeat", type=int, default=1, choices=[1, 2, 4])
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--output_dir", default="rf_link_step4_results_opt")
    ap.add_argument("--outfile", default="recovered.bin")
    ap.add_argument("--kp", type=float, default=0.1)
    ap.add_argument("--ki", type=float, default=0.01)

    # debug knobs (no new features, just visibility/robust peak selection)
    ap.add_argument("--tone_hz", type=float, default=100e3)
    ap.add_argument("--tone_fft_len", type=int, default=30000)
    ap.add_argument("--tone_search_bw", type=float, default=250e3)
    ap.add_argument("--stf_search", type=int, default=50000)
    args = ap.parse_args()

    import adi
    os.makedirs(args.output_dir, exist_ok=True)

    stf_ref = create_stf_ref(N_FFT)
    ltf_ref, ltf_freq_ref = create_ltf_ref(N_FFT)

    print("RF Link Test - Step 4: RX (Full Packet) OPT/DBG")
    print(f"  URI: {args.uri}")
    print(f"  Center freq: {args.fc/1e6:.3f} MHz")
    print(f"  Sample rate: {args.fs/1e6:.3f} Msps")
    print(f"  RX gain: {args.rx_gain} dB")
    print(f"  Expected repetition: {args.repeat}x")
    print("")

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf_size)

    for _ in range(3):
        _ = sdr.rx()

    pilot_pattern = np.array([1, 1, 1, -1], dtype=np.complex64)

    print(f"Attempting up to {args.tries} captures...")
    print("-" * 60)

    for t in range(args.tries):
        rx_raw = sdr.rx().astype(np.complex64) / (2**14)
        rx = rx_raw - np.mean(rx_raw)

        # 1) CFO from tone (FFT peak near expected tone)
        tone_len = min(args.tone_fft_len, len(rx))
        cfo, tone_peak_mag, tone_freqs, tone_mag = estimate_cfo_from_tone(
            rx[:tone_len], args.fs, expected_freq=args.tone_hz, search_bw=args.tone_search_bw
        )
        rx_cfo = apply_cfo_correction(rx, cfo, args.fs)

        # 2) STF detection
        stf_idx, stf_peak, stf_corr = detect_stf(rx_cfo, stf_ref, search_range=args.stf_search)

        # if no STF, still save a debug figure
        if stf_idx < 0 or stf_peak < 0.02:
            outp = os.path.join(args.output_dir, f"try{t+1:02d}_NO_STF.png")
            title = f"try={t+1:02d} NO_STF | CFO={cfo:+.1f}Hz stf_peak={stf_peak:.4f}"
            save_try_figure(
                outp=outp,
                title=title,
                rx=rx_cfo, fs=args.fs,
                tone_freqs=tone_freqs, tone_mag=tone_mag, cfo=cfo, tone_peak_mag=tone_peak_mag,
                stf_corr=stf_corr, stf_idx=max(stf_idx, 0), stf_peak=stf_peak,
                H=None,
                pilot_pow=np.array([], dtype=np.float32),
                phase_err=np.array([], dtype=np.float32),
                phase_acc=np.array([], dtype=np.float32),
                freq_acc=np.array([], dtype=np.float32),
                const_pre=np.array([], dtype=np.complex64),
                const_post=np.array([], dtype=np.complex64),
                evm_sym=np.array([], dtype=np.float32),
                bytes_preview=b"",
                parse_info="STF not detected / too weak",
            )
            print(f"[{t+1:02d}] No signal (STF peak={stf_peak:.4f}) saved {outp}")
            continue

        # 3) LTF channel estimate
        ltf_start = stf_idx + len(stf_ref)
        H, Yltf, (Y0, Y1) = channel_estimate_from_ltf(rx_cfo, ltf_start, ltf_freq_ref)
        if H is None:
            outp = os.path.join(args.output_dir, f"try{t+1:02d}_LTF_FAIL.png")
            title = f"try={t+1:02d} LTF_FAIL | CFO={cfo:+.1f}Hz stf_peak={stf_peak:.4f}"
            save_try_figure(
                outp=outp, title=title,
                rx=rx_cfo, fs=args.fs,
                tone_freqs=tone_freqs, tone_mag=tone_mag, cfo=cfo, tone_peak_mag=tone_peak_mag,
                stf_corr=stf_corr, stf_idx=stf_idx, stf_peak=stf_peak,
                H=None,
                pilot_pow=np.array([], dtype=np.float32),
                phase_err=np.array([], dtype=np.float32),
                phase_acc=np.array([], dtype=np.float32),
                freq_acc=np.array([], dtype=np.float32),
                const_pre=np.array([], dtype=np.complex64),
                const_post=np.array([], dtype=np.complex64),
                evm_sym=np.array([], dtype=np.float32),
                bytes_preview=b"",
                parse_info="LTF channel estimate failed (buffer too short / wrong start)",
            )
            print(f"[{t+1:02d}] LTF failed saved {outp}")
            continue

        # 4) Demod payload with pilot loop (same control law, more logging)
        payload_start = ltf_start + 2 * SYMBOL_LEN

        phase_acc = 0.0
        freq_acc = 0.0

        phase_err_log = []
        phase_acc_log = []
        freq_acc_log = []
        pilot_pow_log = []
        evm_log = []

        const_pre = []
        const_post = []

        all_data_syms_post = []
        all_data_syms_pre = []

        for sym_idx in range(args.max_ofdm_syms):
            sym_start = payload_start + sym_idx * SYMBOL_LEN
            Y = extract_ofdm_symbol_fd(rx_cfo, sym_start)
            if Y is None:
                break

            # Equalize on used bins only (keep same as your code spirit)
            Y_eq = np.zeros_like(Y)
            eps = 1e-9
            for k in used_subs:
                idx = (k + N_FFT) % N_FFT
                if np.abs(H[idx]) > 1e-6:
                    Y_eq[idx] = Y[idx] / (H[idx] + eps)

            # Pilot expected
            pilot_sign = 1.0 if (sym_idx % 2 == 0) else -1.0
            expected_pilots = pilot_sign * pilot_pattern
            rx_pilots = Y_eq[pilot_idx]

            # pilot power
            pilot_pow = float(np.mean(np.abs(rx_pilots)**2))
            pilot_pow_log.append(pilot_pow)

            # phase error
            e = np.sum(rx_pilots * np.conj(expected_pilots))
            pe = float(np.angle(e + 1e-12))
            phase_err_log.append(pe)

            # PI loop
            freq_acc += args.ki * pe
            phase_acc += freq_acc + args.kp * pe

            phase_acc_log.append(float(phase_acc))
            freq_acc_log.append(float(freq_acc))

            # store pre/post constellation (data bins)
            d_pre = Y_eq[data_idx]
            all_data_syms_pre.append(d_pre)

            # apply CPE correction
            Y_eq2 = Y_eq * np.exp(-1j * phase_acc)
            d_post = Y_eq2[data_idx]
            all_data_syms_post.append(d_post)

            # rough EVM on this symbol
            ideal = nearest_qpsk(d_post)
            evm = float(np.sqrt(np.mean(np.abs(d_post - ideal)**2) / (np.mean(np.abs(ideal)**2) + 1e-12)))
            evm_log.append(evm)

        if len(all_data_syms_post) == 0:
            outp = os.path.join(args.output_dir, f"try{t+1:02d}_NO_PAYLOAD.png")
            title = f"try={t+1:02d} NO_PAYLOAD | CFO={cfo:+.1f}Hz stf_peak={stf_peak:.4f}"
            save_try_figure(
                outp=outp, title=title,
                rx=rx_cfo, fs=args.fs,
                tone_freqs=tone_freqs, tone_mag=tone_mag, cfo=cfo, tone_peak_mag=tone_peak_mag,
                stf_corr=stf_corr, stf_idx=stf_idx, stf_peak=stf_peak,
                H=H,
                pilot_pow=np.array(pilot_pow_log, dtype=np.float32),
                phase_err=np.array(phase_err_log, dtype=np.float32),
                phase_acc=np.array(phase_acc_log, dtype=np.float32),
                freq_acc=np.array(freq_acc_log, dtype=np.float32),
                const_pre=np.array([], dtype=np.complex64),
                const_post=np.array([], dtype=np.complex64),
                evm_sym=np.array(evm_log, dtype=np.float32),
                bytes_preview=b"",
                parse_info="No OFDM symbols extracted (payload window out of range)",
            )
            print(f"[{t+1:02d}] No OFDM symbols saved {outp}")
            continue

        all_data_syms_pre = np.concatenate(all_data_syms_pre).astype(np.complex64)
        all_data_syms_post = np.concatenate(all_data_syms_post).astype(np.complex64)

        # Demap bits (POST-CPE)
        bits_raw = qpsk_demap_gray_vec(all_data_syms_post)
        bits = majority_vote(bits_raw, args.repeat)
        bits_bytes = np.packbits(bits).tobytes()

        success, payload, plen, crc_rx, crc_calc = parse_packet(bits_bytes)

        # Prepare parse info text
        if bits_bytes[:4] == MAGIC:
            parse_info = f"MAGIC OK | plen={plen} | crc_rx=0x{crc_rx:08X} crc_calc=0x{crc_calc:08X}"
        else:
            parse_info = f"MAGIC NOT FOUND | first4={bits_bytes[:4].hex()} | CFO={cfo:+.1f}Hz STF={stf_peak:.4f}"

        # Save debug figure always
        status = "OK" if success else "FAIL"
        outp = os.path.join(args.output_dir, f"try{t+1:02d}_{status}.png")
        title = f"try={t+1:02d} {status} | CFO={cfo:+.1f}Hz STF={stf_peak:.4f} | {parse_info}"

        save_try_figure(
            outp=outp, title=title,
            rx=rx_cfo, fs=args.fs,
            tone_freqs=tone_freqs, tone_mag=tone_mag, cfo=cfo, tone_peak_mag=tone_peak_mag,
            stf_corr=stf_corr, stf_idx=stf_idx, stf_peak=stf_peak,
            H=H,
            pilot_pow=np.array(pilot_pow_log, dtype=np.float32),
            phase_err=np.array(phase_err_log, dtype=np.float32),
            phase_acc=np.array(phase_acc_log, dtype=np.float32),
            freq_acc=np.array(freq_acc_log, dtype=np.float32),
            const_pre=all_data_syms_pre,
            const_post=all_data_syms_post,
            evm_sym=np.array(evm_log, dtype=np.float32),
            bytes_preview=bits_bytes,
            parse_info=parse_info,
        )

        if success:
            with open(args.outfile, "wb") as f:
                f.write(payload)

            print(f"[{t+1:02d}] SUCCESS! Payload: {len(payload)} bytes, CRC: 0x{crc_calc:08X}")
            print(f"      CFO: {cfo:.1f} Hz, STF peak: {stf_peak:.4f}")
            print(f"      Saved to: {args.outfile}")
            print(f"      Debug fig: {outp}")

            # Also save a small text summary
            with open(os.path.join(args.output_dir, "success_summary.txt"), "w") as f:
                f.write(title + "\n")
                f.write(f"outfile={args.outfile}\n")
                f.write(f"payload_len={len(payload)}\n")
            break
        else:
            if bits_bytes[:4] == MAGIC:
                print(f"[{t+1:02d}] CRC FAIL: rx=0x{crc_rx:08X} calc=0x{crc_calc:08X} (saved {outp})")
            else:
                print(f"[{t+1:02d}] Magic not found (CFO={cfo:.1f}Hz, STF={stf_peak:.4f}) (saved {outp})")

    else:
        print("\nNo valid packet recovered after all attempts.")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


if __name__ == "__main__":
    main()

"""
python3 rf_step4_rx_opt_dbg.py \
  --uri "ip:192.168.2.2" \
  --fc 915e6 --fs 2e6 \
  --rx_gain 40 \
  --buf_size 131072 \
  --max_ofdm_syms 200 \
  --repeat 1 \
  --tries 10 \
  --outfile recovered_message.bin \
  --output_dir rf_link_step4_results_opt
"""