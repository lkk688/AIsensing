#!/usr/bin/env python3
import argparse
import zlib
import numpy as np
import adi

def make_subcarrier_plan(N: int):
    if N != 64:
        raise ValueError("This reference implementation is written for N=64.")
    pilots = np.array([-21, -7, 7, 21], dtype=int)
    used = np.r_[np.arange(-26, 0), np.arange(1, 27)]
    data = np.array([k for k in used if k not in set(pilots)], dtype=int)
    return data, pilots, used

def pilot_pattern(sym_idx: int, pilots_len: int):
    base = np.array([1, 1, 1, -1], dtype=np.float32)
    if pilots_len != 4:
        base = np.ones(pilots_len, dtype=np.float32)
    if sym_idx % 2 == 1:
        base = -base
    return base + 0j

def qpsk_demap_gray(syms: np.ndarray) -> np.ndarray:
    # Hard decisions for Gray mapping used in TX.
    # Decide quadrant: re,im signs then map back to bits.
    bits = np.zeros((len(syms), 2), dtype=np.uint8)
    re = np.real(syms) >= 0
    im = np.imag(syms) >= 0
    # quadrant -> bits:
    # + + : 00
    # - + : 01
    # - - : 11
    # + - : 10
    for i in range(len(syms)):
        if re[i] and im[i]:
            bits[i] = [0, 0]
        elif (not re[i]) and im[i]:
            bits[i] = [0, 1]
        elif (not re[i]) and (not im[i]):
            bits[i] = [1, 1]
        else:
            bits[i] = [1, 0]
    return bits.reshape(-1)

def majority_vote_repetition(bits_rep: np.ndarray, rep: int) -> np.ndarray:
    if rep == 1:
        return bits_rep.astype(np.uint8)
    L = (len(bits_rep) // rep) * rep
    bits_rep = bits_rep[:L].reshape(-1, rep)
    return (np.sum(bits_rep, axis=1) >= (rep/2)).astype(np.uint8)

def schmidl_cox_detect(rx: np.ndarray, N: int, CP: int):
    """
    Find start of STF which is 2 identical OFDM symbols (each length N+CP).
    We ignore CP in correlation and correlate N samples vs next N samples.
    Returns (best_index, cfo_hz, metric_peak)
    """
    # Search window: avoid first transient region if needed
    L = len(rx)
    if L < 4*N:
        return None, None, None

    # Use sliding correlation P and power R.
    # P(d) = sum_{n=0..N-1} r[d+n] * conj(r[d+n+N])
    # CFO = angle(P) * fs / (2*pi*N)
    P = np.zeros(L - 2*N, dtype=np.complex64)
    R = np.zeros(L - 2*N, dtype=np.float32)

    # Vectorized (fast enough for typical buffers)
    a = rx[:L-2*N]
    b = rx[N:L-N]
    c = rx[2*N:L]
    # We'll compute P over blocks using convolution-like sum; simple loop is ok for clarity.
    # To keep it robust, do a stepped scan and refine.
    step = 4
    best_m = -1
    best_d = None
    best_P = None

    for d in range(0, L - 2*N - 1, step):
        seg1 = rx[d:d+N]
        seg2 = rx[d+N:d+2*N]
        Pd = np.vdot(seg2, seg1)  # sum(seg1*conj(seg2)) but vdot conjugates first arg => conj(seg2)*seg1
        Rd = np.sum(np.abs(seg2)**2) + 1e-12
        m = (np.abs(Pd)**2) / (Rd**2)
        if m > best_m:
            best_m = m
            best_d = d
            best_P = Pd

    if best_d is None:
        return None, None, None

    # Refine around best_d
    refine = range(max(0, best_d - 2*step), min(L - 2*N - 1, best_d + 2*step + 1))
    best_m2 = -1
    best_d2 = best_d
    best_P2 = best_P
    for d in refine:
        seg1 = rx[d:d+N]
        seg2 = rx[d+N:d+2*N]
        Pd = np.vdot(seg2, seg1)
        Rd = np.sum(np.abs(seg2)**2) + 1e-12
        m = (np.abs(Pd)**2) / (Rd**2)
        if m > best_m2:
            best_m2 = m
            best_d2 = d
            best_P2 = Pd

    # CFO from phase of P
    return best_d2, best_P2, best_m2

def extract_ofdm_symbol(rx: np.ndarray, start: int, N: int, CP: int):
    s = rx[start + CP : start + CP + N]
    if len(s) != N:
        return None
    return np.fft.fftshift(np.fft.fft(s))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help='RX Pluto uri, e.g. "usb:1.37.5"')
    ap.add_argument("--fc", type=float, default=2.3e9)
    ap.add_argument("--fs", type=float, default=1e6)
    ap.add_argument("--bw", type=float, default=1.2e6)
    ap.add_argument("--rx_gain", type=float, default=55.0, help="dB, manual")
    ap.add_argument("--buf", type=int, default=2**20, help="RX buffer size (samples)")
    ap.add_argument("--repeat", type=int, default=4, choices=[1,2,4], help="bit repetition used by TX")
    ap.add_argument("--num_syms", type=int, default=300, help="payload OFDM symbols per packet (must match TX)")
    ap.add_argument("--outfile", default="recovered.bin")
    ap.add_argument("--expect_len", type=int, default=0, help="optional expected payload length (bytes) before CRC")
    ap.add_argument("--tries", type=int, default=50, help="how many capture attempts")
    args = ap.parse_args()

    N, CP = 64, 16
    data_bins, pilot_bins, used_bins = make_subcarrier_plan(N)

    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.bw)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    sdr.rx_buffer_size = int(args.buf)

    # Flush old buffers
    for _ in range(3):
        _ = sdr.rx()

    print("RX running")
    print(f"  uri={args.uri} fc={args.fc/1e6:.3f} MHz fs={args.fs/1e6:.3f} Msps bw={args.bw/1e6:.3f} MHz gain={args.rx_gain} dB")
    print(f"  N={N} CP={CP} data_sc={len(data_bins)} pilots={len(pilot_bins)} repeat={args.repeat} payload_syms={args.num_syms}")

    # Known LTF in frequency (must match TX build_preamble)
    Xltf = np.zeros(N, dtype=np.complex64)
    ltf_bpsk = np.ones(len(used_bins), dtype=np.float32)
    ltf_bpsk[::2] = -1.0
    Xltf[(used_bins + N//2) % N] = ltf_bpsk + 0j

    packet_found = False

    for t in range(args.tries):
        rx_raw = sdr.rx().astype(np.complex64) / (2**14)

        # DC removal helps Pluto LO leakage a lot
        rx = rx_raw - np.mean(rx_raw)

        # Schmidl-Cox detect STF (two identical OFDM symbols)
        d, Pd, m = schmidl_cox_detect(rx, N, CP)
        if d is None or m < 0.05:
            print(f"[{t+1:02d}] no lock (metric={m})")
            continue

        # CFO estimate (coarse)
        cfo_hz = (np.angle(Pd) * args.fs) / (2*np.pi*N)
        n = np.arange(len(rx), dtype=np.float32)
        rx_cfo = rx * np.exp(-1j * 2*np.pi * cfo_hz * n / args.fs)

        # Packet layout (must match TX):
        # zeros gap then STF(2 syms) then LTF(2 syms) then payload(num_syms)
        # Our d points to start of repeated region WITHOUT CP alignment guarantee,
        # but it is typically near the symbol boundary. We move forward to the next CP boundary.
        # Heuristic: align to CP start by assuming STF symbol starts at d, then STF has (N+CP) length.
        # Because our detector used N-only correlation, d is approximately at STF symbol start (not CP start).
        stf_start = d
        # We want CP start; shift back by CP if possible:
        stf_cp_start = max(0, stf_start - CP)

        # Extract LTF symbols (after STF)
        ltf0_start = stf_cp_start + 2*(N+CP)
        ltf1_start = ltf0_start + (N+CP)

        Y0 = extract_ofdm_symbol(rx_cfo, ltf0_start, N, CP)
        Y1 = extract_ofdm_symbol(rx_cfo, ltf1_start, N, CP)
        if Y0 is None or Y1 is None:
            print(f"[{t+1:02d}] lock but insufficient samples")
            continue

        Yltf = 0.5*(Y0 + Y1)
        # Channel estimate on used bins
        H = np.ones(N, dtype=np.complex64)
        eps = 1e-9
        used_idx = (used_bins + N//2) % N
        H[used_idx] = Yltf[used_idx] / (Xltf[used_idx] + eps)

        # Payload start
        pay_start = ltf1_start + (N+CP)

        bits_hat = []
        ph = 0.0  # optional phase accumulator (simple)
        alpha = 0.15  # PLL gain for pilot CPE

        for s in range(args.num_syms):
            sym_start = pay_start + s*(N+CP)
            Y = extract_ofdm_symbol(rx_cfo, sym_start, N, CP)
            if Y is None:
                break

            # Equalize
            Ye = np.zeros_like(Y)
            Ye[used_idx] = Y[used_idx] / (H[used_idx] + eps)

            # Residual common phase error from pilots
            pidx = (pilot_bins + N//2) % N
            pilots_rx = Ye[pidx]
            pilots_ref = pilot_pattern(s, len(pilot_bins))
            cpe = np.angle(np.vdot(pilots_ref, pilots_rx))  # angle(sum(conj(ref)*rx))
            ph = (1 - alpha)*ph + alpha*cpe
            Ye[used_idx] *= np.exp(-1j * ph)

            # Extract data
            didx = (data_bins + N//2) % N
            data_syms = Ye[didx]
            bits_sym = qpsk_demap_gray(data_syms)
            bits_hat.append(bits_sym)

        if len(bits_hat) == 0:
            print(f"[{t+1:02d}] lock but no payload symbols decoded")
            continue

        bits_hat = np.concatenate(bits_hat).astype(np.uint8)
        # Undo repetition
        bits_dec = majority_vote_repetition(bits_hat, args.repeat)

        # Convert to bytes and check CRC
        # We do not know exact length unless user provides expect_len.
        bb = np.packbits(bits_dec).tobytes()

        # If expect_len provided: payload is expect_len + 4 CRC
        if args.expect_len > 0:
            total = args.expect_len + 4
            bb = bb[:total]
        else:
            # Otherwise, attempt CRC check over multiple candidate truncations
            # by scanning plausible sizes. This is best-effort.
            pass

        if len(bb) < 5:
            print(f"[{t+1:02d}] too short after decode")
            continue

        payload = bb[:-4]
        crc_rx = int.from_bytes(bb[-4:], "little")
        crc_calc = zlib.crc32(payload) & 0xFFFFFFFF

        if crc_calc == crc_rx:
            open(args.outfile, "wb").write(payload)
            print(f"[{t+1:02d}] ✅ CRC OK! wrote {len(payload)} bytes -> {args.outfile}")
            packet_found = True
            break
        else:
            print(f"[{t+1:02d}] CRC fail (cfo={cfo_hz:+.1f} Hz metric={m:.3f}) calc={crc_calc:08x} rx={crc_rx:08x}")

    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    if not packet_found:
        print("No valid packet recovered. Increase SNR, adjust gains, or increase repetition / num_syms.")

if __name__ == "__main__":
    main()