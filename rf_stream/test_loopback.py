#!/usr/bin/env python3
"""
Offline analysis tool: capture a raw RX window when NCC > 0.1 and analyze what bits we decode.
This tests if the decoding pipeline is correct at all.
"""
import numpy as np
import zlib

N_FFT = 64
N_CP = 16
SYMBOL_LEN = N_FFT + N_CP
PILOT_SUBCARRIERS = [-21, -7, 7, 21]
DATA_SUBCARRIERS = [k for k in range(-26, 27) if k != 0 and k not in PILOT_SUBCARRIERS]
BITS_PER_SYM = len(DATA_SUBCARRIERS) * 2  # 2 bits/subcarrier for QPSK
MAGIC = b'\xAD\xDE\xAD\xDE'

def sc_to_bin(k):
    return (k + N_FFT) % N_FFT

PILOT_BINS = np.array([sc_to_bin(k) for k in PILOT_SUBCARRIERS], dtype=int)
DATA_BINS = np.array([sc_to_bin(k) for k in DATA_SUBCARRIERS], dtype=int)
IDEAL_QPSK = np.array([(1+1j),(1-1j),(-1+1j),(-1-1j)], dtype=np.complex64) / np.sqrt(2)

def qpsk_demap(symbols):
    bits = np.zeros(symbols.size * 2, dtype=np.uint8)
    for i in range(symbols.size):
        r = symbols[i].real >= 0
        m = symbols[i].imag >= 0
        if r and m:   bits[2*i], bits[2*i+1] = 0, 0
        elif (not r) and m: bits[2*i], bits[2*i+1] = 0, 1
        elif (not r) and (not m): bits[2*i], bits[2*i+1] = 1, 1
        else:          bits[2*i], bits[2*i+1] = 1, 0
    return bits

def bits_to_bytes(bits):
    L = (len(bits)//8)*8
    if L <= 0: return b""
    return np.packbits(bits[:L]).tobytes()

# =============================================
# SIMULATE TX: build a known packet and encode it
# =============================================
def make_ref_payload(seed, n):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()

def build_packet_bytes(seq, total, payload):
    hdr = MAGIC + int(seq).to_bytes(2,"little") + int(total).to_bytes(2,"little") + int(len(payload)).to_bytes(2,"little")
    body = hdr + payload
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + int(crc).to_bytes(4,"little")

def bits_from_bytes(bb):
    return np.unpackbits(np.frombuffer(bb, dtype=np.uint8))

def qpsk_map(bits):
    # Gray mapping: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
    n = (len(bits) // 2)
    b0 = bits[0::2]
    b1 = bits[1::2]
    re = np.where((b0 == 0) & (b1 == 0),  1.0,
         np.where((b0 == 0) & (b1 == 1), -1.0,
         np.where((b0 == 1) & (b1 == 1), -1.0,  1.0)))
    im = np.where((b0 == 0) & (b1 == 0),  1.0,
         np.where((b0 == 0) & (b1 == 1),  1.0,
         np.where((b0 == 1) & (b1 == 1), -1.0, -1.0)))
    return (re + 1j*im).astype(np.complex64) / np.sqrt(2.0)

def create_ofdm_symbol(data_syms, pilot_vals, sym_idx):
    pilot_sign = 1 if (sym_idx % 2 == 0) else -1
    X = np.zeros(N_FFT, dtype=np.complex64)
    for i, k in enumerate(DATA_SUBCARRIERS):
        X[sc_to_bin(k)] = data_syms[i]
    PILOT_SUBS_LIST = [-21, -7, 7, 21]
    for i, k in enumerate(PILOT_SUBS_LIST):
        X[sc_to_bin(k)] = pilot_sign * pilot_vals[i]
    # REMOVED ifftshift
    x = np.fft.ifft(X) * np.sqrt(N_FFT)
    return np.concatenate([x[-N_CP:], x]).astype(np.complex64)

# Build and transmit a single packet, then decode it to verify
payload = make_ref_payload(1234, 64)[:32]  # 32 bytes payload
pkt_bytes = build_packet_bytes(0, 8, payload)
print(f"Packet: {len(pkt_bytes)} bytes")
print(f"MAGIC: {pkt_bytes[:4].hex()}")
print(f"First bytes: {pkt_bytes[:10].hex()}")

bits_tx = bits_from_bytes(pkt_bytes)
num_syms = int(np.ceil(len(bits_tx) / BITS_PER_SYM))
bits_padded = np.pad(bits_tx, (0, num_syms*BITS_PER_SYM - len(bits_tx)))

pilot_vals_arr = np.array([1,1,1,-1], dtype=np.complex64)
ofdm_syms = []
for si in range(num_syms):
    sb = bits_padded[si*BITS_PER_SYM : (si+1)*BITS_PER_SYM]
    ds = qpsk_map(sb)
    ofdm_syms.append(create_ofdm_symbol(ds, pilot_vals_arr, si))
ofdm = np.concatenate(ofdm_syms)

# =============================================
# SIMULATE RX: decode the OFDM signal  
# =============================================
# Create LTF channel estimate (using identity channel: H=1 everywhere)
stf_repeats = 6
rng2 = np.random.default_rng(42)
X_stf = np.zeros(N_FFT, dtype=np.complex64)
even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
bpsk = rng2.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
for i, k in enumerate(even_subs):
    X_stf[sc_to_bin(int(k))] = bpsk[i] + 0j
# REMOVED ifftshift
x_stf = np.fft.ifft(X_stf) * np.sqrt(N_FFT)
stf = np.tile(x_stf.astype(np.complex64), stf_repeats)
stf_cp = np.concatenate([stf[-N_CP:], stf])

# LTF
X_ltf = np.zeros(N_FFT, dtype=np.complex64)
used = [k for k in range(-26,27) if k != 0]
for i, k in enumerate(used):
    X_ltf[sc_to_bin(k)] = (1.0 if (i % 2 == 0) else -1.0) + 0j
# REMOVED ifftshift
x_ltf = np.fft.ifft(X_ltf) * np.sqrt(N_FFT)
ltf_sym = np.concatenate([x_ltf[-N_CP:], x_ltf])
ltf_sigs = np.tile(ltf_sym, 4)  # 4 LTF symbols

# Full TX signal: stf + ltf + ofdm
full_tx = np.concatenate([stf_cp, ltf_sigs, ofdm])

# Now receive and decode
stf_start = len(stf_cp)
ltf_start = stf_start
payload_start = ltf_start + 4 * SYMBOL_LEN

# Channel estimate from LTF
Ys = []
for i in range(4):
    start = ltf_start + i*SYMBOL_LEN
    td = full_tx[start + N_CP : start + SYMBOL_LEN]
    # REMOVED fftshift
    Y = np.fft.fft(td)
    Ys.append(Y)
Yavg = np.mean(np.stack(Ys), axis=0)
H = np.ones(N_FFT, dtype=np.complex64)
for k in used:
    idx = sc_to_bin(k)
    if abs(X_ltf[idx]) > 1e-6:
        H[idx] = Yavg[idx] / X_ltf[idx]

print(f"\nH[pilot bins]: {H[PILOT_BINS[:2]]}")

# Demodulate
data_syms_all = []
for si in range(num_syms):
    start = payload_start + si * SYMBOL_LEN
    td = full_tx[start + N_CP : start + SYMBOL_LEN]
    # REMOVED fftshift
    Y = np.fft.fft(td)
    Ye = Y.copy()
    for k in used:
        idx = sc_to_bin(k)
        if abs(H[idx]) > 1e-6:
            Ye[idx] = Ye[idx] / H[idx]
    ds = Ye[DATA_BINS]
    data_syms_all.append(ds)

data_syms_all = np.concatenate(data_syms_all)
bits_rx = qpsk_demap(data_syms_all)
bb_rx = bits_to_bytes(bits_rx)

print(f"\nTX packet first 10 bytes: {pkt_bytes[:10].hex()}")
print(f"RX decoded first 10 bytes: {bb_rx[:10].hex()}")
print(f"Match: {bb_rx[:len(pkt_bytes)] == pkt_bytes}")
if bb_rx[:4] == MAGIC:
    print("MAGIC OK")
else:
    print(f"MAGIC FAIL: {bb_rx[:4].hex()} vs {MAGIC.hex()}")
