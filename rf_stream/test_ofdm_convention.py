#!/usr/bin/env python3
"""Test if TX (ifftshift+ifft) and RX (fftshift+fft) conventions are consistent."""
import numpy as np

N_FFT = 64; N_CP = 16; SYMBOL_LEN = N_FFT + N_CP

def sc_to_bin(k): return (k + N_FFT) % N_FFT
def sc_to_bin_fftshifted(k): return (k + N_FFT//2) % N_FFT

PILOT_SUBCARRIERS = [-21, -7, 7, 21]
DATA_SUBCARRIERS = [k for k in range(-26,27) if k != 0 and k not in PILOT_SUBCARRIERS]
IDEAL_QPSK = np.array([(1+1j),(1-1j),(-1+1j),(-1-1j)], dtype=np.complex64) / np.sqrt(2)
pilot_vals = np.array([1,1,1,-1], dtype=np.complex64)

# Build a TX symbol using sc_to_bin = (k+N_FFT)%N_FFT and ifftshift+ifft
X = np.zeros(N_FFT, dtype=np.complex64)
for i, k in enumerate(DATA_SUBCARRIERS): X[sc_to_bin(k)] = IDEAL_QPSK[0]
for i, k in enumerate(PILOT_SUBCARRIERS): X[sc_to_bin(k)] = pilot_vals[i]

x_td = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
sym_td = np.concatenate([x_td[-N_CP:], x_td])

td = sym_td[N_CP:]
# RX uses fftshift(fft())
Y_shifted = np.fft.fftshift(np.fft.fft(td))
# RX direct fft, no shift
Y_plain = np.fft.fft(td)

print("=== TX: ifft(ifftshift(X)), sc_to_bin=(k+N_FFT)%N_FFT ===")
print()
print("--- RX: fftshift(fft()), access with (k+N_FFT//2)%N_FFT ---")
for k in PILOT_SUBCARRIERS:
    tx_v = X[sc_to_bin(k)]
    rx_v = Y_shifted[sc_to_bin_fftshifted(k)] / np.sqrt(N_FFT)
    ok = abs(tx_v - rx_v) < 0.01
    print(f"  k={k:+3d}: TX={tx_v:.3f}  RX={rx_v:.3f}  {'OK' if ok else 'MISMATCH!'}")

print()
print("--- RX: fftshift(fft()), access with (k+N_FFT)%N_FFT [CURRENT RX CODE] ---")
for k in PILOT_SUBCARRIERS:
    tx_v = X[sc_to_bin(k)]
    rx_v = Y_shifted[sc_to_bin(k)] / np.sqrt(N_FFT)
    ok = abs(tx_v - rx_v) < 0.01
    print(f"  k={k:+3d}: TX={tx_v:.3f}  RX={rx_v:.3f}  {'OK' if ok else 'MISMATCH!'}")

print()
print("--- RX: plain fft(), access with (k+N_FFT)%N_FFT ---")
for k in PILOT_SUBCARRIERS:
    tx_v = X[sc_to_bin(k)]
    rx_v = Y_plain[sc_to_bin(k)] / np.sqrt(N_FFT)
    ok = abs(tx_v - rx_v) < 0.01
    print(f"  k={k:+3d}: TX={tx_v:.3f}  RX={rx_v:.3f}  {'OK' if ok else 'MISMATCH!'}")

print()
print("--- Conclusion ---")
# check which convention fully recovers
errs_fftshift_new = sum(abs(X[sc_to_bin(k)] - Y_shifted[sc_to_bin_fftshifted(k)]/np.sqrt(N_FFT)) for k in range(-26,27) if k != 0)
errs_fftshift_old = sum(abs(X[sc_to_bin(k)] - Y_shifted[sc_to_bin(k)]/np.sqrt(N_FFT)) for k in range(-26,27) if k != 0)
errs_plain       = sum(abs(X[sc_to_bin(k)] - Y_plain[sc_to_bin(k)]/np.sqrt(N_FFT)) for k in range(-26,27) if k != 0)
print(f"fftshift(fft)+fftshifted_bin total err: {errs_fftshift_new:.4f}")
print(f"fftshift(fft)+old_bin total err:        {errs_fftshift_old:.4f}")
print(f"plain fft+old_bin total err:            {errs_plain:.4f}")
