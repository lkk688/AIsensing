import numpy as np
import adi
import matplotlib.pyplot as plt

def run_internal_diagnostics():
    sdr = adi.Pluto("ip:192.168.3.2")
    sdr.sample_rate = int(1e6)
    
    # 1. ENABLE DIGITAL LOOPBACK (Isolates the code from RF hardware)
    sdr._ctrl.debug_attrs['loopback'].value = '1' 
    
    # 2. Build a high-contrast test frame
    N, CP = 64, 16
    tx_bits = np.random.randint(0, 2, 48 * 2)
    qpsk = ((1 - 2*tx_bits[::2]) + 1j*(1 - 2*tx_bits[1::2])) / np.sqrt(2)
    X = np.zeros(N, dtype=complex)
    X[1:25] = qpsk[:24]; X[33:57] = qpsk[24:] # Map to subcarriers
    x_t = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    tx_frame = np.concatenate([x_t[-CP:], x_t])
    
    sdr.tx_cyclic_buffer = True
    sdr.tx((tx_frame * 2**14).astype(np.complex64))
    
    rx = sdr.rx()
    sdr.tx_destroy_buffer()

    # --- INTERNAL STATE FIGURES ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # A. The "Golden" Constellation (Should be 4 perfect dots)
    rx_sym = rx[CP:CP+N] # Perfect timing assumed in digital loopback
    Y = np.fft.fftshift(np.fft.fft(rx_sym))
    axes[0,0].scatter(Y.real, Y.imag, color='gold', s=10)
    axes[0,0].set_title("Digital Loopback Constellation (Math Verification)")
    
    # B. Time Domain Pulse (Check for bit-exactness)
    axes[0,1].plot(rx.real[:100], label='Real')
    axes[0,1].plot(tx_frame.real[:100], '--', label='Ref')
    axes[0,1].set_title("Time Domain Alignment (TX vs RX)")
    
    # C. Phase Error (Digital Noise Floor)
    phase_err = np.angle(Y[1:25] / X[1:25])
    axes[1,0].plot(phase_err)
    axes[1,0].set_title("Phase Error (Should be 0.0 rad)")
    
    # D. Internal IQ Imbalance Check
    axes[1,1].hist2d(rx.real, rx.imag, bins=50)
    axes[1,1].set_title("Internal IQ Density Map")

    plt.tight_layout()
    plt.savefig("internal_loopback_state.png")
    print("Internal diagnostic saved. If this isn't perfect, the FFT logic is the culprit.")

if __name__ == "__main__":
    run_internal_diagnostics()