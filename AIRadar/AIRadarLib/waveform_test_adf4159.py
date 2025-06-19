import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
from AIRadarLib.waveform_utils import generate_sine_wave, generate_ofdm_signal, generate_adf4159_fmcw_chirp
from AIRadarLib.visualization import plot_signal_comparison

# Try to import SIONNA for OFDM generation
SIONNA_AVAILABLE = False
use_sionna = False
if use_sionna:
    try:
        import tensorflow as tf
        import sionna
        from sionna.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator
        from sionna.mapping import Mapper, QPSK
        SIONNA_AVAILABLE = True
        print("Using NVIDIA SIONNA for OFDM signal generation")
    except ImportError:
        print("NVIDIA SIONNA not found. Will use basic OFDM implementation.")
        SIONNA_AVAILABLE = False

def test_adf4159():
    # Parameters
    num_chirps = 4
    total_samples_per_chirp = 1024
    active_samples = 800
    sample_rate = 10e6  # 10 MHz
    start_freq = 77e9  # 77 GHz
    bandwidth = 200e6  # 200 MHz

    # Generate signal
    continuous_signal, chirp_duration = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=sample_rate,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,
        freq_deviation_ppm=2,
        pll_ref_freq=100e6,
        pll_ref_doubler=False,
        pll_ref_div_factor=1,
        window_type='edge',
    )#(4096,), 8e-05

    # Time vector
    t = np.arange(len(continuous_signal)) / sample_rate

    # Spectrogram
    f, t_spec, Sxx = spectrogram(
        continuous_signal,
        fs=sample_rate,
        nperseg=256,
        noverlap=128,
        return_onesided=False,
        mode='magnitude'
    )

    # Instantaneous frequency
    instantaneous_phase = np.unwrap(np.angle(continuous_signal))
    instantaneous_freq = np.diff(instantaneous_phase) * sample_rate / (2 * np.pi)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    axs[0].plot(t * 1e6, np.real(continuous_signal), label='Real')
    axs[0].plot(t * 1e6, np.imag(continuous_signal), label='Imag', alpha=0.6)
    axs[0].set_title('Time-Domain Signal')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    pcm = axs[1].pcolormesh(t_spec * 1e6, f / 1e6, 20 * np.log10(Sxx + 1e-12), shading='gouraud')
    axs[1].set_title('Spectrogram')
    axs[1].set_ylabel('Frequency (MHz)')
    axs[1].set_xlabel('Time (µs)')
    fig.colorbar(pcm, ax=axs[1], label='Magnitude (dB)')

    axs[2].plot(t[1:] * 1e6, instantaneous_freq / 1e6)
    axs[2].set_title('Instantaneous Frequency')
    axs[2].set_ylabel('Frequency (MHz)')
    axs[2].set_xlabel('Time (µs)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('data/adf4159_test.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # # === Plot zoomed-in instantaneous frequency for one chirp ===
    # chirp_idx = 0
    # start = chirp_idx * total_samples_per_chirp
    # end = start + active_samples
    # t_zoom = np.arange(start + 1, end) / sample_rate * 1e6  # µs

    # plt.figure(figsize=(12, 5))
    # #plt.plot(t_zoom, instantaneous_freq[start + 1:end] / 1e6, label='Instantaneous Frequency')
    # plt.plot(t_zoom, (instantaneous_freq[start + 1:end] - start_freq) / 1e6, label='Baseband Frequency (Measured)')

    # # Annotate chirp boundaries
    # plt.axvline(t_zoom[0], color='gray', linestyle='--', linewidth=1, label='Chirp Start')
    # plt.axvline(t_zoom[-1], color='gray', linestyle='--', linewidth=1, label='Chirp End')

    # # Ideal ramp reference
    # ideal_slope = bandwidth / chirp_duration / 1e6  # MHz/sec
    # #ideal_freq_line = start_freq / 1e6 + ideal_slope * (t_zoom - t_zoom[0]) * 1e-6
    # ideal_freq_line = ideal_slope * (t_zoom - t_zoom[0]) * 1e-6 # in MHz, baseband
    # plt.plot(t_zoom, ideal_freq_line, 'r--', label='Ideal Ramp')

    # plt.title("Zoomed-In Instantaneous Frequency of One FMCW Chirp")
    # plt.xlabel("Time (µs)")
    # plt.ylabel("Frequency (MHz)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('data/adf4159_test_zoom.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    # === Zoom-In: Instantaneous Baseband Frequency ===
    chirp_idx = 0
    start = chirp_idx * total_samples_per_chirp
    end = start + active_samples

    # Time vector for this chirp
    t_chirp = np.arange(start + 1, end) / sample_rate  # in seconds
    t_chirp_us = t_chirp * 1e6  # in microseconds

    # Instantaneous frequency (Hz)
    instantaneous_phase = np.unwrap(np.angle(continuous_signal))
    instantaneous_freq = np.diff(instantaneous_phase) * sample_rate / (2 * np.pi)

    # Basebanded frequency
    baseband_freq = instantaneous_freq[start + 1:end] - start_freq

    # Ideal baseband ramp
    slope = bandwidth / chirp_duration  # Hz/s
    ideal_baseband_freq = slope * (t_chirp - t_chirp[0])  # Hz

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(t_chirp_us, baseband_freq / 1e6, label='Measured Baseband Frequency')
    plt.plot(t_chirp_us, ideal_baseband_freq / 1e6, 'r--', label='Ideal Baseband Ramp')
    plt.axvline(t_chirp_us[0], color='gray', linestyle='--', linewidth=1, label='Chirp Start')
    plt.axvline(t_chirp_us[-1], color='gray', linestyle='--', linewidth=1, label='Chirp End')

    plt.title("Zoomed-In Instantaneous Baseband Frequency of One FMCW Chirp")
    plt.xlabel("Time (µs)")
    plt.ylabel("Frequency (MHz)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('data/adf4159_test_zoom.pdf', dpi=300, bbox_inches='tight')

from scipy.signal import spectrogram, get_window, welch
def plot_adf4159():
    # === Simulation Parameters ===
    fs = 1e9            # 1 GHz sampling rate
    duration = 10e-6    # 10 microseconds
    N = int(fs * duration)
    start_freq = 0
    bandwidth = 100e6
    num_chirps = 1

    # === Generate chirp using ADF4159 model ===
    chirp_signal, chirp_duration = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=N,
        active_samples=N,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,
        pll_ref_freq=100e6,
        tx_power=1.0
    )

    # === Sine Wave Input (Tone) ===
    tone_freq = 20e6  # 20 MHz
    t = np.arange(N) / fs
    sine_wave = np.exp(1j * 2 * np.pi * tone_freq * t)

    # === Deramp: Mix chirp with conjugate of sine wave ===
    mixed_signal = chirp_signal * np.conj(sine_wave)

    # === Windowing for PSD ===
    window = get_window('hann', N)
    windowed_mixed = mixed_signal * window

    # === PSD ===
    f_raw, Pxx_raw = welch(sine_wave, fs=fs, nperseg=2048, return_onesided=False)
    f_mixed, Pxx_mixed = welch(windowed_mixed, fs=fs, nperseg=2048, return_onesided=False)
    f_raw = np.fft.fftshift(f_raw)
    Pxx_raw = np.fft.fftshift(Pxx_raw)
    f_mixed = np.fft.fftshift(f_mixed)
    Pxx_mixed = np.fft.fftshift(Pxx_mixed)

    # === Spectrogram ===
    f_spec, t_spec, Sxx_spec = spectrogram(mixed_signal, fs=fs, nperseg=256, noverlap=128, return_onesided=False)

    # === Plotting ===
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ADF4159 Chirp Mixed with Sine Tone (Deramped)", fontsize=16)

    # Time domain
    axs[0, 0].plot(t * 1e6, np.real(mixed_signal), label='Real')
    axs[0, 0].plot(t * 1e6, np.imag(mixed_signal), label='Imag', alpha=0.6)
    axs[0, 0].set_title("Mixed Signal (Time Domain)")
    axs[0, 0].set_xlabel("Time (µs)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # PSD of mixed signal
    axs[0, 1].plot(f_mixed / 1e6, 10 * np.log10(Pxx_mixed + 1e-12))
    axs[0, 1].set_title("Mixed Signal PSD with Hann Window")
    axs[0, 1].set_xlabel("Frequency (MHz)")
    axs[0, 1].set_ylabel("Power Spectral Density (dB/Hz)")
    axs[0, 1].grid(True)

    # Original sine wave PSD
    axs[1, 0].plot(f_raw / 1e6, 10 * np.log10(Pxx_raw + 1e-12))
    axs[1, 0].set_title("Original Sine Wave PSD")
    axs[1, 0].set_xlabel("Frequency (MHz)")
    axs[1, 0].set_ylabel("Power Spectral Density (dB/Hz)")
    axs[1, 0].grid(True)

    # Spectrogram
    pcm = axs[1, 1].pcolormesh(t_spec * 1e6, f_spec / 1e6, 10 * np.log10(Sxx_spec + 1e-12), shading='gouraud')
    axs[1, 1].set_title("Spectrogram of Mixed Signal")
    axs[1, 1].set_xlabel("Time (µs)")
    axs[1, 1].set_ylabel("Frequency (MHz)")
    fig.colorbar(pcm, ax=axs[1, 1], label="Power (dB)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('data/adf4159_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def test_adf4159_with_sine_wave():
    """
    Test the ADF4159 FMCW chirp generator with a sine wave input
    """
    print("Testing ADF4159 FMCW chirp generator with sine wave input...")
    
    # Parameters
    fs = 100e6  # 100 MHz sampling rate
    sine_freq = 10e6  # 10 MHz sine wave
    duration = 0.0001  # 100 μs
    num_samples = int(duration * fs) #10000
    
    # Generate sine wave
    sine_wave = generate_sine_wave(sine_freq, duration, fs) #(10000,)
    
    # Parameters for ADF4159 simulation
    num_chirps = 1
    total_samples_per_chirp = num_samples
    active_samples = num_samples
    start_freq = 2.4e9  # 2.4 GHz
    bandwidth = 100e6  # 100 MHz
    
    # Process the sine wave through the ADF4159 simulation
    # We'll use the sine wave as the envelope of the chirp
    processed_signal, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,  # -85 dBc/Hz phase noise
        pll_ref_freq=100e6,     # 100 MHz reference
        tx_power=1.0
    )
    
    # Modulate the processed signal with the sine wave for comparison
    modulated_signal = processed_signal * sine_wave / np.max(np.abs(sine_wave))
    
    # Plot the results
    fig = plot_signal_comparison(sine_wave, modulated_signal, fs, 
                               "ADF4159 FMCW Chirp Generator with Sine Wave Input")
    plt.savefig('data/adf4159_sine_wave_test.pdf', dpi=300, bbox_inches='tight')
    plt.show()


def test_adf4159_with_ofdm(use_sionna=False):
    """
    Test the ADF4159 FMCW chirp generator with an OFDM signal input
    
    Args:
        use_sionna: Whether to use NVIDIA SIONNA for OFDM generation (if available)
    """
    
    ofdm_type = "SIONNA" if use_sionna and SIONNA_AVAILABLE else "Basic"
    print(f"Testing ADF4159 FMCW chirp generator with {ofdm_type} OFDM signal input...")
    
    # Parameters
    fs = 100e6  # 100 MHz sampling rate
    num_subcarriers = 64
    num_symbols = 10
    subcarrier_spacing = 15e3  # 15 kHz (similar to LTE)
    
    # Generate OFDM signal
    if use_sionna and SIONNA_AVAILABLE:
        # Generate OFDM signal using SIONNA
        # Calculate FFT size based on sampling rate and subcarrier spacing
        fft_size = int(fs / subcarrier_spacing)
        
        # Create a resource grid configuration
        # We'll use a single stream, single user configuration
        rg = ResourceGrid(num_ofdm_symbols=num_symbols,
                          fft_size=fft_size,
                          subcarrier_spacing=subcarrier_spacing,
                          num_tx=1,  # Single transmitter
                          num_streams_per_tx=1,  # Single stream
                          cyclic_prefix_length=int(fft_size * 0.25),
                          pilot_pattern=None,  # No pilots for simplicity
                          pilot_ofdm_symbol_indices=[])
        
        # Create a batch of random bits (batch_size=1)
        num_bits_per_symbol = 2  # QPSK
        num_bits = num_bits_per_symbol * rg.num_data_symbols
        bits = tf.random.uniform([1, num_bits], 0, 2, tf.int32)
        
        # Map bits to constellation symbols (QPSK)
        mapper = Mapper(QPSK())
        x = mapper(bits)
        
        # Map constellation symbols to resource grid
        rg_mapper = ResourceGridMapper(rg)
        x_rg = rg_mapper(x)
        
        # OFDM modulation
        modulator = OFDMModulator(rg)
        x_time = modulator(x_rg)
        
        # Convert to numpy and reshape to 1D array
        ofdm_signal = x_time.numpy().squeeze()
        
        # Normalize the signal
        ofdm_signal = ofdm_signal / np.max(np.abs(ofdm_signal))
    else:
        # Use the basic OFDM implementation
        ofdm_signal = generate_ofdm_signal(num_subcarriers, num_symbols, subcarrier_spacing, fs)
    
    # Parameters for ADF4159 simulation
    num_chirps = 1
    total_samples_per_chirp = len(ofdm_signal)
    active_samples = len(ofdm_signal)
    start_freq = 2.4e9  # 2.4 GHz
    bandwidth = 100e6  # 100 MHz
    
    # Process the OFDM signal through the ADF4159 simulation
    processed_signal, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,  # -85 dBc/Hz phase noise
        pll_ref_freq=100e6,     # 100 MHz reference
        tx_power=1.0
    )
    
    # Modulate the processed signal with the OFDM signal for comparison
    modulated_signal = processed_signal * ofdm_signal / np.max(np.abs(ofdm_signal))
    
    # Plot the results
    fig = plot_signal_comparison(ofdm_signal, modulated_signal, fs, 
                               f"ADF4159 FMCW Chirp Generator with {ofdm_type} OFDM Signal Input")
    plt.savefig(f'adf4159_{ofdm_type.lower()}_ofdm_test.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_ofdm_implementations():
    """
    Compare basic OFDM implementation with SIONNA OFDM implementation
    """
    # Try to import SIONNA
    try:
        import tensorflow as tf
        import sionna
        from sionna.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator
        from sionna.mapping import Mapper, QPSK
        SIONNA_AVAILABLE = True
    except ImportError:
        print("SIONNA not available, skipping comparison")
        return
    
    print("Comparing basic OFDM implementation with SIONNA OFDM implementation...")
    
    # Parameters
    fs = 100e6  # 100 MHz sampling rate
    num_subcarriers = 64
    num_symbols = 10
    subcarrier_spacing = 15e3  # 15 kHz
    
    # Generate basic OFDM signal
    basic_ofdm = generate_ofdm_signal(num_subcarriers, num_symbols, subcarrier_spacing, fs)
    
    # Generate SIONNA OFDM signal
    # Calculate FFT size based on sampling rate and subcarrier spacing
    fft_size = int(fs / subcarrier_spacing)
    
    # Create a resource grid configuration
    rg = ResourceGrid(num_ofdm_symbols=num_symbols,
                      fft_size=fft_size,
                      subcarrier_spacing=subcarrier_spacing,
                      num_tx=1,  # Single transmitter
                      num_streams_per_tx=1,  # Single stream
                      cyclic_prefix_length=int(fft_size * 0.25),
                      pilot_pattern=None,  # No pilots for simplicity
                      pilot_ofdm_symbol_indices=[])
    
    # Create a batch of random bits (batch_size=1)
    num_bits_per_symbol = 2  # QPSK
    num_bits = num_bits_per_symbol * rg.num_data_symbols
    bits = tf.random.uniform([1, num_bits], 0, 2, tf.int32)
    
    # Map bits to constellation symbols (QPSK)
    mapper = Mapper(QPSK())
    x = mapper(bits)
    
    # Map constellation symbols to resource grid
    rg_mapper = ResourceGridMapper(rg)
    x_rg = rg_mapper(x)
    
    # OFDM modulation
    modulator = OFDMModulator(rg)
    x_time = modulator(x_rg)
    
    # Convert to numpy and reshape to 1D array
    sionna_ofdm = x_time.numpy().squeeze()
    
    # Normalize the signal
    sionna_ofdm = sionna_ofdm / np.max(np.abs(sionna_ofdm))
    
    # Plot comparison
    fig = plot_signal_comparison(basic_ofdm, sionna_ofdm, fs, 
                               "Comparison of Basic vs SIONNA OFDM Implementation")
    plt.savefig('ofdm_implementation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_adf4159_parameter_effects():
    """
    Test the effects of different ADF4159 parameters on the output signal
    """
    print("Testing effects of different ADF4159 parameters...")
    
    # Parameters
    fs = 100e6  # 100 MHz sampling rate
    duration = 0.0001  # 100 μs
    num_samples = int(duration * fs)
    
    # Parameters for ADF4159 simulation
    num_chirps = 1
    total_samples_per_chirp = num_samples
    active_samples = num_samples
    start_freq = 2.4e9  # 2.4 GHz
    bandwidth = 100e6  # 100 MHz
    
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Effects of ADF4159 Parameters on Output Signal", fontsize=16)
    
    # 1. Default parameters
    signal_default, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,  # -85 dBc/Hz phase noise
        pll_ref_freq=100e6,     # 100 MHz reference
        tx_power=1.0
    )
    
    # 2. High phase noise
    signal_high_noise, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-65,  # -65 dBc/Hz phase noise (worse)
        pll_ref_freq=100e6,
        tx_power=1.0
    )
    
    # 3. Different reference frequency
    signal_diff_ref, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,
        pll_ref_freq=10e6,      # 10 MHz reference (worse frequency resolution)
        tx_power=1.0
    )
    
    # 4. High frequency deviation
    signal_high_dev, _ = generate_adf4159_fmcw_chirp(
        num_chirps=num_chirps,
        total_samples_per_chirp=total_samples_per_chirp,
        active_samples=active_samples,
        sample_rate=fs,
        start_freq=start_freq,
        bandwidth=bandwidth,
        phase_noise_level=-85,
        pll_ref_freq=100e6,
        freq_deviation_ppm=10,  # 10 ppm frequency deviation (worse)
        tx_power=1.0
    )
    
    # Plot frequency domain for all signals
    signals = [signal_default, signal_high_noise, signal_diff_ref, signal_high_dev]
    titles = ["Default Parameters", "High Phase Noise (-65 dBc/Hz)", 
              "Low Reference Frequency (10 MHz)", "High Frequency Deviation (10 ppm)"]
    
    for i, (signal_item, title) in enumerate(zip(signals, titles)):
        row, col = i // 2, i % 2
        
        # Calculate PSD
        f, Pxx = signal.welch(signal_item, fs, nperseg=1024, return_onesided=False)
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
        Pxx_db = 10 * np.log10(Pxx + 1e-10)
        
        # Plot
        axs[row, col].plot(f, Pxx_db)
        axs[row, col].set_title(title)
        axs[row, col].set_xlabel('Frequency (Hz)')
        axs[row, col].set_ylabel('Power Spectral Density (dB/Hz)')
        axs[row, col].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('adf4159_parameter_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_adf4159()
    plot_adf4159()
    # Test with sine wave
    test_adf4159_with_sine_wave()
    
    # Test with OFDM signal
    #test_adf4159_with_ofdm()
    
    # Test parameter effects
    test_adf4159_parameter_effects()