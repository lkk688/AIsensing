import numpy as np
import random

def generate_fmcw_chirp(chirp_idx, chirp_duration, samples_per_chirp, bandwidth, center_freq):
    """Generate a single FMCW chirp signal with phase continuity"""
    t = np.linspace(0, chirp_duration, samples_per_chirp)
    
    # Calculate phase with proper phase continuity between chirps
    freq_sweep = bandwidth/chirp_duration * t
    phase_accumulation = 2 * np.pi * chirp_idx * bandwidth * chirp_duration
    phase = 2 * np.pi * (center_freq * t + 0.5 * freq_sweep * t) + phase_accumulation
    
    return np.exp(1j * phase)

def add_realistic_effects(rx_signal, tx_signal, radar_params):
    """
    Add realistic effects to the received signal including:

    - Direct coupling (TX leakage)
    - Environmental clutter
    - Crosstalk
    - Ground clutter
    - System noise
    
    Args:
        rx_signal: Received signal with shape [num_rx, num_chirps, samples_per_chirp]
        tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp]
        radar_params: Dictionary containing radar parameters
        
    Returns:
        Modified received signal with realistic effects
    """
    # Extract parameters from radar_params
    num_rx = radar_params['num_rx']
    num_chirps = radar_params['num_chirps']
    samples_per_chirp = radar_params['samples_per_chirp']
    chirp_duration = radar_params['chirp_duration']
    bandwidth = radar_params['bandwidth']
    center_freq = radar_params['center_freq']
    sample_rate = radar_params['sample_rate']
    speed_of_light = radar_params['speed_of_light']
    max_range = radar_params['max_range']
    
    # Add direct coupling component (TX leakage)
    direct_coupling_power = 0.01  # Adjust based on desired coupling strength
    for rx_idx in range(num_rx):
        # Direct coupling is a delayed and attenuated version of TX signal
        delay_samples = int(0.1 * samples_per_chirp)  # Small delay for direct path
        for chirp_idx in range(num_chirps):
            # Add attenuated TX signal with small delay
            tx_chirp = generate_fmcw_chirp(chirp_idx, chirp_duration, samples_per_chirp, bandwidth, center_freq)
            delayed_tx = np.zeros_like(tx_chirp)
            delayed_tx[delay_samples:] = tx_chirp[:-delay_samples] if delay_samples > 0 else tx_chirp
            rx_signal[rx_idx, chirp_idx] += np.sqrt(direct_coupling_power) * delayed_tx

    # Add environmental clutter (static reflections)
    num_clutter_points = random.randint(5, 15)
    for _ in range(num_clutter_points):
        clutter_range = random.uniform(5, max_range)
        clutter_rcs = random.uniform(-40, -20)  # dBsm
        clutter_power = calculate_received_power(clutter_range, clutter_rcs)
        
        # Add clutter to all chirps with same range (static)
        for rx_idx in range(num_rx):
            for chirp_idx in range(num_chirps):
                delay_samples = int((2 * clutter_range / speed_of_light) * sample_rate)
                if delay_samples < samples_per_chirp:
                    # Phase randomization for each clutter point
                    phase = random.uniform(0, 2 * np.pi)
                    rx_signal[rx_idx, chirp_idx, delay_samples:] += np.sqrt(clutter_power) * np.exp(1j * phase)
    
    # Add crosstalk between TX and RX (reduced effect)
    crosstalk_isolation_db = 60  # Increase from 30 to 40 dB isolation
    crosstalk_delay_samples = 5  # Small delay
    
    # Convert dB to linear scale
    crosstalk_factor = 10 ** (-crosstalk_isolation_db / 20)
    
    # Add crosstalk to all RX channels
    for rx_idx in range(num_rx):
        for chirp_idx in range(num_chirps):
            # Create delayed version of TX signal
            delayed_tx = np.zeros(samples_per_chirp, dtype=np.complex64)
            if crosstalk_delay_samples < samples_per_chirp:
                samples_to_copy = samples_per_chirp - crosstalk_delay_samples
                delayed_tx[crosstalk_delay_samples:] = tx_signal[chirp_idx, :samples_to_copy]
            
            # Add to RX signal with attenuation
            rx_signal[rx_idx, chirp_idx, :] += delayed_tx * crosstalk_factor
    
    # Add ground clutter - reduce the probability and power
    clutter_probability = 0.02  # Reduce from 0.05 to 0.02
    max_clutter_distance = max_range * 0.1  # Reduce from 0.2 to 0.1
    
    # Convert distance to samples
    max_clutter_samples = int(2 * max_clutter_distance * sample_rate / speed_of_light)
    
    # Add clutter reflections
    for sample_idx in range(min(max_clutter_samples, samples_per_chirp)):
        # Random chance of clutter at this range
        if random.random() < clutter_probability:
            # Calculate distance for this sample
            distance = sample_idx * speed_of_light / (2 * sample_rate)
            
            # Random RCS for clutter - reduce power
            clutter_rcs = random.uniform(0.05, 0.5)  # Reduce from (0.1, 1.0)
            
            # Calculate attenuation - reduce power
            attenuation = np.sqrt(clutter_rcs) / (distance ** 2) * 5e4  # Reduce from 1e5
            
            # Random phase
            phase = random.uniform(0, 2 * np.pi)
            
            # Add to all RX channels with random variations
            for rx_idx in range(num_rx):
                rx_phase_variation = random.uniform(0, 0.1)
                for chirp_idx in range(num_chirps):
                    rx_signal[rx_idx, chirp_idx, sample_idx] += attenuation * np.exp(1j * (phase + rx_phase_variation))
    
    # Add system noise (thermal noise, phase noise, etc.)
    system_noise_power = 1e-6
    system_noise = np.random.normal(0, np.sqrt(system_noise_power/2), rx_signal.shape) + \
                1j * np.random.normal(0, np.sqrt(system_noise_power/2), rx_signal.shape)
    rx_signal += system_noise
    
    return rx_signal

def add_noise(signal, snr_db):
    """Add realistic noise to the signal"""
    # Calculate signal power
    signal_power = np.mean(np.abs(signal)**2)
    
    # Ensure minimum signal power for noise calculation
    min_power = 1e-10
    signal_power = max(signal_power, min_power)
    
    # Calculate noise power based on SNR
    noise_power = signal_power / (10**(snr_db/10))
    
    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power/2) * (np.random.normal(0, 1, signal.shape) + 
                                    1j * np.random.normal(0, 1, signal.shape))
    
    # Add noise to signal
    return signal + noise

def calculate_received_power(distance, rcs):
    """
    Calculate received power based on radar equation.
    
    Args:
        distance: Target distance in meters
        rcs: Radar cross-section in dBsm
        
    Returns:
        Received power (linear scale)
    """
    # Convert RCS from dBsm to linear scale
    rcs_linear = 10**(rcs/10)
    
    # Simplified radar equation: P_r = P_t * G^2 * λ^2 * σ / ((4π)^3 * R^4)
    # We're using normalized values, so P_t * G^2 * λ^2 / (4π)^3 = 1
    received_power = rcs_linear / (distance**4)
    
    return received_power

def ray_tracing_simulation(tx_signal, targets, radar_params, perfect_mode=False, flatten_output=False, tx_signal_is_upconverted=False, analog_sample_rate=None):
    """
    Perform ray-tracing simulation to generate received signals.

    Args:
        tx_signal: Transmitted signal with shape [num_chirps, samples_per_chirp] or flattened 1D array
        targets: List of target dictionaries
        radar_params: Dictionary containing radar parameters
        perfect_mode: If True, uses a single fixed target for ideal simulation
        flatten_output: If True, returns a flattened 1D array similar to simulate_single_target_echo
        tx_signal_is_upconverted: If True, tx_signal is already upconverted to RF (center frequency)
    Returns:
        Complex RX signal with shape [num_rx, num_chirps, samples_per_chirp] or flattened 1D array
    """
    # Extract parameters from radar_params
    num_rx = radar_params['num_rx']
    num_chirps = radar_params['num_chirps']
    samples_per_chirp = radar_params['samples_per_chirp']
    chirp_duration = radar_params['chirp_duration']
    center_freq = radar_params['center_freq']
    sample_rate = radar_params['sample_rate']
    speed_of_light = radar_params['speed_of_light']
    wavelength = radar_params['wavelength']
    max_range = radar_params['max_range']
    apply_realistic_effects = radar_params.get('apply_realistic_effects', False)
    
    tx_is_flattened = tx_signal.ndim == 1
    if tx_is_flattened:
        tx_signal_reshaped = tx_signal.reshape(num_chirps, -1)
        samples_per_chirp = tx_signal_reshaped.shape[1]
    else:
        tx_signal_reshaped = tx_signal
        samples_per_chirp = tx_signal.shape[1]

    rx_signal = np.zeros((num_rx, num_chirps, samples_per_chirp), dtype=np.complex64)

    rx_positions = []
    rx_spacing = wavelength / 2  # Half-wavelength spacing
    for rx_idx in range(num_rx):
        rx_positions.append((rx_idx * rx_spacing, 0, 0))

    if perfect_mode and (targets is None or len(targets) == 0):
        perfect_target = {
            'distance': 50.0,
            'velocity': 10.0,
            'rcs': 20.0,
            'position': (50.0, 0, 0)
        }
        targets = [perfect_target]

    for target in targets:
        distance = target['distance']
        velocity = target['velocity']
        rcs = target['rcs']
        position = target['position']

        for rx_idx, rx_pos in enumerate(rx_positions):
            dx = position[0] - rx_pos[0]
            dy = position[1] - rx_pos[1]
            dz = position[2] - rx_pos[2]
            exact_distance = np.sqrt(dx**2 + dy**2 + dz**2)

            delay_seconds = 2 * exact_distance / speed_of_light
            # Choose sample rate based on RF or baseband simulation
            if tx_signal_is_upconverted:
                sample_rate_to_use = analog_sample_rate
            else:
                sample_rate_to_use = sample_rate
            delay_samples = int(delay_seconds * sample_rate_to_use)

            doppler_freq = 2 * velocity * center_freq / speed_of_light

            attenuation = np.sqrt(rcs) / (exact_distance ** 2)
            attenuation *= 5e6

            for chirp_idx in range(num_chirps):
                t = np.arange(samples_per_chirp) / sample_rate_to_use
                phase_shift = 2 * np.pi * doppler_freq * (chirp_idx * chirp_duration + t)
                delayed_signal = np.zeros(samples_per_chirp, dtype=np.complex64)
                samples_to_copy = min(samples_per_chirp - delay_samples, samples_per_chirp)
                if samples_to_copy > 0 and delay_samples < samples_per_chirp:
                    delayed_signal[delay_samples:delay_samples+samples_to_copy] = tx_signal_reshaped[chirp_idx, :samples_to_copy]
                    if not tx_signal_is_upconverted:
                        # Baseband simulation: apply upconversion here
                        delayed_signal *= np.exp(1j * 2 * np.pi * center_freq * (t + chirp_idx * chirp_duration))
                    # Both baseband and RF: apply Doppler and attenuation
                    delayed_signal *= attenuation * np.exp(1j * phase_shift)
                    rx_signal[rx_idx, chirp_idx, :] += delayed_signal

    if apply_realistic_effects and not perfect_mode:
        rx_signal = add_realistic_effects(rx_signal, tx_signal_reshaped, radar_params)

    if flatten_output:
        rx_flattened = np.zeros((num_rx, num_chirps * samples_per_chirp), dtype=np.complex64)
        for rx_idx in range(num_rx):
            rx_flattened[rx_idx] = rx_signal[rx_idx].flatten()
        return rx_flattened

    return rx_signal