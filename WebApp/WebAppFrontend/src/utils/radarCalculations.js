/**
 * Radar Calculations Utility
 * This file contains parameter explanations and calculations for radar systems
 */

// Parameter explanations with detailed descriptions, formulas, and relationships
const PARAMETER_EXPLANATIONS = {
    "Bandwidth": {
      "text": "Bandwidth determines the frequency span of the radar signal. Increasing bandwidth improves range resolution.",
      "latex": "Range\\ Resolution = \\frac{c}{2B}",
      "contributes": ["Range Resolution"]
    },
    "Chirp Duration": {
      "text": "Chirp duration is the time for one frequency sweep. Longer chirps can improve SNR but reduce maximum beat frequency.",
      "latex": "Slope = \\frac{B}{T_{chirp}}",
      "contributes": ["Chirp Slope", "Max Beat Frequency"]
    },
    "Center Frequency": {
      "text": "Center frequency is the middle frequency of the radar signal. It affects wavelength and maximum unambiguous velocity.",
      "latex": "\\lambda = \\frac{c}{f_c}",
      "contributes": ["Wavelength", "Max Unambiguous Velocity"]
    },
    "Sample Rate": {
      "text": "Sample rate is how fast the ADC samples the signal. It must be at least twice the maximum beat frequency (Nyquist).",
      "latex": "f_{Nyquist} = \\frac{f_{s}}{2}",
      "contributes": ["Nyquist Frequency", "Frequency Wraparound"]
    },
    "Waveform Type": {
      "text": "The type of frequency modulation used in the radar signal. Different types have different advantages for specific applications.",
      "latex": "",
      "contributes": ["Signal Properties", "Detection Capabilities"]
    },
    "Number of Chirps": {
      "text": "Number of chirps in a frame. More chirps improve velocity resolution but increase frame duration.",
      "latex": "v_{resolution} = \\frac{\\lambda}{2 \\cdot N_{chirps} \\cdot T_{chirp}}",
      "contributes": ["Velocity Resolution", "Frame Duration"]
    },
    "Number of RX Antennas": {
      "text": "Number of receive antennas. More RX antennas improve angular resolution and SNR.",
      "latex": "\\theta_{resolution} \\propto \\frac{1}{N_{RX}}",
      "contributes": ["Angular Resolution"]
    },
    "Number of TX Antennas": {
      "text": "Number of transmit antennas. More TX antennas improve angular resolution through virtual array extension.",
      "latex": "N_{virtual} = N_{TX} \\cdot N_{RX}",
      "contributes": ["Angular Resolution"]
    },
    "Chirp Slope": {
      "text": "Chirp slope is the rate of frequency change during a chirp. It affects the beat frequency for a given range.",
      "latex": "Slope = \\frac{B}{T_{chirp}}",
      "contributes": ["Max Beat Frequency"]
    },
    "Range Resolution": {
      "text": "Range resolution is the minimum distance between two distinguishable targets. It improves with higher bandwidth.",
      "latex": "Range\\ Resolution = \\frac{c}{2B}",
      "contributes": ["Bandwidth"]
    },
    "Maximum Unambiguous Range": {
      "text": "The maximum range that can be detected without ambiguity. Depends on chirp duration and sample rate.",
      "latex": "R_{max} = \\frac{c \\cdot f_s \\cdot T_{chirp}}{2B}",
      "contributes": ["Chirp Duration", "Sample Rate"]
    },
    "Velocity Resolution": {
      "text": "Velocity resolution is the minimum difference in velocity between two distinguishable targets.",
      "latex": "v_{resolution} = \\frac{\\lambda}{2 \\cdot N_{chirps} \\cdot T_{chirp}}",
      "contributes": ["Number of Chirps", "Chirp Duration", "Center Frequency"]
    },
    "Maximum Unambiguous Velocity": {
      "text": "Maximum unambiguous velocity is the highest velocity that can be measured without aliasing.",
      "latex": "v_{max} = \\frac{\\lambda}{4 \\cdot T_{chirp}}",
      "contributes": ["Center Frequency", "Chirp Duration"]
    },
    "Wavelength": {
      "text": "Wavelength is the distance between consecutive peaks of the radar signal. It affects velocity measurements.",
      "latex": "\\lambda = \\frac{c}{f_c}",
      "contributes": ["Center Frequency"]
    },
    "Max Beat Frequency": {
      "text": "Maximum beat frequency is the highest frequency difference between TX and RX signals for the farthest target.",
      "latex": "f_{beat,max} = \\frac{2 \\cdot Slope \\cdot R_{max}}{c}",
      "contributes": ["Chirp Slope", "Max Range"]
    },
    "Nyquist Frequency": {
      "text": "Nyquist frequency is half the sample rate. Beat frequencies above this will alias and cause errors.",
      "latex": "f_{Nyquist} = \\frac{f_{s}}{2}",
      "contributes": ["Sample Rate"]
    },
    "Frequency Wraparound": {
      "text": "Frequency wraparound (aliasing) occurs if the max beat frequency exceeds the Nyquist frequency. Increase sample rate or reduce max range/bandwidth to avoid.",
      "latex": "f_{beat,max} \\leq f_{Nyquist}",
      "contributes": ["Sample Rate", "Max Beat Frequency"]
    },
    "Samples per Chirp": {
      "text": "Number of samples collected during one chirp. Determined by sample rate and chirp duration.",
      "latex": "N_{samples} = f_s \\cdot T_{chirp}",
      "contributes": ["Sample Rate", "Chirp Duration"]
    },
    "Range FFT Size": {
      "text": "Size of the FFT used for range processing. Usually the next power of 2 above samples per chirp.",
      "latex": "N_{range\\_fft} = 2^{\\lceil \\log_2(N_{samples}) \\rceil}",
      "contributes": ["Samples per Chirp"]
    },
    "Doppler FFT Size": {
      "text": "Size of the FFT used for Doppler processing. Usually the next power of 2 above number of chirps.",
      "latex": "N_{doppler\\_fft} = 2^{\\lceil \\log_2(N_{chirps}) \\rceil}",
      "contributes": ["Number of Chirps"]
    },
    "Frame Duration": {
      "text": "Total time for all chirps in a frame. Affects the refresh rate of the radar.",
      "latex": "T_{frame} = N_{chirps} \\cdot T_{chirp}",
      "contributes": ["Number of Chirps", "Chirp Duration"]
    },
    "Refresh Rate": {
      "text": "How frequently the radar updates its measurements. Inverse of frame duration.",
      "latex": "f_{refresh} = \\frac{1}{T_{frame}}",
      "contributes": ["Frame Duration"]
    },
    "Actual Sweep": {
      "text": "The actual frequency sweep achieved during the chirp.",
      "latex": "Sweep = Slope \\cdot T_{chirp}",
      "contributes": ["Chirp Slope", "Chirp Duration"]
    },
    "Angular Resolution": {
      "text": "The minimum angular separation between two targets that can be distinguished.",
      "latex": "\\theta_{resolution} \\propto \\frac{\\lambda}{N_{virtual} \\cdot d}",
      "contributes": ["Number of RX Antennas", "Number of TX Antennas", "Wavelength"]
    }
};

// Simple parameter explanations for tooltips
const PARAM_EXPLANATIONS = {
    rangeResolution: "Range resolution is the minimum distance between two targets that can be distinguished by the radar. It is calculated as c/(2*bandwidth), where c is the speed of light.",
    maxRange: "Maximum range is the farthest distance the radar can detect targets. It depends on the chirp duration, bandwidth, and sample rate.",
    velocityResolution: "Velocity resolution is the minimum difference in velocity between two targets that can be distinguished. It is calculated as λ/(2*num_chirps*chirp_duration), where λ is the wavelength.",
    maxVelocity: "Maximum velocity is the highest target speed that can be measured without ambiguity. It is calculated as λ/(4*chirp_duration).",
    snr: "Signal-to-Noise Ratio (SNR) is a measure of signal strength relative to background noise. Higher SNR means better detection capability.",
    processingGain: "Processing gain is the improvement in SNR achieved through signal processing. It is proportional to the time-bandwidth product.",
};

/**
 * Calculate radar parameters based on input values
 * @param {Object} params - Input parameters
 * @returns {Object} Calculated radar parameters
 */
const calculateRadarParameters = (params) => {
    const {
        bandwidth = 200e6, // Hz
        chirpDuration = 100e-6, // seconds
        centerFrequency = 77e9, // Hz
        sampleRate = 10e6, // Hz
        numChirps = 128,
        numRxAntennas = 4,
        numTxAntennas = 1
    } = params;

    const c = 3e8; // Speed of light in m/s
    const wavelength = c / centerFrequency;
    
    // Calculate basic parameters
    const rangeResolution = c / (2 * bandwidth);
    const samplesPerChirp = Math.floor(sampleRate * chirpDuration);
    const chirpSlope = bandwidth / chirpDuration;
    
    // Calculate range parameters
    const maxRange = (sampleRate * c * chirpDuration) / (2 * bandwidth);
    const maxBeatFrequency = (2 * chirpSlope * maxRange) / c;
    const nyquistFrequency = sampleRate / 2;
    
    // Calculate velocity parameters
    const velocityResolution = wavelength / (2 * numChirps * chirpDuration);
    const maxUnambiguousVelocity = wavelength / (4 * chirpDuration);
    
    // Calculate FFT sizes (next power of 2)
    const rangeFftSize = Math.pow(2, Math.ceil(Math.log2(samplesPerChirp)));
    const dopplerFftSize = Math.pow(2, Math.ceil(Math.log2(numChirps)));
    
    // Calculate timing parameters
    const frameDuration = numChirps * chirpDuration;
    const refreshRate = 1 / frameDuration;
    
    // Calculate angular resolution
    const virtualArraySize = numRxAntennas * numTxAntennas;
    const antennaSpacing = wavelength / 2; // Typical spacing
    const angularResolution = wavelength / (virtualArraySize * antennaSpacing);
    
    return {
        rangeResolution,
        maxRange,
        velocityResolution,
        maxUnambiguousVelocity,
        wavelength,
        samplesPerChirp,
        chirpSlope,
        maxBeatFrequency,
        nyquistFrequency,
        rangeFftSize,
        dopplerFftSize,
        frameDuration,
        refreshRate,
        angularResolution,
        frequencyWraparound: maxBeatFrequency > nyquistFrequency
    };
};

/**
 * Check if radar parameters are valid
 * @param {Object} params - Radar parameters
 * @returns {Object} Validation results with warnings
 */
const validateRadarParameters = (params) => {
    const {
        maxBeatFrequency,
        nyquistFrequency,
        samplesPerChirp,
        maxRange,
        rangeResolution,
        maxUnambiguousVelocity
    } = params;
    
    const warnings = [];
    
    // Check for frequency wraparound
    if (maxBeatFrequency > nyquistFrequency) {
        warnings.push({
            type: 'error',
            message: 'Frequency wraparound detected! The maximum beat frequency exceeds the Nyquist frequency.',
            suggestion: 'Increase sample rate, reduce bandwidth, or reduce maximum range.'
        });
    }
    
    // Check for minimum samples per chirp
    if (samplesPerChirp < 16) {
        warnings.push({
            type: 'warning',
            message: 'Very few samples per chirp.',
            suggestion: 'Increase chirp duration or sample rate for better performance.'
        });
    }
    
    // Check for reasonable maximum range
    if (maxRange > 1000) {
        warnings.push({
            type: 'warning',
            message: 'Very large maximum range.',
            suggestion: 'Ensure transmit power is sufficient for this range.'
        });
    }
    
    // Check for reasonable range resolution
    if (rangeResolution > 1) {
        warnings.push({
            type: 'warning',
            message: 'Coarse range resolution.',
            suggestion: 'Increase bandwidth for better range resolution.'
        });
    }
    
    return {
        isValid: warnings.filter(w => w.type === 'error').length === 0,
        warnings
    };
};

export {
    PARAMETER_EXPLANATIONS,
    PARAM_EXPLANATIONS,
    calculateRadarParameters,
    validateRadarParameters
};