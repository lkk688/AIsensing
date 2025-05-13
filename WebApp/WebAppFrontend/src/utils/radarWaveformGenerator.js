/**
 * Utility functions for radar waveform generation
 */

/**
 * Generates waveform data by calling the backend API
 * 
 * @param {Object} params - Radar parameters
 * @param {number} params.bandwidth - Bandwidth in MHz
 * @param {number} params.chirpDuration - Chirp duration in μs
 * @param {number} params.centerFreq - Center frequency in GHz
 * @param {number} params.sampleRate - Sample rate in MHz
 * @param {string} params.waveformType - Type of waveform ('linear', etc.)
*/
export const generateWaveformData = async (params) => {
    const { 
      bandwidth, 
      chirpDuration, 
      centerFreq, 
      sampleRate, 
      waveformType 
    } = params;
  
    // Convert parameters to base units for API call
    const chirpDurationSec = chirpDuration * 1e-6;
    const bandwidthHz = bandwidth * 1e6;
    const centerFreqHz = centerFreq * 1e9;
    const sampleRateHz = sampleRate * 1e6;
    
    // For local calculation (as fallback)
    const numSamples = Math.floor(chirpDurationSec * sampleRateHz);
    
    console.log("Calling backend with:", {
      bandwidth, chirpDuration, centerFreq, sampleRate, waveformType
    });
    
    try {
      const response = await fetch('http://localhost:8000/api/radar/waveform', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          bandwidth: bandwidth,
          chirpDuration: chirpDuration,
          centerFreq: centerFreq,
          sampleRate: sampleRate,
          waveformType: waveformType
        }),
      });
      
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      
      const data = await response.json();
      console.log("Waveform data received:", data);
      
      // Return the original Plotly data without conversion
      return {
        timeDomainPlot: data.timeDomainPlot,
        frequencyDomainPlot: data.frequencyDomainPlot,
        derivedParams: data.derivedParams || {}
      };
    } catch (error) {
      console.error('Error fetching waveform data:', error);
      // Return null or an error object to indicate failure
      return {
        error: 'Failed to fetch waveform data from backend. Using local calculation instead.'
      };
    }
  };
  
  /**
   * Calculates radar performance metrics based on input parameters
   * 
   * @param {number} bandwidth - Bandwidth in MHz
   * @param {number} chirpDuration - Chirp duration in μs
   * @param {number} sampleRate - Sample rate in MHz
   * @param {number} centerFreq - Center frequency in GHz
   * @param {number} maxRange - Maximum range in meters
   * @param {number} numChirps - Number of chirps
   * @param {number} numRx - Number of receive antennas
   * @param {number} numTx - Number of transmit antennas
   * @returns {Object} - Object containing derived radar parameters
   */
  export const calculateMetrics = (
    bandwidth, 
    chirpDuration, 
    sampleRate, 
    centerFreq, 
    maxRange, 
    numChirps, 
    numRx, 
    numTx
  ) => {
    // Convert to base units
    const bandwidthHz = bandwidth * 1e6;
    const chirpDurationSec = chirpDuration * 1e-6;
    const sampleRateHz = sampleRate * 1e6;
    const centerFreqHz = centerFreq * 1e9;
    
    // Physical constants
    const c = 3e8; // Speed of light in m/s
    const wavelength = c / centerFreqHz;
    
    // Range calculations
    const rangeResolution = c / (2 * bandwidthHz);
    const maxUnambiguousRange = (sampleRateHz * c) / (2 * bandwidthHz);
    
    // Time calculations
    const sweepSlope = bandwidthHz / chirpDurationSec;
    const samplesPerChirp = Math.floor(chirpDurationSec * sampleRateHz);
    
    // Velocity calculations
    const frameTime = chirpDurationSec * numChirps;
    const velocityResolution = wavelength / (2 * frameTime);
    const maxUnambiguousVelocity = wavelength / (4 * chirpDurationSec);
    
    // Angular calculations
    const angularResolution = wavelength / (numRx * 0.5 * wavelength); // Assuming half-wavelength spacing
    
    // SNR calculations
    const processingGain = 10 * Math.log10(samplesPerChirp);
    const coherentIntegrationGain = 10 * Math.log10(numChirps);
    const totalGain = processingGain + coherentIntegrationGain;
    
    return {
      rangeResolution: rangeResolution.toFixed(2),
      maxUnambiguousRange: Math.min(maxUnambiguousRange, maxRange).toFixed(2),
      velocityResolution: velocityResolution.toFixed(2),
      maxUnambiguousVelocity: maxUnambiguousVelocity.toFixed(2),
      angularResolution: (angularResolution * (180 / Math.PI)).toFixed(2),
      sweepSlope: (sweepSlope / 1e12).toFixed(2), // THz/s
      samplesPerChirp: samplesPerChirp,
      processingGain: processingGain.toFixed(2),
      coherentIntegrationGain: coherentIntegrationGain.toFixed(2),
      totalGain: totalGain.toFixed(2),
      wavelength: (wavelength * 1000).toFixed(2), // mm
    };
  };