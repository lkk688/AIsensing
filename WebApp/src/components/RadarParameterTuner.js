import React, { useState } from 'react';
import { 
  Box, Typography, Slider, Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow, Paper, Button, Grid
} from '@mui/material';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';

// Parameter explanations from the original Python code
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
  "Max Range": {
    "text": "Maximum range is the farthest distance the radar can detect, determined by chirp duration and speed of light.",
    "latex": "T_{chirp} = \\frac{2 \\cdot R_{max}}{c}",
    "contributes": ["Chirp Duration"]
  },
  "Sample Rate": {
    "text": "Sample rate is how fast the ADC samples the signal. It must be at least twice the maximum beat frequency (Nyquist).",
    "latex": "f_{Nyquist} = \\frac{f_{s}}{2}",
    "contributes": ["Nyquist Frequency", "Frequency Wraparound"]
  },
  "Center Frequency": {
    "text": "Center frequency is the middle frequency of the radar signal. It affects wavelength and maximum unambiguous velocity.",
    "latex": "\\lambda = \\frac{c}{f_c}",
    "contributes": ["Wavelength", "Max Unambiguous Velocity"]
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
  "Velocity Resolution": {
    "text": "Velocity resolution is the minimum difference in velocity between two distinguishable targets.",
    "latex": "v_{resolution} = \\frac{\\lambda}{2 \\cdot N_{chirps} \\cdot T_{chirp}}",
    "contributes": ["Number of Chirps", "Chirp Duration", "Center Frequency"]
  },
  "Max Unambiguous Velocity": {
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
  "Actual Sweep": {
    "text": "The actual frequency sweep achieved during the chirp.",
    "latex": "Sweep = Slope \\cdot T_{chirp}",
    "contributes": ["Chirp Slope", "Chirp Duration"]
  }
};

const RadarParameterTuner = () => {
  // State for slider values
  const [bandwidth, setBandwidth] = useState(100);
  const [chirpDuration, setChirpDuration] = useState(100);
  const [maxRange, setMaxRange] = useState(100);
  const [sampleRate, setSampleRate] = useState(50);
  const [centerFreq, setCenterFreq] = useState(77); // GHz
  const [numChirps, setNumChirps] = useState(128);
  const [numRx, setNumRx] = useState(4);
  const [numTx, setNumTx] = useState(1);
  
  // State for explanation
  const [explanation, setExplanation] = useState(null);
  
  // Calculate metrics based on slider values
  const calculateMetrics = () => {
    const c = 3e8; // Speed of light in m/s
    const bandwidthHz = bandwidth * 1e6; // Convert MHz to Hz
    const chirpDurationSec = chirpDuration * 1e-6; // Convert us to seconds
    const sampleRateHz = sampleRate * 1e6; // Convert MHz to Hz
    const centerFreqHz = centerFreq * 1e9; // Convert GHz to Hz
    
    // Calculate wavelength
    const wavelength = c / centerFreqHz;
    
    // Calculate FMCW slope
    const slope = bandwidthHz / chirpDurationSec;
    
    // Calculate range resolution
    const rangeResolution = c / (2 * bandwidthHz);
    
    // Calculate max beat frequency
    const fBeatMax = 2 * slope * maxRange / c;
    
    // Calculate Nyquist frequency
    const nyquist = sampleRateHz / 2;
    
    // Check for frequency wraparound
    const freqWrap = fBeatMax <= nyquist ? "No" : "Yes";
    
    // Calculate samples per chirp
    const samplesPerChirp = Math.floor(sampleRateHz * chirpDurationSec);
    
    // Calculate velocity resolution
    const velocityResolution = wavelength / (2 * numChirps * chirpDurationSec);
    
    // Calculate max unambiguous velocity
    const maxUnambiguousVelocity = wavelength / (4 * chirpDurationSec);
    
    // Calculate frame duration
    const frameDuration = numChirps * chirpDurationSec;
    
    // Calculate refresh rate
    const refreshRate = 1.0 / frameDuration;
    
    // Calculate actual sweep
    const actualSweep = (slope * chirpDurationSec) / 1e6; // MHz
    
    // Calculate range FFT size (next power of 2)
    const rangeFftSize = Math.pow(2, Math.ceil(Math.log2(samplesPerChirp)));
    
    // Calculate doppler FFT size (next power of 2)
    const dopplerFftSize = Math.pow(2, Math.ceil(Math.log2(numChirps)));
    
    return [
      { name: "Bandwidth", value: `${bandwidth.toFixed(2)} MHz` },
      { name: "Chirp Duration", value: `${chirpDuration.toFixed(2)} μs` },
      { name: "Max Range", value: `${maxRange.toFixed(2)} m` },
      { name: "Sample Rate", value: `${sampleRate.toFixed(2)} MHz` },
      { name: "Center Frequency", value: `${centerFreq.toFixed(2)} GHz` },
      { name: "Number of Chirps", value: `${numChirps}` },
      { name: "Number of RX Antennas", value: `${numRx}` },
      { name: "Number of TX Antennas", value: `${numTx}` },
      { name: "Wavelength", value: `${(wavelength * 1000).toFixed(2)} mm` },
      { name: "Chirp Slope", value: `${(slope / 1e12).toFixed(2)} THz/s` },
      { name: "Range Resolution", value: `${rangeResolution.toFixed(2)} m` },
      { name: "Velocity Resolution", value: `${velocityResolution.toFixed(2)} m/s` },
      { name: "Max Unambiguous Velocity", value: `${maxUnambiguousVelocity.toFixed(2)} m/s` },
      { name: "Frame Duration", value: `${(frameDuration * 1000).toFixed(2)} ms` },
      { name: "Refresh Rate", value: `${refreshRate.toFixed(2)} Hz` },
      { name: "Actual Sweep", value: `${actualSweep.toFixed(2)} MHz` },
      { name: "Samples per Chirp", value: `${samplesPerChirp}` },
      { name: "Range FFT Size", value: `${rangeFftSize}` },
      { name: "Doppler FFT Size", value: `${dopplerFftSize}` },
      { name: "Max Beat Frequency", value: `${(fBeatMax / 1e6).toFixed(2)} MHz` },
      { name: "Nyquist Frequency", value: `${(nyquist / 1e6).toFixed(2)} MHz` },
      { name: "Frequency Wraparound", value: freqWrap }
    ];
  };
  
  const metrics = calculateMetrics();
  
  const showExplanation = (metricName) => {
    setExplanation(PARAMETER_EXPLANATIONS[metricName]);
  };
  
  return (
    <Box sx={{ width: '100%', maxWidth: 1200, margin: '0 auto', padding: 3 }}>
      <Typography variant="h4" gutterBottom>
        FMCW Radar Parameter Tuner
      </Typography>
      
      <Grid container spacing={2} sx={{ marginBottom: 4 }}>
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Bandwidth (MHz)</Typography>
          <Slider
            value={bandwidth}
            onChange={(e, newValue) => setBandwidth(newValue)}
            min={1}
            max={500}
            valueLabelDisplay="auto"
            marks={[
              { value: 1, label: '1' },
              { value: 500, label: '500' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Chirp Duration (μs)</Typography>
          <Slider
            value={chirpDuration}
            onChange={(e, newValue) => setChirpDuration(newValue)}
            min={1}
            max={1000}
            valueLabelDisplay="auto"
            marks={[
              { value: 1, label: '1' },
              { value: 1000, label: '1000' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Max Range (m)</Typography>
          <Slider
            value={maxRange}
            onChange={(e, newValue) => setMaxRange(newValue)}
            min={10}
            max={1000}
            valueLabelDisplay="auto"
            marks={[
              { value: 10, label: '10' },
              { value: 1000, label: '1000' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Sample Rate (MHz)</Typography>
          <Slider
            value={sampleRate}
            onChange={(e, newValue) => setSampleRate(newValue)}
            min={1}
            max={500}
            valueLabelDisplay="auto"
            marks={[
              { value: 1, label: '1' },
              { value: 500, label: '500' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Center Frequency (GHz)</Typography>
          <Slider
            value={centerFreq}
            onChange={(e, newValue) => setCenterFreq(newValue)}
            min={24}
            max={81}
            valueLabelDisplay="auto"
            marks={[
              { value: 24, label: '24' },
              { value: 77, label: '77' },
              { value: 81, label: '81' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Number of Chirps</Typography>
          <Slider
            value={numChirps}
            onChange={(e, newValue) => setNumChirps(newValue)}
            min={16}
            max={512}
            step={16}
            valueLabelDisplay="auto"
            marks={[
              { value: 16, label: '16' },
              { value: 512, label: '512' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Number of RX Antennas</Typography>
          <Slider
            value={numRx}
            onChange={(e, newValue) => setNumRx(newValue)}
            min={1}
            max={8}
            step={1}
            valueLabelDisplay="auto"
            marks={[
              { value: 1, label: '1' },
              { value: 8, label: '8' }
            ]}
          />
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Typography gutterBottom>Number of TX Antennas</Typography>
          <Slider
            value={numTx}
            onChange={(e, newValue) => setNumTx(newValue)}
            min={1}
            max={4}
            step={1}
            valueLabelDisplay="auto"
            marks={[
              { value: 1, label: '1' },
              { value: 4, label: '4' }
            ]}
          />
        </Grid>
      </Grid>
      
      <TableContainer component={Paper} sx={{ marginBottom: 4 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {metrics.map((metric) => (
              <TableRow key={metric.name}>
                <TableCell>{metric.name}</TableCell>
                <TableCell>{metric.value}</TableCell>
                <TableCell>
                  <Button 
                    variant="contained" 
                    size="small"
                    onClick={() => showExplanation(metric.name)}
                  >
                    Explain
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      {explanation && (
        <Paper sx={{ padding: 2 }}>
          <Typography variant="h6">{explanation.text}</Typography>
          <Box sx={{ marginTop: 2, fontFamily: 'monospace' }}>
            <Typography variant="subtitle1">Formula:</Typography>
            <Latex math={explanation.latex} />
          </Box>
          <Typography variant="subtitle1" sx={{ marginTop: 2 }}>
            Contributing Parameters:
          </Typography>
          <ul>
            {explanation.contributes.map((param) => (
              <li key={param}>{param}</li>
            ))}
          </ul>
        </Paper>
      )}
    </Box>
  );
};

export default RadarParameterTuner;