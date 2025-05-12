import React, { useState, useEffect, useRef, useMemo, memo, useCallback } from 'react';
import {
    Box, Typography, Slider, Container, Paper, Grid,
    FormControl, InputLabel, Select, MenuItem, Button,
    CircularProgress, Popover
} from '@mui/material';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';
import Plot from 'react-plotly.js';

// Parameter explanations from RadarParameterTuner.js
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

const calculateMetrics = (bandwidth, chirpDuration, sampleRate, centerFreq, maxRange, numChirps, numRx, numTx) => {
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
    
    // Calculate virtual array size
    const virtualArraySize = numRx * numTx;
    
    // Calculate angular resolution (simplified formula)
    const angularResolution = (wavelength / (virtualArraySize * 0.5)) * (180 / Math.PI);
    
    return {
      chirpSlope: { value: slope / 1e12, unit: 'THz/s' },
      maxBeatFrequency: { value: fBeatMax / 1e6, unit: 'MHz' },
      rangeResolution: { value: rangeResolution, unit: 'm' },
      maxRange: { value: maxRange, unit: 'm' },
      velocityResolution: { value: velocityResolution, unit: 'm/s' },
      maxVelocity: { value: maxUnambiguousVelocity, unit: 'm/s' },
      wavelength: { value: wavelength * 1000, unit: 'mm' },
      samplesPerChirp: { value: samplesPerChirp },
      frameDuration: { value: frameDuration * 1000, unit: 'ms' },
      refreshRate: { value: refreshRate, unit: 'Hz' },
      actualSweep: { value: actualSweep, unit: 'MHz' },
      rangeFFTSize: { value: rangeFftSize },
      dopplerFFTSize: { value: dopplerFftSize },
      nyquistFrequency: { value: nyquist / 1e6, unit: 'MHz' },
      angularResolution: { value: angularResolution, unit: '°' },
      virtualArraySize: { value: virtualArraySize },
      frequencyWraparound: { value: freqWrap }
    };
};

// Memoized Plot component to prevent re-renders when props haven't changed
const MemoizedPlot = memo(({ data, layout, config, style, useResizeHandler }) => (
    <Plot
        data={data}
        layout={layout}
        config={config}
        style={style}
        useResizeHandler={useResizeHandler}
    />
));

// Memoized component for displaying derived parameters
const DerivedParameters = memo(({ derivedParams, loading, error }) => {
    const [anchorEl, setAnchorEl] = useState(null);
    const [currentParam, setCurrentParam] = useState(null);

    const handleExplainClick = (event, paramName) => {
        setCurrentParam(paramName);
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const open = Boolean(anchorEl);

    // Create a reusable parameter row component for better modularity
    const ParameterRow = ({ label, value, unit, paramName }) => (
        <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="body2">
            {label}: {typeof value === 'number' ? value.toFixed(2) : value || 0} {unit || ''}
            </Typography>
            <Button 
                size="small" 
                variant="outlined" 
                onClick={(e) => handleExplainClick(e, paramName)}
            >
                Explain
            </Button>
        </Box>
    );

    // Group parameters by category for better organization
    const BasicParameters = () => (
        <>
            <ParameterRow 
                label="Chirp Slope" 
                value={derivedParams.chirpSlope?.value} 
                unit={derivedParams.chirpSlope?.unit || 'THz/s'} 
                paramName="Chirp Slope" 
            />
            
            <ParameterRow 
                label="Max Beat Frequency" 
                value={derivedParams.maxBeatFrequency?.value} 
                unit={derivedParams.maxBeatFrequency?.unit || 'MHz'} 
                paramName="Max Beat Frequency" 
            />
            
            <ParameterRow 
                label="Range Resolution" 
                value={derivedParams.rangeResolution?.value} 
                unit={derivedParams.rangeResolution?.unit || 'm'} 
                paramName="Range Resolution" 
            />
            
            <ParameterRow 
                label="Maximum Unambiguous Range" 
                value={derivedParams.maxRange?.value} 
                unit={derivedParams.maxRange?.unit || 'm'} 
                paramName="Maximum Unambiguous Range" 
            />
        </>
    );

    const VelocityParameters = () => (
        <>
            <ParameterRow 
                label="Velocity Resolution" 
                value={derivedParams.velocityResolution?.value} 
                unit={derivedParams.velocityResolution?.unit || 'm/s'} 
                paramName="Velocity Resolution" 
            />
            
            <ParameterRow 
                label="Maximum Unambiguous Velocity" 
                value={derivedParams.maxVelocity?.value} 
                unit={derivedParams.maxVelocity?.unit || 'm/s'} 
                paramName="Maximum Unambiguous Velocity" 
            />
        </>
    );

    const SignalParameters = () => (
        <>
            <ParameterRow 
                label="Wavelength" 
                value={derivedParams.wavelength?.value} 
                unit={derivedParams.wavelength?.unit || 'mm'} 
                paramName="Wavelength" 
            />
            
            <ParameterRow 
                label="Samples per Chirp" 
                value={derivedParams.samplesPerChirp?.value} 
                unit="" 
                paramName="Samples per Chirp" 
            />
        </>
    );

    const TimingParameters = () => (
        <>
            <ParameterRow 
                label="Frame Duration" 
                value={derivedParams.frameDuration?.value} 
                unit={derivedParams.frameDuration?.unit || 'ms'} 
                paramName="Frame Duration" 
            />
            
            <ParameterRow 
                label="Refresh Rate" 
                value={derivedParams.refreshRate?.value} 
                unit={derivedParams.refreshRate?.unit || 'Hz'} 
                paramName="Refresh Rate" 
            />
            
            <ParameterRow 
                label="Actual Sweep" 
                value={derivedParams.actualSweep?.value} 
                unit={derivedParams.actualSweep?.unit || 'MHz'} 
                paramName="Actual Sweep" 
            />
        </>
    );

    // New component for additional derived parameters
    const AdvancedParameters = () => (
        <>
            <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                Advanced Parameters:
            </Typography>
            
            <ParameterRow 
                label="Range FFT Size" 
                value={derivedParams.rangeFFTSize?.value} 
                unit="" 
                paramName="Range FFT Size" 
            />
            
            <ParameterRow 
                label="Doppler FFT Size" 
                value={derivedParams.dopplerFFTSize?.value} 
                unit="" 
                paramName="Doppler FFT Size" 
            />
            
            <ParameterRow 
                label="Nyquist Frequency" 
                value={derivedParams.nyquistFrequency?.value} 
                unit={derivedParams.nyquistFrequency?.unit || 'MHz'} 
                paramName="Nyquist Frequency" 
            />
            
            <ParameterRow 
                label="Angular Resolution" 
                value={derivedParams.angularResolution?.value} 
                unit={derivedParams.angularResolution?.unit || '°'} 
                paramName="Angular Resolution" 
            />
            
            <ParameterRow 
                label="Virtual Array Size" 
                value={derivedParams.virtualArraySize?.value} 
                unit="" 
                paramName="Angular Resolution" 
            />
            
            <ParameterRow 
                label="Frequency Wraparound" 
                value={derivedParams.frequencyWraparound?.value} 
                unit={derivedParams.frequencyWraparound?.unit || 'MHz'} 
                paramName="Frequency Wraparound" 
            />
        </>
    );

    return (
        // Add a fixed height container to prevent layout shifts
        <Box sx={{ minHeight: '220px' }}>
            {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                    <CircularProgress size={24} />
                </Box>
            ) : error ? (
                <Typography color="error">{error}</Typography>
            ) : (
                <>
                    <BasicParameters />
                    <VelocityParameters />
                    <SignalParameters />
                    <TimingParameters />
                    <AdvancedParameters />
                    
                    <Popover
                        open={open}
                        anchorEl={anchorEl}
                        onClose={handleClose}
                        anchorOrigin={{
                            vertical: 'bottom',
                            horizontal: 'center',
                        }}
                        transformOrigin={{
                            vertical: 'top',
                            horizontal: 'center',
                        }}
                    >
                        <Box sx={{ p: 2, maxWidth: 400 }}>
                            {currentParam && PARAMETER_EXPLANATIONS[currentParam] && (
                                <>
                                    <Typography variant="h6" gutterBottom>{currentParam}</Typography>
                                    <Typography variant="body2" paragraph>
                                        {PARAMETER_EXPLANATIONS[currentParam].text}
                                    </Typography>
                                    {PARAMETER_EXPLANATIONS[currentParam].latex && (
                                        <Box sx={{ my: 2, display: 'flex', justifyContent: 'center' }}>
                                            <Latex>{PARAMETER_EXPLANATIONS[currentParam].latex}</Latex>
                                        </Box>
                                    )}
                                    {PARAMETER_EXPLANATIONS[currentParam].contributes && (
                                        <>
                                            <Typography variant="subtitle2">Contributes to:</Typography>
                                            <ul style={{ margin: '8px 0' }}>
                                                {PARAMETER_EXPLANATIONS[currentParam].contributes.map((item, idx) => (
                                                    <li key={idx}><Typography variant="body2">{item}</Typography></li>
                                                ))}
                                            </ul>
                                        </>
                                    )}
                                </>
                            )}
                        </Box>
                    </Popover>
                </>
            )}
        </Box>
    );
});

// Memoized component for parameter controls
const ParameterControls = memo(({ 
    bandwidth, setBandwidth,
    chirpDuration, setChirpDuration,
    centerFreq, setCenterFreq,
    sampleRate, setSampleRate,
    waveformType, setWaveformType,
    numChirps, setNumChirps,
    numRx, setNumRx,
    numTx, setNumTx,
    maxRange, setMaxRange,
    setIsSliding
}) => {
    const [anchorEl, setAnchorEl] = useState(null);
    const [currentParam, setCurrentParam] = useState(null);

    const handleExplainClick = (event, paramName) => {
        setCurrentParam(paramName);
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const open = Boolean(anchorEl);

    return (
        <Paper elevation={2} sx={{ p: 2, minHeight: '400px' }}>
            <Typography variant="h6" gutterBottom>
                Waveform Parameters
            </Typography>
            
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Waveform Type
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Waveform Type")}
                >
                    Explain
                </Button>
            </Box>
            <FormControl fullWidth sx={{ mb: 2 }}>
                <Select
                    value={waveformType}
                    onChange={(e) => setWaveformType(e.target.value)}
                >
                    <MenuItem value="linear">Linear Chirp</MenuItem>
                    <MenuItem value="sawtooth">Sawtooth Chirp</MenuItem>
                    <MenuItem value="triangular">Triangular Chirp</MenuItem>
                </Select>
            </FormControl>
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Bandwidth: {bandwidth} MHz
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Bandwidth")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={bandwidth}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setBandwidth(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={10}
                max={500}
                step={10}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Chirp Duration: {chirpDuration} μs
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Chirp Duration")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={chirpDuration}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setChirpDuration(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={10}
                max={500}
                step={10}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Center Frequency: {centerFreq} GHz
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Center Frequency")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={centerFreq}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setCenterFreq(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={24}
                max={81}
                step={1}
                marks={[
                    { value: 24, label: '24 GHz' },
                    { value: 60, label: '60 GHz' },
                    { value: 77, label: '77 GHz' }
                ]}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Sample Rate: {sampleRate} MHz
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Sample Rate")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={sampleRate}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setSampleRate(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={10}
                max={200}
                step={10}
                sx={{ mb: 2 }}
            />
            
            {/* New parameters added to match RadarParameterTuner.js */}
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Number of Chirps: {numChirps}
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Number of Chirps")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={numChirps}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setNumChirps(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={1}
                max={512}
                step={1}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Number of RX Antennas: {numRx}
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Number of RX Antennas")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={numRx}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setNumRx(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={1}
                max={8}
                step={1}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Number of TX Antennas: {numTx}
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Number of TX Antennas")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={numTx}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setNumTx(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={1}
                max={4}
                step={1}
                sx={{ mb: 2 }}
            />
            
            <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Typography gutterBottom>
                    Max Range: {maxRange} m
                </Typography>
                <Button 
                    size="small" 
                    variant="outlined" 
                    onClick={(e) => handleExplainClick(e, "Maximum Unambiguous Range")}
                >
                    Explain
                </Button>
            </Box>
            <Slider
                value={maxRange}
                onChange={(e, newValue) => {
                    setIsSliding(true);
                    setMaxRange(newValue);
                }}
                onChangeCommitted={() => {
                    setIsSliding(false);
                }}
                min={10}
                max={1000}
                step={10}
                sx={{ mb: 2 }}
            />

            <Popover
                open={open}
                anchorEl={anchorEl}
                onClose={handleClose}
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'center',
                }}
                transformOrigin={{
                    vertical: 'top',
                    horizontal: 'center',
                }}
            >
                <Box sx={{ p: 2, maxWidth: 400 }}>
                    {currentParam && PARAMETER_EXPLANATIONS[currentParam] && (
                        <>
                            <Typography variant="h6" gutterBottom>{currentParam}</Typography>
                            <Typography variant="body2" paragraph>
                                {PARAMETER_EXPLANATIONS[currentParam].text}
                            </Typography>
                            {PARAMETER_EXPLANATIONS[currentParam].latex && (
                                <Box sx={{ my: 2, display: 'flex', justifyContent: 'center' }}>
                                    <Latex>{PARAMETER_EXPLANATIONS[currentParam].latex}</Latex>
                                </Box>
                            )}
                            {PARAMETER_EXPLANATIONS[currentParam].contributes && (
                                <>
                                    <Typography variant="subtitle2" gutterBottom>
                                        Contributing Parameters:
                                    </Typography>
                                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                                        {PARAMETER_EXPLANATIONS[currentParam].contributes.map((param, index) => (
                                            <li key={index}>
                                                <Typography variant="body2">{param}</Typography>
                                            </li>
                                        ))}
                                    </ul>
                                </>
                            )}
                        </>
                    )}
                </Box>
            </Popover>
        </Paper>
    );
});

// Memoized component for equations
const RadarEquations = memo(() => (
    <>
        <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
                FMCW Equation:
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                <Latex>
                    {`s(t) = \\exp\\left(j2\\pi \\left(f_c t + \\frac{B}{2T}t^2\\right)\\right)`}
                </Latex>
            </Box>
        </Box>
        
        <Box sx={{ mt: 2 }}>
            <Typography variant="body2">
                Beat Frequency Equation:
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                <Latex>
                    {`f_b = \\frac{2R}{c} \\cdot \\frac{B}{T}`}
                </Latex>
            </Box>
        </Box>
    </>
));

const RadarWaveformGeneratorPy = memo(({ setIsSliding: setParentIsSliding }) => {
    // State for radar parameters
    const [bandwidth, setBandwidth] = useState(100); // MHz
    const [chirpDuration, setChirpDuration] = useState(100); // μs
    const [centerFreq, setCenterFreq] = useState(77); // GHz
    const [sampleRate, setSampleRate] = useState(50); // MHz
    const [waveformType, setWaveformType] = useState('linear');
    
    // Add new state variables for the additional parameters
    const [numChirps, setNumChirps] = useState(128);
    const [numRx, setNumRx] = useState(4);
    const [numTx, setNumTx] = useState(1);
    const [maxRange, setMaxRange] = useState(100); // meters
    
    // State for derived parameters
    const [derivedParams, setDerivedParams] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [isSliding, setIsSliding] = useState(false);
    
    // State for waveform data
    const [waveformData, setWaveformData] = useState(null);

    const [isLoading, setIsLoading] = useState(false);
    const [timeDomainPlot, setTimeDomainPlot] = useState(null);
    const [frequencyDomainPlot, setFrequencyDomainPlot] = useState(null);
    
    // Function to handle slider sliding state
    const handleSliding = (sliding) => {
        setIsSliding(sliding);
        if (setParentIsSliding) setParentIsSliding(sliding);
    };
    
    // Wrap generateWaveformData in useCallback to prevent recreation on every render
    const generateWaveformData = useCallback(() => {
        // Convert parameters to base units for API call
        const chirpDurationSec = chirpDuration * 1e-6;
        const bandwidthHz = bandwidth * 1e6;
        const centerFreqHz = centerFreq * 1e9;
        const sampleRateHz = sampleRate * 1e6;
        
        // For local calculation (as fallback)
        const numSamples = Math.floor(chirpDurationSec * sampleRateHz);
        const timeStep = chirpDurationSec / numSamples;
        
        // Call the backend API to get the waveform data
        setIsLoading(true);
        
        fetch('http://localhost:8000/api/radar/waveform', {
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
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Use the Plotly data from the backend
            setTimeDomainPlot(data.timeDomainPlot);
            setFrequencyDomainPlot(data.frequencyDomainPlot);
            
            // Update derived parameters if provided
            if (data.derivedParams) {
                setDerivedParams(data.derivedParams);
            }
            
            setIsLoading(false);
        })
        .catch(error => {
            console.error('Error fetching waveform data:', error);
            setError('Failed to fetch waveform data from backend. Using local calculation instead.');
            
            // Fallback to local calculation if API call fails
            const timeData = Array.from({ length: numSamples }, (_, i) => i * timeStep * 1e6); // Convert to μs
            let freqData = [];
            
            // Generate frequency data based on waveform type
            switch (waveformType) {
                case 'linear':
                    freqData = timeData.map(t => 
                        (centerFreqHz - bandwidthHz/2) / 1e9 + 
                        (bandwidthHz * t) / (chirpDuration * 1e9)
                    );
                    break;
                case 'sawtooth':
                    freqData = timeData.map(t => {
                        const normalizedTime = t / chirpDuration;
                        return (centerFreqHz - bandwidthHz/2) / 1e9 + 
                               (bandwidthHz * normalizedTime) / 1e9;
                    });
                    break;
                case 'triangular':
                    freqData = timeData.map(t => {
                        const normalizedTime = t / chirpDuration;
                        const triangularPattern = normalizedTime <= 0.5 
                            ? normalizedTime * 2 
                            : 2 - normalizedTime * 2;
                        return (centerFreqHz - bandwidthHz/2) / 1e9 + 
                               (bandwidthHz * triangularPattern) / 1e9;
                    });
                    break;
                default:
                    freqData = timeData.map(t => 
                        (centerFreqHz - bandwidthHz/2) / 1e9 + 
                        (bandwidthHz * t) / (chirpDuration * 1e9)
                    );
            }
            
            setWaveformData({
                time: timeData,
                frequency: freqData
            });
            
            setIsLoading(false);
        });
    }, [bandwidth, chirpDuration, centerFreq, sampleRate, waveformType]);

    // Calculate derived parameters when input parameters change
    useEffect(() => {
        // Skip calculation during sliding for better performance
        if (isSliding) return;
        
        setLoading(true);
        setError(null);
        
        try {
            // Calculate derived parameters using the updated function
            const derivedParamsResult = calculateMetrics(
                bandwidth, 
                chirpDuration, 
                sampleRate, 
                centerFreq, 
                maxRange, 
                numChirps, 
                numRx, 
                numTx
            );
            
            setDerivedParams(derivedParamsResult);
            
            // Generate waveform data for plotting
            generateWaveformData();
        } catch (err) {
            console.error("Error calculating parameters:", err);
            setError("Error calculating parameters. Please check your inputs.");
        } finally {
            setLoading(false);
        }
    }, [bandwidth, chirpDuration, centerFreq, sampleRate, waveformType, numChirps, numRx, numTx, maxRange, isSliding, generateWaveformData]);
    
    // Memoize plot data to prevent unnecessary re-renders
    const plotData = useMemo(() => {
        if (!waveformData) return [];
        
        return [
            {
                x: waveformData.time,
                y: waveformData.frequency,
                type: 'scatter',
                mode: 'lines',
                name: 'Frequency vs Time',
                line: { color: '#1976d2', width: 2 }
            }
        ];
    }, [waveformData]);
    
    // Memoize plot layout to prevent unnecessary re-renders
    const plotLayout = useMemo(() => ({
        title: 'Radar Waveform',
        xaxis: {
            title: 'Time (μs)',
            showgrid: true,
            zeroline: true
        },
        yaxis: {
            title: 'Frequency (GHz)',
            showgrid: true,
            zeroline: true
        },
        margin: { l: 50, r: 50, b: 50, t: 50 },
        autosize: true,
        showlegend: false
    }), []);
    
    // Memoize plot config to prevent unnecessary re-renders
    const plotConfig = useMemo(() => ({
        displayModeBar: true,
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    }), []);
    
    return (
        <Container maxWidth="lg">
            <Paper elevation={3} sx={{ p: 3, my: 2 }}>
                <Typography variant="h5" gutterBottom>
                    Radar Waveform Generator
                </Typography>
                
                <Grid container spacing={3}>
                    {/* Left column - Parameters */}
                    <Grid item xs={12} md={4}>
                        <ParameterControls
                            bandwidth={bandwidth}
                            setBandwidth={setBandwidth}
                            chirpDuration={chirpDuration}
                            setChirpDuration={setChirpDuration}
                            centerFreq={centerFreq}
                            setCenterFreq={setCenterFreq}
                            sampleRate={sampleRate}
                            setSampleRate={setSampleRate}
                            waveformType={waveformType}
                            setWaveformType={setWaveformType}
                            numChirps={numChirps}
                            setNumChirps={setNumChirps}
                            numRx={numRx}
                            setNumRx={setNumRx}
                            numTx={numTx}
                            setNumTx={setNumTx}
                            maxRange={maxRange}
                            setMaxRange={setMaxRange}
                            setIsSliding={handleSliding}
                        />
                    </Grid>
                    
                    {/* Right column - Visualization and Derived Parameters */}
                    <Grid item xs={12} md={8}>
                        {/* Waveform Visualization */}
                        <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Waveform Visualization
                            </Typography>
                            
                            <Box sx={{ height: 300, width: '100%' }}>
                                {loading ? (
                                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                                        <CircularProgress />
                                    </Box>
                                ) : (
                                    <MemoizedPlot
                                        data={plotData}
                                        layout={plotLayout}
                                        config={plotConfig}
                                        style={{ width: '100%', height: '100%' }}
                                        useResizeHandler={true}
                                    />
                                )}
                            </Box>
                        </Paper>
                        
                        {/* Derived Parameters */}
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Derived Parameters
                            </Typography>
                            
                            <DerivedParameters 
                                derivedParams={derivedParams}
                                loading={loading}
                                error={error}
                            />
                        </Paper>
                    </Grid>
                </Grid>
            </Paper>
        </Container>
    );
});

export default RadarWaveformGeneratorPy;