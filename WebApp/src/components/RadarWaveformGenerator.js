import React, { useState, useEffect, useRef } from 'react';
import {
    Box, Typography, Slider, Container, Paper, Grid,
    FormControl, InputLabel, Select, MenuItem, Button
} from '@mui/material';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';

const RadarWaveformGenerator = () => {
    // Canvas refs for drawing
    const timeCanvasRef = useRef(null);
    const freqCanvasRef = useRef(null);

    // State for radar parameters
    const [bandwidth, setBandwidth] = useState(100); // MHz
    const [chirpDuration, setChirpDuration] = useState(100); // μs
    const [centerFreq, setCenterFreq] = useState(77); // GHz
    const [sampleRate, setSampleRate] = useState(50); // MHz
    const [waveformType, setWaveformType] = useState('linear');

    // Derived parameters
    const [slope, setSlope] = useState(0);
    const [maxBeatFreq, setMaxBeatFreq] = useState(0);

    // Calculate derived parameters when inputs change
    // Calculate derived parameters when inputs change
    useEffect(() => {
        // Convert to base units
        const bandwidthHz = bandwidth * 1e6;
        const chirpDurationSec = chirpDuration * 1e-6;

        // Calculate FMCW slope
        const calculatedSlope = bandwidthHz / chirpDurationSec;
        setSlope(calculatedSlope);

        // Calculate max beat frequency (assuming 100m max range)
        const c = 3e8; // Speed of light
        const maxRange = 100; // meters
        const calculatedMaxBeatFreq = (2 * calculatedSlope * maxRange) / c;
        setMaxBeatFreq(calculatedMaxBeatFreq);

        // Generate and draw waveforms
        generateWaveforms();
    }, [bandwidth, chirpDuration, centerFreq, sampleRate, waveformType]);

    // Function to generate and draw waveforms
    const generateWaveforms = () => {
        if (!timeCanvasRef.current || !freqCanvasRef.current) return;

        // Get canvas contexts
        const timeCtx = timeCanvasRef.current.getContext('2d');
        const freqCtx = freqCanvasRef.current.getContext('2d');

        // Clear canvases
        timeCtx.clearRect(0, 0, timeCanvasRef.current.width, timeCanvasRef.current.height);
        freqCtx.clearRect(0, 0, freqCanvasRef.current.width, freqCanvasRef.current.height);

        // Convert parameters to base units
        const bandwidthHz = bandwidth * 1e6;
        const chirpDurationSec = chirpDuration * 1e-6;
        const centerFreqHz = centerFreq * 1e9;

        // Draw time domain waveform
        drawTimeDomainWaveform(timeCtx, bandwidthHz, chirpDurationSec, centerFreqHz);

        // Draw frequency domain waveform
        drawFrequencyDomainWaveform(freqCtx, bandwidthHz, chirpDurationSec, centerFreqHz);
    };

    const drawTimeDomainWaveform = (ctx, bandwidthHz, chirpDurationSec, centerFreqHz) => {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const padding = 20;

        // Set up coordinate system
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw axis labels
        ctx.font = '12px Arial';
        ctx.fillStyle = '#000';
        ctx.fillText('Time (μs)', width / 2, height - 5);
        ctx.save();
        ctx.translate(10, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Amplitude', 0, 0);
        ctx.restore();

        // Draw time domain signal (real part of complex signal)
        ctx.beginPath();

        const startFreq = centerFreqHz - bandwidthHz / 2;
        const endFreq = centerFreqHz + bandwidthHz / 2;
        const slope = bandwidthHz / chirpDurationSec;

        // Scale factors
        const timeScale = (width - 2 * padding) / chirpDurationSec;
        const amplitudeScale = (height - 2 * padding) / 2;
        const midHeight = padding + amplitudeScale;

        // Number of samples to draw
        const numSamples = 1000;
        const dt = chirpDurationSec / numSamples;

        // Draw based on waveform type
        if (waveformType === 'linear') {
            // Linear chirp
            for (let i = 0; i <= numSamples; i++) {
                const t = i * dt;
                const x = padding + t * timeScale;

                // Calculate instantaneous phase
                const phase = 2 * Math.PI * (startFreq * t + (slope * t * t) / 2);

                // Real part of complex exponential (cosine)
                const amplitude = Math.cos(phase);
                const y = midHeight - amplitude * amplitudeScale * 0.8;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
        } else if (waveformType === 'sawtooth') {
            // Sawtooth chirp
            const numSaws = 3;
            const sawDuration = chirpDurationSec / numSaws;
            const samplesPerSaw = numSamples / numSaws;

            for (let saw = 0; saw < numSaws; saw++) {
                for (let i = 0; i <= samplesPerSaw; i++) {
                    const t = saw * sawDuration + (i / samplesPerSaw) * sawDuration;
                    const x = padding + t * timeScale;

                    // Calculate normalized time within this sawtooth
                    const normalizedT = i / samplesPerSaw;

                    // Calculate instantaneous phase for this segment
                    const instantFreq = startFreq + bandwidthHz * normalizedT;
                    const phase = 2 * Math.PI * (startFreq * t + (slope * normalizedT * normalizedT * sawDuration) / 2);

                    // Real part of complex exponential (cosine)
                    const amplitude = Math.cos(phase);
                    const y = midHeight - amplitude * amplitudeScale * 0.8;

                    if (i === 0 && saw === 0) {
                        ctx.moveTo(x, y);
                    } else if (i === 0) {
                        // Phase discontinuity at sawtooth boundaries
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
            }
        } else if (waveformType === 'triangular') {
            // Triangular chirp
            const halfDuration = chirpDurationSec / 2;
            const halfSamples = numSamples / 2;

            // Up chirp
            for (let i = 0; i <= halfSamples; i++) {
                const t = (i / halfSamples) * halfDuration;
                const x = padding + t * timeScale;
                const normalizedT = i / halfSamples;

                // Calculate instantaneous phase for up-chirp
                const phase = 2 * Math.PI * (startFreq * t + (slope * t * t) / 2);

                // Real part of complex exponential (cosine)
                const amplitude = Math.cos(phase);
                const y = midHeight - amplitude * amplitudeScale * 0.8;

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            // Down chirp
            for (let i = 0; i <= halfSamples; i++) {
                const t = halfDuration + (i / halfSamples) * halfDuration;
                const x = padding + t * timeScale;
                const normalizedT = 1 - (i / halfSamples);

                // Calculate instantaneous phase for down-chirp
                const downChirpStartFreq = endFreq;
                const downChirpSlope = -slope;
                const relativeT = t - halfDuration;
                const phase = 2 * Math.PI * (downChirpStartFreq * relativeT + (downChirpSlope * relativeT * relativeT) / 2);

                // Real part of complex exponential (cosine)
                const amplitude = Math.cos(phase);
                const y = midHeight - amplitude * amplitudeScale * 0.8;

                ctx.lineTo(x, y);
            }
        }

        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw frequency vs time curve (to show the chirp pattern)
        ctx.beginPath();

        // Calculate the actual instantaneous frequency over time
        const drawInstantaneousFrequency = () => {
            if (waveformType === 'linear') {
                // Linear chirp frequency curve
                for (let i = 0; i <= 200; i++) {
                    const t = (i / 200) * chirpDurationSec;
                    const x = padding + t * timeScale;

                    // Calculate instantaneous frequency (Hz)
                    const instantFreq = startFreq + slope * t;

                    // Normalize to display range
                    const normalizedFreq = (instantFreq - startFreq) / bandwidthHz;
                    const y = height - padding - normalizedFreq * amplitudeScale * 2;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
            } else if (waveformType === 'sawtooth') {
                // Sawtooth chirp frequency curve
                const numSaws = 3;
                const sawDuration = chirpDurationSec / numSaws;

                for (let saw = 0; saw < numSaws; saw++) {
                    for (let i = 0; i <= 100; i++) {
                        const t = saw * sawDuration + (i / 100) * sawDuration;
                        const x = padding + t * timeScale;
                        const normalizedT = (i / 100);
                        const instantFreq = startFreq + (bandwidthHz * normalizedT);
                        const normalizedFreq = (instantFreq - startFreq) / bandwidthHz;
                        const y = height - padding - normalizedFreq * amplitudeScale * 2;

                        if (i === 0 && saw === 0) {
                            ctx.moveTo(x, y);
                        } else if (i === 0) {
                            ctx.moveTo(x, height - padding - 0);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                }
            } else if (waveformType === 'triangular') {
                // Triangular chirp frequency curve
                const halfDuration = chirpDurationSec / 2;

                // Up chirp
                for (let i = 0; i <= 100; i++) {
                    const t = (i / 100) * halfDuration;
                    const x = padding + t * timeScale;
                    const normalizedT = (i / 100);
                    const instantFreq = startFreq + (bandwidthHz * normalizedT);
                    const normalizedFreq = (instantFreq - startFreq) / bandwidthHz;
                    const y = height - padding - normalizedFreq * amplitudeScale * 2;

                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }

                // Down chirp
                for (let i = 0; i <= 100; i++) {
                    const t = halfDuration + (i / 100) * halfDuration;
                    const x = padding + t * timeScale;
                    const normalizedT = 1 - (i / 100);
                    const instantFreq = startFreq + (bandwidthHz * normalizedT);
                    const normalizedFreq = (instantFreq - startFreq) / bandwidthHz;
                    const y = height - padding - normalizedFreq * amplitudeScale * 2;

                    ctx.lineTo(x, y);
                }
            }
        };

        // Draw the instantaneous frequency
        drawInstantaneousFrequency();

        ctx.strokeStyle = '#dc004e';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Add legend
        ctx.font = '10px Arial';
        ctx.fillStyle = '#1976d2';
        ctx.fillText('Signal Amplitude', width - 120, padding + 15);
        ctx.fillStyle = '#dc004e';
        ctx.fillText('Frequency vs Time', width - 120, padding + 30);
    };

    // Function to draw frequency domain waveform
    const drawFrequencyDomainWaveform = (ctx, bandwidthHz, chirpDurationSec, centerFreqHz) => {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const padding = 20;

        // Set up coordinate system
        ctx.beginPath();
        ctx.moveTo(padding, padding);
        ctx.lineTo(padding, height - padding);
        ctx.lineTo(width - padding, height - padding);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw axis labels
        ctx.font = '12px Arial';
        ctx.fillStyle = '#000';
        ctx.fillText('Frequency (MHz)', width / 2, height - 5);
        ctx.save();
        ctx.translate(10, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Magnitude (dB)', 0, 0);
        ctx.restore();

        // Calculate frequency spectrum
        const startFreq = centerFreqHz - bandwidthHz / 2;
        const endFreq = centerFreqHz + bandwidthHz / 2;
        const freqStep = bandwidthHz / 200;

        // Draw spectrum
        ctx.beginPath();

        // Scale factors
        const freqScale = (width - 2 * padding) / bandwidthHz;

        // Calculate frequency range to display (show a bit more than the bandwidth)
        const displayBandwidth = bandwidthHz * 1.5;
        const displayStartFreq = centerFreqHz - displayBandwidth / 2;
        const displayEndFreq = centerFreqHz + displayBandwidth / 2;

        // Simulate spectrum based on waveform type
        for (let f = 0; f <= displayBandwidth; f += freqStep) {
            const freq = displayStartFreq + f;
            const x = padding + ((freq - displayStartFreq) / displayBandwidth) * (width - 2 * padding);

            // Calculate magnitude based on waveform type and frequency
            let magnitude = 0;

            // Check if frequency is within the chirp bandwidth
            const normalizedFreq = (freq - startFreq) / bandwidthHz;

            if (waveformType === 'linear') {
                // Linear chirp has relatively flat spectrum within bandwidth
                if (freq >= startFreq && freq <= endFreq) {
                    // Main lobe
                    magnitude = 0.9;

                    // Add some rolloff at the edges
                    if (normalizedFreq < 0.05 || normalizedFreq > 0.95) {
                        magnitude *= 0.8;
                    }
                } else {
                    // Side lobes
                    const distanceFromBand = Math.min(
                        Math.abs(freq - startFreq),
                        Math.abs(freq - endFreq)
                    ) / bandwidthHz;

                    magnitude = Math.max(0.1, 0.3 * Math.exp(-5 * distanceFromBand));
                }
            } else if (waveformType === 'sawtooth') {
                // Sawtooth has harmonics and more side lobes
                if (freq >= startFreq && freq <= endFreq) {
                    // Main lobe
                    magnitude = 0.85;

                    // Add harmonics
                    const harmonicSpacing = bandwidthHz / 3;
                    if (Math.abs((freq - startFreq) % harmonicSpacing) < freqStep * 5) {
                        magnitude = 0.95;
                    }
                } else {
                    // Side lobes with harmonics
                    const distanceFromBand = Math.min(
                        Math.abs(freq - startFreq),
                        Math.abs(freq - endFreq)
                    ) / bandwidthHz;

                    magnitude = Math.max(0.15, 0.4 * Math.exp(-3 * distanceFromBand));

                    // Add harmonics outside the band
                    const harmonicSpacing = bandwidthHz / 3;
                    const distFromHarmonic = Math.min(
                        Math.abs(((freq - startFreq) % harmonicSpacing) / harmonicSpacing),
                        Math.abs(1 - ((freq - startFreq) % harmonicSpacing) / harmonicSpacing)
                    );

                    if (distFromHarmonic < 0.05) {
                        magnitude += 0.2;
                    }
                }
            } else if (waveformType === 'triangular') {
                // Triangular has smoother spectrum with less side lobes
                if (freq >= startFreq && freq <= endFreq) {
                    // Main lobe with smooth shape
                    magnitude = 0.9 * (1 - 0.3 * Math.pow(2 * normalizedFreq - 1, 2));
                } else {
                    // Smoother side lobes
                    const distanceFromBand = Math.min(
                        Math.abs(freq - startFreq),
                        Math.abs(freq - endFreq)
                    ) / bandwidthHz;

                    magnitude = Math.max(0.05, 0.25 * Math.exp(-8 * distanceFromBand));
                }
            }

            // Add some noise
            magnitude += 0.02 * (Math.random() - 0.5);

            // Convert to dB scale (0 to -60 dB)
            const dbMagnitude = Math.max(-60, 20 * Math.log10(magnitude));

            // Map to canvas coordinates (0 dB at top, -60 dB at bottom)
            const y = padding + (dbMagnitude / -60) * (height - 2 * padding);

            if (f === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.strokeStyle = '#dc004e';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw center frequency marker
        const centerX = padding + ((centerFreqHz - displayStartFreq) / displayBandwidth) * (width - 2 * padding);
        ctx.beginPath();
        ctx.moveTo(centerX, height - padding);
        ctx.lineTo(centerX, padding);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label center frequency
        ctx.font = '10px Arial';
        ctx.fillStyle = '#000';
        ctx.fillText(`fc = ${centerFreq} GHz`, centerX - 30, padding + 15);

        // Draw bandwidth markers
        const startX = padding + ((startFreq - displayStartFreq) / displayBandwidth) * (width - 2 * padding);
        const endX = padding + ((endFreq - displayStartFreq) / displayBandwidth) * (width - 2 * padding);

        ctx.beginPath();
        ctx.moveTo(startX, height - padding - 5);
        ctx.lineTo(startX, height - padding + 5);
        ctx.moveTo(endX, height - padding - 5);
        ctx.lineTo(endX, height - padding + 5);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Label bandwidth
        ctx.fillText(`B = ${bandwidth} MHz`, (startX + endX) / 2 - 30, height - padding - 10);
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom>
                    FMCW Radar Waveform Generator
                </Typography>

                <Grid container spacing={3}>
                    {/* Parameter controls */}
                    <Grid item xs={12} md={4}>
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Waveform Parameters
                            </Typography>

                            <Box sx={{ mb: 2 }}>
                                <Typography gutterBottom>
                                    Waveform Type
                                </Typography>
                                <FormControl fullWidth>
                                    <Select
                                        value={waveformType}
                                        onChange={(e) => setWaveformType(e.target.value)}
                                    >
                                        <MenuItem value="linear">Linear Chirp</MenuItem>
                                        <MenuItem value="sawtooth">Sawtooth Chirp</MenuItem>
                                        <MenuItem value="triangular">Triangular Chirp</MenuItem>
                                    </Select>
                                </FormControl>
                            </Box>

                            <Box sx={{ mb: 2 }}>
                                <Typography gutterBottom>
                                    Bandwidth: {bandwidth} MHz
                                </Typography>
                                <Slider
                                    value={bandwidth}
                                    onChange={(e, newValue) => setBandwidth(newValue)}
                                    min={10}
                                    max={500}
                                    step={10}
                                />
                            </Box>

                            <Box sx={{ mb: 2 }}>
                                <Typography gutterBottom>
                                    Chirp Duration: {chirpDuration} μs
                                </Typography>
                                <Slider
                                    value={chirpDuration}
                                    onChange={(e, newValue) => setChirpDuration(newValue)}
                                    min={10}
                                    max={1000}
                                    step={10}
                                />
                            </Box>

                            <Box sx={{ mb: 2 }}>
                                <Typography gutterBottom>
                                    Center Frequency: {centerFreq} GHz
                                </Typography>
                                <Slider
                                    value={centerFreq}
                                    onChange={(e, newValue) => setCenterFreq(newValue)}
                                    min={24}
                                    max={81}
                                    step={1}
                                />
                            </Box>

                            <Box sx={{ mb: 2 }}>
                                <Typography gutterBottom>
                                    Sample Rate: {sampleRate} MHz
                                </Typography>
                                <Slider
                                    value={sampleRate}
                                    onChange={(e, newValue) => setSampleRate(newValue)}
                                    min={10}
                                    max={100}
                                    step={5}
                                />
                            </Box>
                        </Paper>

                        <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Derived Parameters
                            </Typography>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Chirp Slope: {(slope / 1e12).toFixed(2)} THz/s
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Max Beat Frequency: {(maxBeatFreq / 1e6).toFixed(2)} MHz
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Range Resolution: {(3e8 / (2 * bandwidth * 1e6)).toFixed(2)} m
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Maximum Unambiguous Range: {((3e8 * sampleRate * 1e6) / (2 * bandwidth * 1e6 * bandwidth * 1e6)).toFixed(2)} m
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Velocity Resolution: {((3e8 * 1000) / (4 * centerFreq * 1e9 * chirpDuration * 1e-6)).toFixed(2)} m/s
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Maximum Unambiguous Velocity: {((3e8 * 1000) / (4 * centerFreq * 1e9 * (chirpDuration * 1e-6 / 100))).toFixed(2)} m/s
                                </Typography>
                            </Box>

                            <Box sx={{ mb: 1 }}>
                                <Typography variant="body2">
                                    Wavelength: {((3e8) / (centerFreq * 1e9) * 1000).toFixed(2)} mm
                                </Typography>
                            </Box>

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
                        </Paper>
                    </Grid>

                    {/* Waveform visualizations */}
                    <Grid item xs={12} md={8}>
                        <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Time-Domain Waveform
                            </Typography>
                            <Box sx={{ width: '100%', height: '300px', position: 'relative' }}>
                                <canvas
                                    ref={timeCanvasRef}
                                    width={700}
                                    height={300}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </Box>
                        </Paper>

                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Frequency-Domain Spectrum
                            </Typography>
                            <Box sx={{ width: '100%', height: '300px', position: 'relative' }}>
                                <canvas
                                    ref={freqCanvasRef}
                                    width={700}
                                    height={300}
                                    style={{ width: '100%', height: '100%' }}
                                />
                            </Box>
                        </Paper>
                    </Grid>
                </Grid>
            </Paper>
        </Container>
    );
};

export default RadarWaveformGenerator;