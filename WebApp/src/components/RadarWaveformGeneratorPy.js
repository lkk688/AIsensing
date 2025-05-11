import React, { useState, useEffect, useRef } from 'react';
import {
    Box, Typography, Slider, Container, Paper, Grid,
    FormControl, InputLabel, Select, MenuItem, Button,
    CircularProgress
} from '@mui/material';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';
import Plot from 'react-plotly.js';

const RadarWaveformGenerator = () => {
    // State for radar parameters
    const [bandwidth, setBandwidth] = useState(100); // MHz
    const [chirpDuration, setChirpDuration] = useState(100); // μs
    const [centerFreq, setCenterFreq] = useState(77); // GHz
    const [sampleRate, setSampleRate] = useState(50); // MHz
    const [waveformType, setWaveformType] = useState('linear');
    
    // State for plots and derived parameters
    const [timeDomainPlot, setTimeDomainPlot] = useState(null);
    const [frequencyDomainPlot, setFrequencyDomainPlot] = useState(null);
    const [derivedParams, setDerivedParams] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Fetch data from backend when parameters change
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            
            try {
                const response = await fetch('http://localhost:8000/api/radar/waveform', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        bandwidth,
                        chirpDuration,
                        centerFreq,
                        sampleRate,
                        waveformType
                    }),
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Update state with received data
                setTimeDomainPlot(data.timeDomainPlot);
                setFrequencyDomainPlot(data.frequencyDomainPlot);
                setDerivedParams(data.derivedParams);
            } catch (err) {
                console.error('Error fetching radar data:', err);
                setError('Failed to fetch radar data. Please try again.');
            } finally {
                setLoading(false);
            }
        };
        
        // Debounce the API call to prevent too many requests
        const timeoutId = setTimeout(() => {
            fetchData();
        }, 300);
        
        return () => clearTimeout(timeoutId);
    }, [bandwidth, chirpDuration, centerFreq, sampleRate, waveformType]);

    return (
        <Container maxWidth="xl" sx={{ mt: 4 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom>
                    FMCW Radar Waveform Generator
                </Typography>
                
                <Grid container spacing={3}>
                    {/* Parameter controls */}
                    <Grid item xs={12} md={3}>
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
                                    max={500}
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
                                    marks={[
                                        { value: 24, label: '24 GHz' },
                                        { value: 60, label: '60 GHz' },
                                        { value: 77, label: '77 GHz' }
                                    ]}
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
                                    max={200}
                                    step={10}
                                />
                            </Box>
                        </Paper>
                        
                        {/* Derived Parameters */}
                        <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Derived Parameters
                            </Typography>
                            
                            {loading ? (
                                <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                                    <CircularProgress size={24} />
                                </Box>
                            ) : error ? (
                                <Typography color="error">{error}</Typography>
                            ) : (
                                <>
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Chirp Slope: {derivedParams.slope?.toFixed(2) || 0} THz/s
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Max Beat Frequency: {derivedParams.maxBeatFreq?.toFixed(2) || 0} MHz
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Range Resolution: {derivedParams.rangeResolution?.value?.toFixed(2) || 0} m
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Maximum Unambiguous Range: {derivedParams.maxRange?.value?.toFixed(2) || 0} m
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Velocity Resolution: {derivedParams.velocityResolution?.value?.toFixed(2) || 0} m/s
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Maximum Unambiguous Velocity: {derivedParams.maxVelocity?.value?.toFixed(2) || 0} m/s
                                        </Typography>
                                    </Box>
                                    
                                    <Box sx={{ mb: 1 }}>
                                        <Typography variant="body2">
                                            Wavelength: {derivedParams.wavelength?.value?.toFixed(2) || 0} mm
                                        </Typography>
                                    </Box>
                                </>
                            )}
                            
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
                        </Paper>
                    </Grid>
                    
                    {/* Waveform visualizations */}
                    <Grid item xs={12} md={9}>
                        <Paper elevation={2} sx={{ p: 2, mb: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Time Domain Waveform
                            </Typography>
                            
                            {loading ? (
                                <Box sx={{ display: 'flex', justifyContent: 'center', height: 400, alignItems: 'center' }}>
                                    <CircularProgress />
                                </Box>
                            ) : error ? (
                                <Typography color="error">{error}</Typography>
                            ) : timeDomainPlot ? (
                                <Plot
                                    data={timeDomainPlot.data}
                                    layout={{
                                        ...timeDomainPlot.layout,
                                        width: 800,
                                        autosize: true,
                                        height: 400
                                    }}
                                    config={{ 
                                        responsive: true,
                                        displayModeBar: true,
                                        displaylogo: false
                                    }}
                                    style={{ width: '100%', height: '400px' }}
                                    useResizeHandler={true}
                                />
                            ) : (
                                <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Typography>No data available</Typography>
                                </Box>
                            )}
                        </Paper>
                        
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Frequency Domain Spectrum
                            </Typography>
                            
                            {loading ? (
                                <Box sx={{ display: 'flex', justifyContent: 'center', height: 400, alignItems: 'center' }}>
                                    <CircularProgress />
                                </Box>
                            ) : error ? (
                                <Typography color="error">{error}</Typography>
                            ) : frequencyDomainPlot ? (
                                <Plot
                                    data={frequencyDomainPlot.data}
                                    layout={{
                                        ...frequencyDomainPlot.layout,
                                        width: 800,
                                        autosize: true,
                                        height: 400
                                    }}
                                    config={{ responsive: true }}
                                    style={{ width: '100%', height: '400px' }}
                                    useResizeHandler={true}
                                />
                            ) : (
                                <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Typography>No data available</Typography>
                                </Box>
                            )}
                        </Paper>
                    </Grid>
                </Grid>
            </Paper>
        </Container>
    );
};

export default RadarWaveformGenerator;