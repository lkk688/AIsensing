import React, { useState, useEffect, useRef, useMemo, memo } from 'react';
import {
    Box, Typography, Slider, Container, Paper, Grid,
    FormControl, InputLabel, Select, MenuItem, Button,
    CircularProgress
} from '@mui/material';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';
import Plot from 'react-plotly.js';

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
    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
                <CircularProgress size={24} />
            </Box>
        );
    }
    
    if (error) {
        return <Typography color="error">{error}</Typography>;
    }
    
    return (
        <>
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Chirp Slope: {derivedParams.chirpSlope?.value?.toFixed(2) || 0} {derivedParams.chirpSlope?.unit || 'THz/s'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Max Beat Frequency: {derivedParams.maxBeatFrequency?.value?.toFixed(2) || 0} {derivedParams.maxBeatFrequency?.unit || 'MHz'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Range Resolution: {derivedParams.rangeResolution?.value?.toFixed(2) || 0} {derivedParams.rangeResolution?.unit || 'm'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Maximum Unambiguous Range: {derivedParams.maxRange?.value?.toFixed(2) || 0} {derivedParams.maxRange?.unit || 'm'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Velocity Resolution: {derivedParams.velocityResolution?.value?.toFixed(2) || 0} {derivedParams.velocityResolution?.unit || 'm/s'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Maximum Unambiguous Velocity: {derivedParams.maxVelocity?.value?.toFixed(2) || 0} {derivedParams.maxVelocity?.unit || 'm/s'}
                </Typography>
            </Box>
            
            <Box sx={{ mb: 1 }}>
                <Typography variant="body2">
                    Wavelength: {derivedParams.wavelength?.value?.toFixed(2) || 0} {derivedParams.wavelength?.unit || 'mm'}
                </Typography>
            </Box>
        </>
    );
});

// Memoized component for parameter controls
const ParameterControls = memo(({ 
    bandwidth, setBandwidth,
    chirpDuration, setChirpDuration,
    centerFreq, setCenterFreq,
    sampleRate, setSampleRate,
    waveformType, setWaveformType,
    setIsSliding  // Add this prop
}) => (
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
                //onChange={(e, newValue) => setBandwidth(newValue)}
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
            />
        </Box>
        
        <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
                Chirp Duration: {chirpDuration} μs
            </Typography>
            <Slider
                value={chirpDuration}
                //onChange={(e, newValue) => setChirpDuration(newValue)}
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
            />
        </Box>
        
        <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
                Center Frequency: {centerFreq} GHz
            </Typography>
            <Slider
                value={centerFreq}
                //onChange={(e, newValue) => setCenterFreq(newValue)}
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
            />
        </Box>
        
        <Box sx={{ mb: 2 }}>
            <Typography gutterBottom>
                Sample Rate: {sampleRate} MHz
            </Typography>
            <Slider
                value={sampleRate}
                //onChange={(e, newValue) => setSampleRate(newValue)}
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
            />
        </Box>
    </Paper>
));

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

const RadarWaveformGenerator = () => {
    // State for radar parameters
    const [bandwidth, setBandwidth] = useState(100); // MHz
    const [chirpDuration, setChirpDuration] = useState(100); // μs
    const [centerFreq, setCenterFreq] = useState(77); // GHz
    const [sampleRate, setSampleRate] = useState(50); // MHz
    const [waveformType, setWaveformType] = useState('linear');
    
    // Add state to track if slider is being dragged
    const [isSliding, setIsSliding] = useState(false);

    // State for plots and derived parameters
    const [timeDomainPlot, setTimeDomainPlot] = useState(null);
    const [frequencyDomainPlot, setFrequencyDomainPlot] = useState(null);
    const [derivedParams, setDerivedParams] = useState({});
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    // Ref for tracking previous parameters to avoid unnecessary API calls
    const prevParamsRef = useRef({ bandwidth, chirpDuration, centerFreq, sampleRate, waveformType });
    
    // Debounce timer ref
    const timerRef = useRef(null);

    // Memoize the parameters object to use for dependency tracking
    const parameters = useMemo(() => ({
        bandwidth,
        chirpDuration,
        centerFreq,
        sampleRate,
        waveformType
    }), [bandwidth, chirpDuration, centerFreq, sampleRate, waveformType]);

    // Fetch data from backend when parameters change
    useEffect(() => {
        // Check if parameters have actually changed
        const prevParams = prevParamsRef.current;
        const hasChanged = Object.keys(parameters).some(key => parameters[key] !== prevParams[key]);
        
        // Don't make API calls if parameters haven't changed or if user is still sliding
        if (!hasChanged || isSliding) return;
        
        // Update the ref with current parameters
        prevParamsRef.current = { ...parameters };
        
        // Clear any existing timer
        if (timerRef.current) {
            clearTimeout(timerRef.current);
        }
        
        // Set loading state immediately for better UX
        setLoading(true);
        
        // Debounce the API call
        timerRef.current = setTimeout(async () => {
            setError(null);
            
            try {
                const response = await fetch('http://localhost:8000/api/radar/waveform', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(parameters),
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
        }, 300);
        
        return () => {
            if (timerRef.current) {
                clearTimeout(timerRef.current);
            }
        };
    }, [parameters, isSliding]);// Add isSliding to dependencies

    // Memoize plot configurations to prevent unnecessary re-renders
    const timeDomainConfig = useMemo(() => ({
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    }), []);
    
    const frequencyDomainConfig = useMemo(() => ({
        responsive: true,
        displayModeBar: true,
        displaylogo: false
    }), []);
    
    const plotStyle = useMemo(() => ({ 
        width: '100%', 
        height: '400px' 
    }), []);

    // Set a minimum width for the plots to ensure they're not too small
    const plotLayout = useMemo(() => ({
        autosize: true,
        height: 400,
        width: Math.max(window.innerWidth * 0.6, 600), // Set minimum width based on window size
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }), []);

    return (
        <Container maxWidth="xl" sx={{ mt: 4 }}>
            <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h4" gutterBottom>
                    FMCW Radar Waveform Generator
                </Typography>
                
                <Grid container spacing={3}>
                    {/* Parameter controls */}
                    <Grid item xs={12} md={3} sx={{ minWidth: '250px' }}>
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
                            setIsSliding={setIsSliding}  // Pass the new prop
                        />
                        
                        {/* Derived Parameters */}
                        <Paper elevation={2} sx={{ p: 2, mt: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Derived Parameters
                            </Typography>
                            
                            <DerivedParameters 
                                derivedParams={derivedParams}
                                loading={loading}
                                error={error}
                            />
                            
                            <RadarEquations />
                        </Paper>
                    </Grid>
                    
                    {/* Waveform visualizations */}
                    <Grid item xs={12} md={9} sx={{ minWidth: '600px' }}>
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
                                <MemoizedPlot
                                    data={timeDomainPlot.data}
                                    layout={{
                                        ...timeDomainPlot.layout,
                                        ...plotLayout
                                        //autosize: true,
                                        //height: 400
                                    }}
                                    config={timeDomainConfig}
                                    style={plotStyle}
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
                                <MemoizedPlot
                                    data={frequencyDomainPlot.data}
                                    layout={{
                                        ...frequencyDomainPlot.layout,
                                        ...plotLayout
                                        //autosize: true,
                                        //height: 400
                                    }}
                                    config={frequencyDomainConfig}
                                    style={plotStyle}
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