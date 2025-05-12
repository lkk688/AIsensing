import React, { useState, useEffect, useRef, useMemo, memo } from 'react';
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
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Chirp Slope: {derivedParams.chirpSlope?.value?.toFixed(2) || 0} {derivedParams.chirpSlope?.unit || 'THz/s'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Chirp Slope")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Max Beat Frequency: {derivedParams.maxBeatFrequency?.value?.toFixed(2) || 0} {derivedParams.maxBeatFrequency?.unit || 'MHz'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Max Beat Frequency")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Range Resolution: {derivedParams.rangeResolution?.value?.toFixed(2) || 0} {derivedParams.rangeResolution?.unit || 'm'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Range Resolution")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Maximum Unambiguous Range: {derivedParams.maxRange?.value?.toFixed(2) || 0} {derivedParams.maxRange?.unit || 'm'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Maximum Unambiguous Range")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Velocity Resolution: {derivedParams.velocityResolution?.value?.toFixed(2) || 0} {derivedParams.velocityResolution?.unit || 'm/s'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Velocity Resolution")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Maximum Unambiguous Velocity: {derivedParams.maxVelocity?.value?.toFixed(2) || 0} {derivedParams.maxVelocity?.unit || 'm/s'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Maximum Unambiguous Velocity")}
                        >
                            Explain
                        </Button>
                    </Box>
                    
                    <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Typography variant="body2">
                            Wavelength: {derivedParams.wavelength?.value?.toFixed(2) || 0} {derivedParams.wavelength?.unit || 'mm'}
                        </Typography>
                        <Button 
                            size="small" 
                            variant="outlined" 
                            onClick={(e) => handleExplainClick(e, "Wavelength")}
                        >
                            Explain
                        </Button>
                    </Box>

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
    setIsSliding  // Add this prop
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
                            
                            <Box sx={{ height: 400, width: '100%' }}>
                                {loading ? (
                                    <Box sx={{ display: 'flex', justifyContent: 'center', height: '100%', alignItems: 'center' }}>
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
                                        }}
                                        config={timeDomainConfig}
                                        style={plotStyle}
                                        useResizeHandler={true}
                                    />
                                ) : (
                                    <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Typography>No data available</Typography>
                                    </Box>
                                )}
                            </Box>
                        </Paper>
                        
                        <Paper elevation={2} sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Frequency Domain Spectrum
                            </Typography>
                            
                            <Box sx={{ height: 400, width: '100%' }}>
                                {loading ? (
                                    <Box sx={{ display: 'flex', justifyContent: 'center', height: '100%', alignItems: 'center' }}>
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
                                        }}
                                        config={frequencyDomainConfig}
                                        style={plotStyle}
                                        useResizeHandler={true}
                                    />
                                ) : (
                                    <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                        <Typography>No data available</Typography>
                                    </Box>
                                )}
                            </Box>
                        </Paper>
                    </Grid>
                </Grid>
            </Paper>
        </Container>
    );
};

export default RadarWaveformGenerator;