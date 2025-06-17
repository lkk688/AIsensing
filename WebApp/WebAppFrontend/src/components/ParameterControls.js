import React from 'react';
import { 
  Box, 
  Typography, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Paper 
} from '@mui/material';
import ParameterSlider from './ParameterSlider';

/**
 * Component for radar parameter controls
 */
const ParameterControls = ({
  bandwidth,
  setBandwidth,
  chirpDuration,
  setChirpDuration,
  centerFreq,
  setCenterFreq,
  sampleRate,
  setSampleRate,
  waveformType,
  setWaveformType,
  numChirps,
  setNumChirps,
  numRx,
  setNumRx,
  numTx,
  setNumTx,
  maxRange,
  setMaxRange,
  setIsSliding
}) => {
  return (
    <Paper elevation={2} sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Radar Parameters
      </Typography>
      
      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle2" gutterBottom>
          Waveform Configuration
        </Typography>
        
        <FormControl fullWidth size="small" sx={{ mb: 2 }}>
          <InputLabel id="waveform-type-label">Waveform Type</InputLabel>
          <Select
            labelId="waveform-type-label"
            id="waveform-type"
            value={waveformType}
            label="Waveform Type"
            onChange={(e) => setWaveformType(e.target.value)}
          >
            <MenuItem value="linear">Linear Chirp</MenuItem>
            <MenuItem value="nonlinear">Non-Linear Chirp</MenuItem>
            <MenuItem value="stepped">Stepped Frequency</MenuItem>
          </Select>
        </FormControl>
        
        <ParameterSlider
          label="Bandwidth"
          unit="MHz"
          value={bandwidth}
          onChange={setBandwidth}
          min={10}
          max={1000}
          step={5}
          info="Signal bandwidth affects range resolution"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="Chirp Duration"
          unit="μs"
          value={chirpDuration}
          onChange={setChirpDuration}
          min={1}
          max={500}
          step={1}
          info="Longer chirps provide better SNR but reduce max velocity"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="Center Frequency"
          unit="GHz"
          value={centerFreq}
          onChange={setCenterFreq}
          min={5}
          max={81}
          step={1}
          info="Higher frequencies provide better angular resolution"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="Sample Rate"
          unit="MHz"
          value={sampleRate}
          onChange={setSampleRate}
          min={5}
          max={500}
          step={5}
          info="Higher sample rates increase maximum unambiguous range"
          setIsSliding={setIsSliding}
        />
      </Box>
      
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          System Configuration
        </Typography>
        
        <ParameterSlider
          label="Number of Chirps"
          unit=""
          value={numChirps}
          onChange={setNumChirps}
          min={4}
          max={512}
          step={16}
          info="More chirps improve velocity resolution and SNR"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="RX Antennas"
          unit=""
          value={numRx}
          onChange={setNumRx}
          min={1}
          max={8}
          step={1}
          info="More RX antennas improve angular resolution"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="TX Antennas"
          unit=""
          value={numTx}
          onChange={setNumTx}
          min={1}
          max={4}
          step={1}
          info="Multiple TX antennas enable MIMO capabilities"
          setIsSliding={setIsSliding}
        />
        
        <ParameterSlider
          label="Maximum Range"
          unit="m"
          value={maxRange}
          onChange={setMaxRange}
          min={10}
          max={500}
          step={10}
          info="Target maximum detection range"
          setIsSliding={setIsSliding}
        />
      </Box>
    </Paper>
  );
};

export default ParameterControls;