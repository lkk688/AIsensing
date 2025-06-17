import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import RadarParameterTuner from './components/RadarParameterTuner';
import RadarWaveformGenerator from './components/RadarWaveformGeneratorPy3';
import './App.css';
import RadarDataViewer from './components/RadarDataViewer';
import './components/RadarDataViewer.css';

// Create a theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/radar-tuner" element={<RadarParameterTuner />} />
          <Route path="/waveform-generator" element={<RadarWaveformGenerator />} />
          <Route path="/radar-data-viewer" element={<RadarDataViewer />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;