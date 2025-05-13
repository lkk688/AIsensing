import React, { useState } from 'react';
import { 
  Grid, 
  Typography, 
  Box, 
  //Chip, 
  CircularProgress, 
  Alert, 
  Tooltip, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  Button 
} from '@mui/material';
//import { MathJax } from 'better-react-mathjax';
import 'katex/dist/katex.min.css';
import Latex from '@matejmazur/react-katex';
import DownloadIcon from '@mui/icons-material/Download';
import yaml from 'js-yaml';
/**
 * Component to display derived radar parameters
 */
const DerivedParameters = ({ derivedParams, loading, error, radarParams }) => {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedParam, setSelectedParam] = useState(null);
  
  const parameterInfo = {
    rangeResolution: {
      title: "Range Resolution",
      description: "The minimum distance between two targets that can be distinguished by the radar. Calculated as c/(2*bandwidth), where c is the speed of light.",
      unit: "m",
      latex: "Range\\ Resolution = \\frac{c}{2B}"
    },
    maxRange: {
      title: "Maximum Unambiguous Range",
      description: "The maximum distance at which a target can be detected without range ambiguity. Determined by the sample rate and bandwidth.",
      unit: "m",
      latex: "T_{chirp} = \\frac{2 \\cdot R_{max}}{c}"
    },
    velocityResolution: {
      title: "Velocity Resolution",
      description: "The minimum difference in velocity between two targets that can be distinguished. Calculated as λ/(2*T_frame), where λ is the wavelength and T_frame is the total frame time.",
      unit: "m/s",
      latex: "v_{resolution} = \\frac{\\lambda}{2 \\cdot N_{chirps} \\cdot T_{chirp}}"
    },
    maxVelocity: {
      title: "Maximum Unambiguous Velocity",
      description: "The maximum velocity that can be measured without ambiguity. Calculated as λ/(4*T_chirp), where T_chirp is the chirp duration.",
      unit: "m/s",
      latex: "v_{max} = \\frac{\\lambda}{4 \\cdot T_{chirp}}"
    },
    angularResolution: {
      title: "Angular Resolution",
      description: "The minimum angular separation between two targets that can be distinguished. Depends on the number of RX antennas and wavelength.",
      unit: "°",
      latex: "\\theta_{resolution} \\propto \\frac{1}{N_{RX}}"
    },
    chirpSlope: {
      title: "Sweep Slope",
      description: "The rate of frequency change during the chirp. Calculated as bandwidth/chirp_duration.",
      unit: "THz/s",
      latex: "Slope = \\frac{B}{T_{chirp}}"
    },
    samplesPerChirp: {
      title: "Samples Per Chirp",
      description: "The number of samples collected during each chirp. Calculated as sample_rate*chirp_duration.",
      unit: "",
      latex: "N_{samples} = f_s \\cdot T_{chirp}"
    },
    processingGain: {
      title: "Processing Gain",
      description: "The SNR improvement from range FFT processing. Calculated as 10*log10(samples_per_chirp).",
      unit: "dB",
      latex: "Processing\\ Gain = 10\\log_{10}(N_{samples})"
    },
    coherentIntegrationGain: {
      title: "Coherent Integration Gain",
      description: "The SNR improvement from Doppler FFT processing. Calculated as 10*log10(num_chirps).",
      unit: "dB",
      latex: "Coherent\\ Integration\\ Gain = 10\\log_{10}(N_{chirps})"
    },
    totalGain: {
      title: "Total Processing Gain",
      description: "The total SNR improvement from both range and Doppler processing.",
      unit: "dB",
      latex: "Total\\ Gain = 10\\log_{10}(N_{samples} \\cdot N_{chirps})"
    },
    wavelength: {
      title: "Wavelength",
      description: "The wavelength of the radar signal. Calculated as c/center_frequency.",
      unit: "mm",
      latex: "\\lambda = \\frac{c}{f_c}"
    },
    maxBeatFrequency: {
        title: "Maximum Beat Frequency",
        description: "The highest frequency component in the beat signal, corresponding to the maximum range. Calculated as 2*slope*max_range/c.",
        unit: "MHz",
        latex: "f_{beat,max} = \\frac{2 \\cdot Slope \\cdot R_{max}}{c}"
      },
      nyquistFrequency: {
        title: "Nyquist Frequency",
        description: "Half of the sampling frequency, representing the maximum frequency that can be accurately represented without aliasing.",
        unit: "MHz",
        latex: "f_{Nyquist} = \\frac{f_s}{2}"
      },
      rangeFftSize: {
        title: "Range FFT Size",
        description: "The size of the Fast Fourier Transform used for range processing, typically a power of 2 that is greater than or equal to the number of samples per chirp.",
        unit: "",
        latex: "N_{range,FFT} = 2^{\\lceil \\log_2(N_{samples}) \\rceil}"
      },
      dopplerFftSize: {
        title: "Doppler FFT Size",
        description: "The size of the Fast Fourier Transform used for Doppler processing, typically a power of 2 that is greater than or equal to the number of chirps.",
        unit: "",
        latex: "N_{doppler,FFT} = 2^{\\lceil \\log_2(N_{chirps}) \\rceil}"
      },
      frameDuration: {
        title: "Frame Duration",
        description: "The total time required to complete one frame of radar measurements, which includes all chirps in the frame.",
        unit: "ms",
        latex: "T_{frame} = N_{chirps} \\cdot T_{chirp}"
      },
      refreshRate: {
        title: "Refresh Rate",
        description: "The number of complete radar frames that can be processed per second, which determines how frequently the radar data is updated.",
        unit: "Hz",
        latex: "f_{refresh} = \\frac{1}{T_{frame}}"
      },
      frequencyWraparound: {
        title: "Frequency Wraparound",
        description: "Indicates whether the maximum beat frequency exceeds the Nyquist frequency, which would cause aliasing in the range measurements.",
        unit: "",
        latex: "Wraparound = \\begin{cases} True & f_{beat,max} > f_{Nyquist} \\\\ False & f_{beat,max} \\leq f_{Nyquist} \\end{cases}"
      }
    };
  
  const handleParamClick = (param) => {
    setSelectedParam(param);
    setDialogOpen(true);
  };
  
  const handleCloseDialog = () => {
    setDialogOpen(false);
  };

  // Function to handle exporting parameters to YAML
  const handleExportYAML = () => {
    // Combine radar parameters and derived parameters
    const exportData = {
      radarParameters: {
        ...radarParams
      },
      derivedParameters: {
        ...derivedParams
      },
      exportTimestamp: new Date().toISOString()
    };

    // Convert to YAML
    const yamlString = yaml.dump(exportData, {
      indent: 2,
      lineWidth: -1 // No line wrapping
    });

    // Create a blob and download link
    const blob = new Blob([yamlString], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `radar_parameters_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.yaml`;
    document.body.appendChild(link);
    link.click();
    
    // Clean up
    URL.revokeObjectURL(url);
    document.body.removeChild(link);
  };
  
  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }
  
  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }
  
  return (
    <>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6" component="h2">
          Derived Parameters
        </Typography>
        <Button 
          variant="contained" 
          color="primary" 
          size="small"
          startIcon={<DownloadIcon />}
          onClick={handleExportYAML}
        >
          Export Parameters
        </Button>
      </Box>
      
      <Grid container spacing={2} sx={{ maxHeight: '70vh', overflow: 'auto' }}>
        {Object.entries(derivedParams || {}).map(([key, value]) => {
          const info = parameterInfo[key] || { title: key, unit: "" };
          return (
            <Grid item xs={6} sm={4} md={3} key={key}>
              <Tooltip title="Click for details">
                <Box 
                  onClick={() => handleParamClick(key)}
                  sx={{ 
                    p: 1.5, 
                    border: '1px solid #e0e0e0', 
                    borderRadius: 1, 
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'rgba(0, 0, 0, 0.04)' }
                  }}
                >
                  <Typography variant="body2" color="text.secondary" gutterBottom noWrap>
                    {info.title}
                  </Typography>
                  <Typography variant="h6" sx={{ fontSize: '1.1rem' }}>
                    {typeof value === 'object' 
                      ? parseFloat(value.value || 0).toFixed(2) 
                      : typeof value === 'number' 
                        ? parseFloat(value).toFixed(2) 
                        : value} {info.unit}
                  </Typography>
                </Box>
              </Tooltip>
            </Grid>
          );
        })}
      </Grid>
      
      {selectedParam && (
        <Dialog open={dialogOpen} onClose={handleCloseDialog} maxWidth="sm" fullWidth>
          <DialogTitle>{parameterInfo[selectedParam]?.title || selectedParam}</DialogTitle>
          <DialogContent>
            <Typography variant="body1" paragraph>
              {parameterInfo[selectedParam]?.description || "No description available."}
            </Typography>
            {parameterInfo[selectedParam]?.latex && (
                <Box sx={{ my: 2, textAlign: 'center' }}>
                    <Latex>{parameterInfo[selectedParam].latex}</Latex>
                </Box>
                )}
            <Typography variant="body1" fontWeight="bold">
                Current Value: {
                    typeof derivedParams[selectedParam] === 'object' 
                    ? derivedParams[selectedParam].value.toFixed(2) 
                    : typeof derivedParams[selectedParam] === 'number'
                      ? derivedParams[selectedParam].toFixed(2)
                      : derivedParams[selectedParam]
                } {parameterInfo[selectedParam]?.unit || ""}
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseDialog}>Close</Button>
          </DialogActions>
        </Dialog>
      )}
    </>
  );
};

export default DerivedParameters;