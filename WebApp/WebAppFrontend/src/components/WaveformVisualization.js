import React from 'react';
import { Box, CircularProgress } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// // Expected timeDomainData structure
// {
//     x: [array of time values],
//     real: [array of real values],
//     imag: [array of imaginary values]
//   }
  
//   // Expected frequencyDomainData structure
//   {
//     x: [array of frequency values],
//     magnitude: [array of magnitude values]
//   }

const WaveformVisualization = ({ 
    timeDomainData, 
    frequencyDomainData,
    params = {},
    plotTitle = "Radar Waveform",
    plotStyle = {},
    showFrequencyDomain = true
  }) => {
    const { bandwidth, chirpDuration, centerFreq, sampleRate } = params;
    
    // Return loading state if data is not available
    if (!timeDomainData) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <CircularProgress />
        </Box>
      );
    }
  
    // Debug the data structure
    console.log("Time Domain Data:", timeDomainData);
    console.log("Time Domain Data x type:", Array.isArray(timeDomainData.x) ? "Array" : typeof timeDomainData.x);
    
    // Ensure labels (x values) are arrays
    const timeLabels = Array.isArray(timeDomainData.x) ? timeDomainData.x : 
                      (timeDomainData.x ? Object.values(timeDomainData.x) : []);
    const realValues = Array.isArray(timeDomainData.real) ? timeDomainData.real : 
                      (timeDomainData.real ? Object.values(timeDomainData.real) : []);
    const imagValues = Array.isArray(timeDomainData.imag) ? timeDomainData.imag : 
                      (timeDomainData.imag ? Object.values(timeDomainData.imag) : []);
    
    // Prepare time domain data for chart with array conversion
    const timeDomainChartData = {
      labels: timeLabels,
      datasets: [
        {
          label: 'Real',
          data: realValues,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1,
        },
        {
          label: 'Imaginary',
          data: imagValues,
          borderColor: 'rgb(255, 99, 132)',
          backgroundColor: 'rgba(255, 99, 132, 0.5)',
          tension: 0.1,
        },
      ],
    };
  
    // Prepare frequency domain data for chart with array conversion
    let frequencyDomainChartData = null;
    if (frequencyDomainData) {
      const freqLabels = Array.isArray(frequencyDomainData.x) ? frequencyDomainData.x : 
                        (frequencyDomainData.x ? Object.values(frequencyDomainData.x) : []);
      const magnitudeValues = Array.isArray(frequencyDomainData.magnitude) ? frequencyDomainData.magnitude : 
                             (frequencyDomainData.magnitude ? Object.values(frequencyDomainData.magnitude) : []);
      
      frequencyDomainChartData = {
        labels: freqLabels,
        datasets: [
          {
            label: 'Magnitude (dB)',
            data: magnitudeValues,
            borderColor: 'rgb(53, 162, 235)',
            backgroundColor: 'rgba(53, 162, 235, 0.5)',
            tension: 0.1,
          },
        ],
      };
    }
  
    // Chart options
    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: plotTitle,
        },
      },
      scales: {
        y: {
          beginAtZero: false,
        },
      },
    };
  
    return (
      <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', ...plotStyle }}>
        <Box sx={{ height: '45%', minHeight: '150px', mb: 2 }}>
          <Line options={chartOptions} data={timeDomainChartData} />
        </Box>
        
        {showFrequencyDomain && frequencyDomainChartData && (
          <Box sx={{ height: '45%', minHeight: '150px' }}>
            <Line 
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    ...chartOptions.plugins.title,
                    text: 'Frequency Domain',
                  },
                },
              }} 
              data={frequencyDomainChartData} 
            />
          </Box>
        )}
      </Box>
    );
  };
  
  export default WaveformVisualization;