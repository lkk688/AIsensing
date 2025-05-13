import React, { memo } from 'react';
import Plot from 'react-plotly.js';
import { Box, CircularProgress } from '@mui/material';

/**
 * WaveformVisualizationPlotly Component
 * 
 * This component renders Plotly figures that are returned from the backend as JSON.
 * It uses memoization to prevent unnecessary re-renders when props haven't changed.
 * 
 * @param {Object} props - Component props
 * @param {Object} props.figure - The Plotly figure object from backend (contains data, layout)
 * @param {Object} props.config - Optional Plotly configuration options
 * @param {Object} props.style - Optional styling for the Plot container
 * @param {boolean} props.useResizeHandler - Whether to automatically resize the plot on window resize
 * @param {string} props.className - Optional CSS class name
 */
// If your component expects figure props instead of data props
const WaveformVisualizationPlotly = ({ 
    timeDomainData, 
    frequencyDomainData,
    params = {},
    plotStyle = {},
  }) => {
    // Return loading state if data is not available
    if (!timeDomainData) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <CircularProgress />
        </Box>
      );
    }
  
    // Now timeDomainData and frequencyDomainData are the Plotly figure objects
    return (
      <Box sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column', ...plotStyle }}>
        <Box sx={{ height: '45%', minHeight: '350px', mb: 2 }}>
          <Plot
            data={timeDomainData.data}
            layout={{
              ...timeDomainData.layout,
              autosize: true,
              margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '100%' }}
          />
        </Box>
        
        {frequencyDomainData && (
          <Box sx={{ height: '45%', minHeight: '350px' }}>
            <Plot
              data={frequencyDomainData.data}
              layout={{
                ...frequencyDomainData.layout,
                autosize: true,
                margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '100%' }}
            />
          </Box>
        )}
      </Box>
    );
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

export default WaveformVisualizationPlotly;