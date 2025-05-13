import React, { useState } from 'react';
import { Slider, Typography, Box, Tooltip } from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

/**
 * A reusable parameter slider component with label and info tooltip
 * 
 * @param {Object} props - Component props
 * @param {string} props.label - Label for the slider
 * @param {string} props.unit - Unit of measurement
 * @param {number} props.value - Current value
 * @param {Function} props.onChange - Function to call when value changes
 * @param {number} props.min - Minimum value
 * @param {number} props.max - Maximum value
 * @param {number} props.step - Step size
 * @param {string} props.info - Information to display in tooltip
 * @param {Function} props.setIsSliding - Function to set sliding state
 */
const ParameterSlider = ({ 
  label, 
  unit, 
  value, 
  onChange, 
  min, 
  max, 
  step = 1, 
  info,
  setIsSliding
}) => {
  const [localValue, setLocalValue] = useState(value);
  
  const handleChange = (event, newValue) => {
    setLocalValue(newValue);
  };
  
  const handleChangeCommitted = (event, newValue) => {
    onChange(newValue);
    if (setIsSliding) {
      setIsSliding(false);
    }
  };
  
  const handleSliderStart = () => {
    if (setIsSliding) {
      setIsSliding(true);
    }
  };
  
  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
        <Typography variant="body2" color="text.secondary" sx={{ flexGrow: 1 }}>
          {label}
        </Typography>
        <Typography variant="body2" fontWeight="medium">
          {localValue} {unit}
        </Typography>
        {info && (
          <Tooltip title={info} arrow placement="top">
            <InfoIcon fontSize="small" color="action" sx={{ ml: 0.5, fontSize: 16 }} />
          </Tooltip>
        )}
      </Box>
      <Slider
        value={localValue}
        onChange={handleChange}
        onChangeCommitted={handleChangeCommitted}
        onMouseDown={handleSliderStart}
        onTouchStart={handleSliderStart}
        min={min}
        max={max}
        step={step}
        valueLabelDisplay="auto"
        size="small"
      />
    </Box>
  );
};

export default ParameterSlider;