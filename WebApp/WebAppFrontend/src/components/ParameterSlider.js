import React from 'react';

const ParameterSlider = ({ label, min, max, value, onChange }) => {
  return (
    <div className="parameter-slider">
      <h3>{label}</h3>
      <div className="slider-container">
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value))}
          className="slider"
        />
        <span className="slider-value">{value}</span>
      </div>
    </div>
  );
};

export default ParameterSlider;