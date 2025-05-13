import React, { useState } from 'react';
import { Table, Modal, Typography, Card } from 'antd';
import { MathJax } from 'better-react-mathjax';
import { InfoCircleOutlined } from '@ant-design/icons';

const { Title, Paragraph, Text } = Typography;

const RadarParameterTable = ({ derivedParams, loading, error }) => {
  const [selectedParameter, setSelectedParameter] = useState(null);
  const [modalVisible, setModalVisible] = useState(false);

  // Parameter explanations imported from radarCalculations.js
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

  // Format the value with appropriate units
  const formatValue = (key, value) => {
    if (typeof value !== 'number') return value;
    
    // Add appropriate units based on parameter type
    if (key.includes('Frequency')) return `${value.toFixed(2)} MHz`;
    if (key.includes('Duration')) return `${value.toFixed(2)} μs`;
    if (key.includes('Bandwidth')) return `${value.toFixed(2)} MHz`;
    if (key.includes('Rate') && !key.includes('Refresh')) return `${value.toFixed(2)} MHz`;
    if (key.includes('Refresh')) return `${value.toFixed(2)} Hz`;
    if (key.includes('Resolution') && key.includes('Range')) return `${value.toFixed(2)} m`;
    if (key.includes('Resolution') && key.includes('Velocity')) return `${value.toFixed(2)} m/s`;
    if (key.includes('Velocity') && !key.includes('Resolution')) return `${value.toFixed(2)} m/s`;
    if (key.includes('Range') && !key.includes('Resolution')) return `${value.toFixed(2)} m`;
    if (key.includes('Wavelength')) return `${(value * 1000).toFixed(2)} mm`;
    if (key.includes('Slope')) return `${(value / 1e12).toFixed(2)} THz/s`;
    if (key.includes('Sweep')) return `${value.toFixed(2)} MHz`;
    
    return value.toFixed(2);
  };

  // Prepare data for the table
  const tableData = derivedParams ? Object.entries(derivedParams).map(([key, value], index) => ({
    key: index,
    parameter: key,
    value: formatValue(key, value),
    description: PARAMETER_EXPLANATIONS[key]?.text || 'No description available'
  })) : [];

  // Table columns
  const columns = [
    {
      title: 'Parameter',
      dataIndex: 'parameter',
      key: 'parameter',
      sorter: (a, b) => a.parameter.localeCompare(b.parameter),
    },
    {
      title: 'Value',
      dataIndex: 'value',
      key: 'value',
    },
    {
      title: 'Info',
      key: 'info',
      width: 50,
      render: (_, record) => (
        <InfoCircleOutlined 
          style={{ cursor: 'pointer', color: '#1890ff' }}
          onClick={() => showParameterDetails(record.parameter)}
        />
      ),
    }
  ];

  // Show parameter details in modal
  const showParameterDetails = (paramName) => {
    setSelectedParameter(paramName);
    setModalVisible(true);
  };

  // Handle modal close
  const handleModalClose = () => {
    setModalVisible(false);
  };

  // Render related parameters section
  const renderRelatedParameters = (paramName) => {
    const contributes = PARAMETER_EXPLANATIONS[paramName]?.contributes || [];
    
    if (contributes.length === 0) return null;
    
    return (
      <div style={{ marginTop: '20px' }}>
        <Title level={4}>Related Parameters</Title>
        <ul>
          {contributes.map((param, index) => (
            <li key={index}>
              <Text strong>{param}</Text>
              <Text> - {PARAMETER_EXPLANATIONS[param]?.text || 'No description available'}</Text>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  if (loading) return <div>Loading parameters...</div>;
  if (error) return <div>Error loading parameters: {error}</div>;
  if (!derivedParams) return <div>No parameters available</div>;

  return (
    <div>
      <Card title="Radar Parameters" style={{ marginBottom: '20px' }}>
        <Table 
          columns={columns} 
          dataSource={tableData} 
          pagination={false}
          size="middle"
          bordered
          style={{ marginBottom: '20px' }}
          onRow={(record) => ({
            onClick: () => showParameterDetails(record.parameter),
            style: { cursor: 'pointer' }
          })}
        />
        <Text type="secondary">Click on any row or info icon to see detailed explanation</Text>
      </Card>

      <Modal
        title={selectedParameter}
        open={modalVisible}
        onCancel={handleModalClose}
        footer={null}
        width={700}
      >
        {selectedParameter && (
          <>
            <Paragraph>
              {PARAMETER_EXPLANATIONS[selectedParameter]?.text || 'No description available'}
            </Paragraph>
            
            {PARAMETER_EXPLANATIONS[selectedParameter]?.latex && (
              <Card style={{ marginTop: '16px', marginBottom: '16px', textAlign: 'center' }}>
                <MathJax>
                  {`\\[${PARAMETER_EXPLANATIONS[selectedParameter].latex}\\]`}
                </MathJax>
              </Card>
            )}
            
            {renderRelatedParameters(selectedParameter)}
          </>
        )}
      </Modal>
    </div>
  );
};

export default RadarParameterTable;