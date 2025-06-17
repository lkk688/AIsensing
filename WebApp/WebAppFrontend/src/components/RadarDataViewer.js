import React, { useEffect, useState, useRef } from 'react';
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

const RadarDataViewer = () => {
  const [radarData, setRadarData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/radar');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setRadarData(data);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Failed to connect to radar data stream');
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Try to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
      };
      
      wsRef.current = ws;
    };
    
    connectWebSocket();
    
    // Cleanup on component unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Prepare chart data
  const chartData = {
    labels: radarData ? Array.from({ length: radarData.real.length }, (_, i) => i) : [],
    datasets: [
      {
        label: 'Real',
        data: radarData ? radarData.real : [],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Imaginary',
        data: radarData ? radarData.imag : [],
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Radar Signal Data',
      },
    },
    animation: false, // Disable animation for better performance
  };

  return (
    <div className="radar-data-viewer">
      <h2>Radar Data Stream</h2>
      <div className="connection-status">
        Status: {isConnected ? 
          <span className="connected">Connected</span> : 
          <span className="disconnected">Disconnected</span>
        }
      </div>
      
      {error && <div className="error-message">{error}</div>}
      
      {radarData && (
        <div className="data-info">
          <p>Data Length: {radarData.datalen}</p>
          <p>Timestamp: {new Date(radarData.timestamp * 1000).toLocaleTimeString()}</p>
        </div>
      )}
      
      <div className="chart-container">
        {radarData ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <p>Waiting for radar data...</p>
        )}
      </div>
    </div>
  );
};

export default RadarDataViewer;