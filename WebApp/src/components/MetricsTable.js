import React from 'react';

const MetricsTable = ({ metrics }) => {
  return (
    <div className="metrics-table">
      <h3>Radar Metrics</h3>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((metric, index) => (
            <tr key={index}>
              <td>{metric.name}</td>
              <td>{metric.value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MetricsTable;