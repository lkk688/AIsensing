import React from 'react';
import { Box, Typography, Container, Paper } from '@mui/material';

const Home = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Welcome to AI Sensing Radar Parameter Tuner
        </Typography>
        <Typography variant="body1" paragraph>
          This application helps you visualize and understand the relationships between 
          different FMCW radar parameters. Use the Radar Tuner page to experiment with 
          different settings and see how they affect radar performance.
        </Typography>
        <Typography variant="body1">
          Click on "Radar Tuner" in the navigation bar to get started.
        </Typography>
      </Paper>
    </Container>
  );
};

export default Home;