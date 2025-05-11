import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          AI Sensing
        </Typography>
        <Box>
          <Button color="inherit" component={RouterLink} to="/">
            Home
          </Button>
          <Button color="inherit" component={RouterLink} to="/radar-tuner">
            Radar Tuner
          </Button>
          <Button color="inherit" component={RouterLink} to="/waveform-generator">
            Waveform Generator
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;