# AIsensing WebApp

## Introduction
We developed a Radar UI application using PyQt (sdradi/radarappwdevice4.py), supporting both real-time data from physical radar devices and playback from recorded datasets. Building on this foundation, we redesigned the application as a modern web-based system, featuring a React frontend and a Python FastAPI backend. The backend handles all core functionalities, including radar parameter configuration, signal processing, and data delivery. The React frontend provides an intuitive and interactive interface for users to adjust radar parameters, view real-time waveforms, and examine detection results. The frontend communicates with the backend via RESTful APIs and WebSocket for real-time data updates, ensuring seamless data flow and responsive user interactions.

Specifically, we implemented two user interfaces:
    - Radar Parameter Tuning Interface – This interactive UI enables users to configure radar parameters, view derived performance metrics, and visualize waveform outputs. It helps users explore the relationships between parameter settings and radar system performance.
	- Real-Time Detection Interface – This UI displays real-time radar waveforms and detection results. It allows users to dynamically adjust radar settings and immediately observe their impact on detection performance, offering a clear and engaging visualization of the radar processing pipeline.

## Setup the Repo
```bash
git clone https://github.com/yourusername/AIsensing.git
cd AIsensing/WebApp
```


```bash
~/Developer/AIsensing/WebApp$ npm install
~/Developer/AIsensing/WebApp$ 
npm install @matejmazur/react-katex katex #significantly faster and lighter than MathJax 
npm install react-router-dom @mui/material @emotion/react @emotion/styled
npm install chart.js react-chartjs-2
npm install better-react-mathjax
npm install js-yaml @mui/icons-material
AIsensing/WebApp/WebAppFrontend$ npm install @mui/icons-material
HOST=0.0.0.0 npm start
```
Access the application:
- Local: http://localhost:3000
- Remote: http://YOUR_SERVER_IP:3000

Architecture:
- public/ : Contains static files like index.html
- src/ : Contains the React components and application logic
    - components/ : Reusable UI components
    - pages/ : Components that represent entire pages

Favicon generated from https://favicon.io/favicon-converter/

Python FastAPI server:
```bash
AIsensing/WebApp$ npm install react-plotly.js plotly.js
AIsensing/FastAPIbackend$ pip install -r requirements.txt
AIsensing/WebApp/FastAPIbackend$ uvicorn main:app --reload
```

## Key Files:
Entry Point (index.js)
- This is the standard entry point for a React application. It imports the main App component
    - Creates a root using ReactDOM
    - Renders the App component inside React.StrictMode (a tool for highlighting potential problems)

App Component (App.js)
- Sets up routing using react-router-dom
- Configures a Material-UI theme
- Defines the application structure with a Navbar and Routes

Navigation (Navbar.js)
- Uses Material-UI components (AppBar, Toolbar, etc.)
- Provides navigation links to different routes

Main Component (RadarParameterTuner.js)
- Uses React hooks (useState) for state management
- Implements interactive sliders for parameter tuning
- Calculates radar metrics based on user inputs
- Displays explanations with mathematical formulas using KaTeX