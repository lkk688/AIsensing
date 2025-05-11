
This React-based web application provides an interactive interface for tuning FMCW (Frequency-Modulated Continuous Wave) radar parameters. It allows users to visualize and understand the relationships between different radar parameters and their effects on radar performance.

## Setup the Repo
```bash
git clone https://github.com/yourusername/AIsensing.git
cd AIsensing/WebApp
```


```bash
~/Developer/AIsensing/WebApp$ npm install
~/Developer/AIsensing/WebApp$ HOST=0.0.0.0 npm start
npm install @matejmazur/react-katex katex #significantly faster and lighter than MathJax 
npm install react-router-dom @mui/material @emotion/react @emotion/styled
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