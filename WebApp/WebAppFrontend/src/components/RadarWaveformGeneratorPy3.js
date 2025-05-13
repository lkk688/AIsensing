import { memo, useState, useEffect, useMemo } from "react";
import {
  Container,
  Paper,
  Typography,
  Box,
  CircularProgress,
} from "@mui/material";
import ParameterControls from "./ParameterControls";
import DerivedParameters from "./DerivedParameters";
//import WaveformVisualization from "./WaveformVisualization";
import WaveformVisualizationPlotly from './WaveformVisualizationPlotly';
import { generateWaveformData } from "../utils/radarWaveformGenerator";
import { calculateRadarParameters } from "../utils/radarCalculations";

const RadarWaveformGeneratorPy = memo(({ setIsSliding: setParentIsSliding }) => {
  // Radar parameters
  const [bandwidth, setBandwidth] = useState(100); // MHz
  const [chirpDuration, setChirpDuration] = useState(100); // μs
  const [centerFreq, setCenterFreq] = useState(77); // GHz
  const [sampleRate, setSampleRate] = useState(50); // MHz
  const [waveformType, setWaveformType] = useState("linear");
  const [numChirps, setNumChirps] = useState(128);
  const [numRx, setNumRx] = useState(4);
  const [numTx, setNumTx] = useState(1);
  const [maxRange, setMaxRange] = useState(100); // meters

  // Derived and waveform state
  const [derivedParams, setDerivedParams] = useState({});
  const [paramLoading, setParamLoading] = useState(false);
  const [waveformLoading, setWaveformLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isSliding, setIsSliding] = useState(false);
  const [timeDomainData, setTimeDomainData] = useState(null);
  const [frequencyDomainData, setFrequencyDomainData] = useState(null);

  // Group radar params for dependency tracking
  const radarParams = useMemo(() => ({
    bandwidth,
    chirpDuration,
    centerFreq,
    sampleRate,
    waveformType,
    numChirps,
    numRx,
    numTx,
    maxRange,
  }), [
    bandwidth, chirpDuration, centerFreq, sampleRate,
    waveformType, numChirps, numRx, numTx, maxRange
  ]);

  // Handle slider interaction
  const handleSliding = (sliding) => {
    setIsSliding(sliding);
    if (setParentIsSliding) setParentIsSliding(sliding);
  };

  // Effect to recalculate and regenerate waveform
  useEffect(() => {
    if (isSliding) return;

    const updateWaveform = async () => {
      setParamLoading(true);
      setWaveformLoading(true);
      setError(null);

      try {
        // Derived parameters
        const derived = calculateRadarParameters(
          radarParams.bandwidth,
          radarParams.chirpDuration,
          radarParams.sampleRate,
          radarParams.centerFreq,
          radarParams.maxRange,
          radarParams.numChirps,
          radarParams.numRx,
          radarParams.numTx
        );
        setDerivedParams(derived);

        // Waveform data
        console.log("Calling generateWaveformData with params:", JSON.stringify(radarParams, null, 2));
        const result = await generateWaveformData(
          radarParams
        );
        if (result) {
            if (result.error) {
              setError(result.error);
            } else {
              // Update these lines to match the property names from generateWaveformData
              setTimeDomainData(result.timeDomainPlot);
              setFrequencyDomainData(result.frequencyDomainPlot);
              setDerivedParams(result.derivedParams);
              setError(null); // Clear any previous errors
            }
            setWaveformLoading(false);
        }
        //// The generateWaveformData function directly updates several state variables through the setter functions,
        // so we don't need to use the returned result for updating state again.
        //The generateWaveformData function directly updates several state variables through the setter functions:
        console.log("Waveform data generated successfully:", {
            timeDomainData: timeDomainData ? "Present" : "Missing",
            frequencyDomainData: frequencyDomainData ? "Present" : "Missing",
            derivedParams: derivedParams ? Object.keys(derivedParams) : "Missing"
          });
      } catch (err) {
        console.error(err);
        setError("Error generating waveform. Please check your inputs.");
      } finally {
        setParamLoading(false);
      }
    };
    console.log("Radar Params:", radarParams);
    updateWaveform();
  }, [radarParams, isSliding]);

  return (
    <Container maxWidth="xl" sx={{ mt: 2 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Radar Waveform Generator
        </Typography>
        <Box sx={{ display: "flex", flexDirection: "column", height: "100%" }}>
          {/* Main Section */}
          <Box sx={{ display: "flex", mb: 3 }}>
            {/* Left: Parameter Controls */}
            <Box sx={{ width: "30%", minWidth: "250px", pr: 2 }}>
              <ParameterControls
                bandwidth={bandwidth}
                setBandwidth={setBandwidth}
                chirpDuration={chirpDuration}
                setChirpDuration={setChirpDuration}
                centerFreq={centerFreq}
                setCenterFreq={setCenterFreq}
                sampleRate={sampleRate}
                setSampleRate={setSampleRate}
                waveformType={waveformType}
                setWaveformType={setWaveformType}
                numChirps={numChirps}
                setNumChirps={setNumChirps}
                numRx={numRx}
                setNumRx={setNumRx}
                numTx={numTx}
                setNumTx={setNumTx}
                maxRange={maxRange}
                setMaxRange={setMaxRange}
                setIsSliding={handleSliding}
              />
            </Box>

            {/* Right: Visualization */}
            <Box sx={{ width: "70%", flexGrow: 1 }}>
              <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
                <Typography variant="h6" gutterBottom>
                  Waveform Visualization
                </Typography>
                <Box sx={{ height: 400, width: "100%" }}>
                  {waveformLoading ? (
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "100%",
                      }}
                    >
                      <CircularProgress />
                    </Box>
                  ) : (
                    <WaveformVisualizationPlotly
                      timeDomainData={timeDomainData}
                      frequencyDomainData={frequencyDomainData}
                      params={radarParams}
                    />
                  )}
                </Box>
              </Paper>
            </Box>
          </Box>

          {/* Bottom: Derived Parameters */}
          <Paper elevation={2} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Radar Performance Metrics
            </Typography>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Click on any parameter for detailed explanation
            </Typography>
            <DerivedParameters
              derivedParams={derivedParams}
              loading={paramLoading}
              error={error}
              DerivedParameters={radarParams}
            />
          </Paper>
        </Box>
      </Paper>
    </Container>
  );
});

export default RadarWaveformGeneratorPy;