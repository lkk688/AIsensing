from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from plotly.utils import PlotlyJSONEncoder

# Import the waveform generation function from our radar module
from radar.waveform import generate_waveform

app = FastAPI(title="FMCW Radar API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update with your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RadarParams(BaseModel):
    bandwidth: float  # MHz
    chirpDuration: float  # μs
    centerFreq: float  # GHz
    sampleRate: float  # MHz
    waveformType: str  # 'linear', 'sawtooth', or 'triangular'

class RadarResponse(BaseModel):
    timeDomainPlot: Dict[str, Any]
    frequencyDomainPlot: Dict[str, Any]
    derivedParams: Dict[str, Any]

@app.get("/")
def read_root():
    return {"message": "FMCW Radar API is running"}

@app.post("/api/radar/waveform", response_model=RadarResponse)
def radar_waveform(params: RadarParams):
    import datetime
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Debug: Print input parameters with timestamp
    print(f"[{current_time}] RADAR API CALL - Input Parameters:")
    print(f"  Bandwidth: {params.bandwidth} MHz")
    print(f"  Chirp Duration: {params.chirpDuration} μs")
    print(f"  Center Frequency: {params.centerFreq} GHz")
    print(f"  Sample Rate: {params.sampleRate} MHz")
    print(f"  Waveform Type: {params.waveformType}")
    
    try:
        # Call the waveform generation function from our module
        time_domain_plot, freq_domain_plot, derived_params = generate_waveform(
            bandwidth=params.bandwidth,
            chirp_duration=params.chirpDuration,
            center_freq=params.centerFreq,
            sample_rate=params.sampleRate,
            waveform_type=params.waveformType
        )
        
        # Debug: Print output summary with timestamp
        print(f"[{current_time}] RADAR API CALL - Output Summary:")
        print(f"  Time Domain Plot: {'Generated successfully' if time_domain_plot else 'Failed'}")
        print(f"  Frequency Domain Plot: {'Generated successfully' if freq_domain_plot else 'Failed'}")
        print(f"  Derived Parameters: {list(derived_params.keys()) if derived_params else 'None'}")
        
        # Return the response
        return {
            "timeDomainPlot": time_domain_plot,
            "frequencyDomainPlot": freq_domain_plot,
            "derivedParams": derived_params
        }
    except ValueError as e:
        error_msg = str(e)
        print(f"[{current_time}] RADAR API CALL - ValueError: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = str(e)
        print(f"[{current_time}] RADAR API CALL - Exception: {error_msg}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {error_msg}")