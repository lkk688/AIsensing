clear; close all; clc; format long;
M = 4;
txQPSKMod = comm.PSKModulator(M,PhaseOffset=pi/M,OutputDataType="double",SymbolMapping="Gray");

deltaF = 30e3;
fftlen = 128;
cplen = 16;
plutoSampRate = deltaF*fftlen; % total ofdm bw
numDataCarriers = 100;
SX_numGuardCarrier = ceil((fftlen-1 -numDataCarriers)/2); % -1 accounts for DC-null
DX_numGuardCarrier = floor((fftlen-1 -numDataCarriers)/2);
dataBW = numDataCarriers * deltaF;

txOfdmMod = comm.OFDMModulator( ...
    'FFTLength', fftlen, ...
    'NumGuardBandCarriers', [SX_numGuardCarrier; DX_numGuardCarrier], ...
    'InsertDCNull', true, ...
    'CyclicPrefixLength', cplen, ...
    'Windowing', true, ...
    'NumSymbols', 100, ...
    'NumTransmitAntennas', 1, ...
    'PilotInputPort', false);

ofdmInfo = info(txOfdmMod);
ofdmSize = ofdmInfo.DataInputSize;

txRowData = randi([0 1], [ofdmSize(1)*ofdmSize(2)*log2(M), 1]);
txSymbolData = bit2int(txRowData, log2(M));
txQPSKdata = txQPSKMod(txSymbolData);
ofdmInput = reshape(txQPSKdata, ofdmSize); %serie/parallelo
%pilotInput = ones(ofdmInfo.PilotInputSize);
%pilotInput = repmat([1;1;1;-1], 1, ofdmInfo.DataInputSize(2));
%tx_ofdm_wave = txOfdmMod(ofdmInput,pilotInput);
tx_ofdm_wave = txOfdmMod(ofdmInput);

spectrum = spectrumAnalyzer('SampleRate', plutoSampRate, "NumInputPorts",2);
%showResourceMapping(txOfdmMod);

%% pluto tx
plutoTx = sdrtx('Pluto'); %comm.SDRTxPluto(RadioID='ip:pluto2.local');
plutoTx.CenterFrequency = 1e9; % dopo il mixer abbiamo RF=2*LO + IF; LO=13Ghz, IF=2.55Ghz; RF=28.55Ghz
plutoTx.Gain = 0;
plutoTx.BasebandSampleRate = plutoSampRate; % default 1.0e6
plutoTx.ShowAdvancedProperties = true;
plutoTx.FrequencyCorrection = 0;
info(plutoTx)

%%
% Transmit waveform continously:
transmitRepeat(plutoTx, tx_ofdm_wave);

%% PLUTO RX

plutoRx = sdrrx('Pluto'); %comm.SDRRxPluto(RadioID='ip:pluto2.local');
plutoRx.ShowAdvancedProperties = true;
plutoRx.CenterFrequency = 1e9;
plutoRx.OutputDataType = "double";
plutoRx.SamplesPerFrame = txOfdmMod.NumSymbols * (fftlen + cplen); % better be > 4000 samples
plutoRx.GainSource = "Manual";
plutoRx.Gain = 0;
plutoRx.BasebandSampleRate = plutoSampRate;

tic
while toc < 30
    rxSig = plutoRx();
    spectrum(tx_ofdm_wave, rxSig);
end

%%
release(plutoTx);
release(plutoRx);