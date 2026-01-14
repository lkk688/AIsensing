# AIRadar Communication Dataset G2/G3 Documentation

## Overview

The `AIradar_comm_dataset_g2.py` implements an enhanced joint radar-communication (ISAC) dataset generator with realistic 5G-like features. This document summarizes all G2/G3 enhancements, performance characteristics, and analysis.

---

## G2 Core Features

### 1. Adaptive CFAR Detection
- **Dynamic threshold adjustment** based on estimated SNR
- Higher SNR → lower threshold (better detection)
- Lower SNR → higher threshold (reduce false alarms)

### 2. Realistic Clutter Modeling
- **K-distributed ground clutter** (spiky returns from rough surfaces)
- **Weather clutter** (concentrated at low Doppler)
- Range-dependent power decay (R^-n law)
- Time-domain clutter injection for CNR tests

### 3. Imperfect CSI
- Models realistic channel estimation errors (10% default)
- Degrades equalization performance at high SNR

---

## G3 5G-Realistic Enhancements

### 1. DMRS Channel Estimation
- 5G NR-like Demodulation Reference Signals
- Zadoff-Chu sequences for constant amplitude
- MMSE estimation with interpolation
- **Trade-off**: 1/4 pilot density (realistic) vs 100% (ideal LS)

### 2. TDL Channel Models (3GPP TR 38.901)
| Model | Type | Delay Spread | Taps |
|-------|------|-------------|------|
| TDL-A | NLOS | 400 ns | 7 |
| TDL-B | NLOS | 130 ns | 5 |
| TDL-D | LOS | 500 ns | 5 |
| TDL-E | LOS | 500 ns | 5 |

### 3. FEC Coding (Simplified LDPC)
- Repetition code (R=1/3) as baseline
- Soft decoding with LLR combining
- **Coding gain**: ~6dB at SNR=0dB, ~13dB at SNR=4dB

### 4. Radar ROC Curve Analysis
- Pd vs Pfa curves at various thresholds
- Multi-SNR comparison
- RCS-based evaluation (vehicle/bicycle/pedestrian)
- CNR-based evaluation (clutter effects)

---

## Performance Analysis

### Communication BER vs SNR

#### Why BER is High at Low SNR (16-QAM):

| SNR (dB) | Observed BER | Explanation |
|----------|-------------|-------------|
| 0 | ~0.85 | 16-QAM needs ~20dB for low BER |
| 5 | ~0.71 | Still very low SNR for 16-QAM |
| 10 | ~0.46 | Starting to improve |
| 15 | ~0.22 | Multipath effects visible |
| 20 | ~0.06 | Reasonable for 16-QAM |
| 25 | ~0.02 | Approaching target |
| 30 | ~0.01 | Near error-free |

**Key factors affecting BER:**
1. **Modulation order**: 16-QAM requires ~20dB SNR for BER < 1%
2. **Multipath fading**: Causes symbol spreading and ISI
3. **Imperfect CSI**: 10% estimation error degrades equalization
4. **No FEC**: Raw uncoded symbol errors

#### Improving Low-SNR Performance:
- Use 4-QAM (QPSK) instead of 16-QAM
- Enable FEC coding (repetition or LDPC)
- Use DMRS with better interpolation
- Disable imperfect CSI simulation

---

### Radar Performance

#### TRADITIONAL vs OTFS Mode:

| Metric | TRADITIONAL | OTFS |
|--------|-------------|------|
| Range Error | ~0.03 m | ~2.0 m |
| F1 Score | 0.85-0.95 | 0.70-0.80 |
| Match Threshold | 3 m | 20 m |

**Why OTFS has lower performance:**
1. Delay-Doppler domain has coarser resolution
2. Different processing pipeline
3. Larger match threshold accommodates resolution

#### CNR Effects on Detection (Clutter-to-Noise Ratio):

| CNR (dB) | Peak F1 | CFAR Pd |
|----------|---------|---------|
| 0 | 0.98 | 1.00 |
| 20 | 0.16 | 1.00 |
| 40 | 0.04 | 1.00 |
| 50 | 0.04 | 0.96 |
| 60 | 0.03 | 0.88 |

**Key insight**: High clutter primarily increases false alarms (degrading F1), while Pd only drops at extreme CNR (>50dB) where clutter masks targets.

---

## Configuration Reference

### RADAR_COMM_CONFIGS_G2 Dictionary

```python
'CN0566_TRADITIONAL': {
    'mode': 'TRADITIONAL',
    'fc': 10.5e9,           # 10.5 GHz
    'radar_B': 500e6,       # 500 MHz bandwidth
    'mod_order': 16,        # 16-QAM
    'channel_model': 'multipath',
    'adaptive_cfar': True,
    'csi_error': 0.1        # 10% CSI error
}
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clutter_intensity` | 0.1 | Clutter power scaling |
| `fixed_snr` | None | Fixed SNR for evaluation |
| `enable_clutter` | True | Enable clutter models |
| `enable_imperfect_csi` | True | Enable CSI errors |
| `target_rcs_range` | (10, 30) | RCS range in dB |

---

## Evaluation Functions

### Multi-SNR Evaluation
```python
evaluate_multi_snr(config_names, snr_range, samples_per_snr)
```

### QAM Comparison
```python
evaluate_qam_comparison(base_config, qam_orders=[4, 16])
```

### FEC Coding Gain
```python
evaluate_fec_coding_gain(snr_range, code_rate=1/3)
```

### Radar ROC Curves
```python
evaluate_radar_roc(config_name, threshold_range, snr_db)
evaluate_roc_multi_snr(config_name, snr_list)
evaluate_roc_by_rcs(config_name, rcs_ranges)
evaluate_radar_by_cnr(config_name, cnr_list)
```

---

## Output Directory Structure

```
data/AIradar_comm_dataset_g2d/
├── multi_snr/           # SNR comparison plots
├── qam_comparison/      # 4-QAM vs 16-QAM
├── roc_curve/           # Radar ROC curves
├── roc_multi_snr/       # ROC at different SNR
├── roc_by_rcs/          # ROC by target RCS
├── roc_by_cnr/          # ROC by clutter level
├── fec_comparison/      # FEC coding gain
└── g3_ber_comparison/   # BER with G3 features
```

---

## Usage Example

```python
from AIradar_comm_dataset_g2 import (
    AIRadar_Comm_Dataset_G2,
    evaluate_multi_snr,
    evaluate_radar_by_cnr,
    evaluate_fec_coding_gain
)

# Create dataset
ds = AIRadar_Comm_Dataset_G2(
    config_name='CN0566_TRADITIONAL',
    num_samples=100,
    fixed_snr=20,
    enable_clutter=True,
    target_rcs_range=(0, 15)  # Weaker targets
)

# Evaluate
evaluate_multi_snr(['CN0566_TRADITIONAL'], snr_range=[0, 10, 20, 30])
evaluate_radar_by_cnr('CN0566_TRADITIONAL', cnr_list=[0, 20, 40, 60])
evaluate_fec_coding_gain(snr_range=[0, 4, 8, 12])
```
