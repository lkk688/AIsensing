# AIRadar Communication Model G3: Technical Documentation

**Document Version:** 1.0  
**Last Updated:** 2026-01-19  
**Author:** AI Assistant

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Architecture Evolution](#2-model-architecture-evolution)
3. [Communication Model Details](#3-communication-model-details)
4. [Training Optimizations](#4-training-optimizations)
5. [Theoretical Limits and DL Performance](#5-theoretical-limits-and-dl-performance)
6. [Usage Guide](#6-usage-guide)
7. [Results and Analysis](#7-results-and-analysis)

---

## 1. Overview

The G3 communication model (`AIradar_comm_model_g3.py`) is a deep learning-based symbol detection system for ISAC (Integrated Sensing and Communication) radar systems. It supports multiple modulation schemes:

- **4-QAM (QPSK)**: 2 bits per symbol
- **8-QAM (Cross constellation)**: 3 bits per symbol  
- **16-QAM**: 4 bits per symbol

### Key Features

| Feature | Description |
|---------|-------------|
| **Modulation Support** | 4-QAM, 8-QAM, 16-QAM |
| **Channel Modes** | AWGN (clean) and Realistic (multipath + clutter + CSI error) |
| **Training Modes** | Mixed-QAM or Modulation-Specific |
| **Model Versions** | v1 (Basic MLP), v2 (Attention), v3 (AdaptiveCommNet) |

---

## 2. Model Architecture Evolution

### 2.1 v1: CommNetG3 (Basic MLP)

The original v1 architecture uses a simple per-pixel MLP approach:

```
Input: [B, 5, H, W] → (eq_real, eq_imag, H_mag, H_phase, snr)
       ↓
ConfigEncoder → [B, 64] conditioning vector
       ↓
MLP: Linear(69, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 6)
       ↓
Output: [B, 6, H, W] LLR (Log-Likelihood Ratios) per bit
```

**Parameters:** ~189,000

**Issues:**
- LLR-based approach struggles with higher-order modulations
- Simple MLP lacks spatial awareness
- BER floor at ~20% for 16-QAM

### 2.2 v2: CommNetG3V2 (Attention-Enhanced)

Enhanced architecture with modulation-aware attention:

```
Input: [B, 5, H, W]
       ↓
Input Projection: Linear(69, 512)
       ↓
Modulation-Aware Attention × 2 (8 heads)
       ↓
Residual Blocks × 3
       ↓
Output: [B, 6, H, W] LLR
```

**Parameters:** ~3,860,000

**Improvements:**
- Larger capacity (hidden_dim=512)
- Adaptive scaling based on modulation order
- Residual connections for better gradient flow

**Issues:**
- Still uses LLR approach → fundamental limitation
- Computationally expensive

### 2.3 v3: AdaptiveCommNet (RECOMMENDED)

The v3 model uses the proven G2C architecture with direct symbol classification:

```
Input: [B, 5, H, W] → (eq_real, eq_imag, H_mag, H_phase, snr)
       ↓
Shared Backbone:
├── Conv2d(5, 64, 3×3) + BN + FiLM + ReLU
├── Conv2d(64, 128, 3×3) + BN + FiLM + ReLU
├── Conv2d(128, 128, 3×3) + BN + FiLM + ReLU
└── Conv2d(128, 128, 3×3) + BN + ReLU
       ↓
Per-Modulation Adapters:
├── 4-QAM:  Conv layers → [B, 4, H, W]
├── 8-QAM:  Conv layers → [B, 8, H, W]
├── 16-QAM: Conv layers (deeper) → [B, 16, H, W]
└── 64-QAM: Conv layers (deeper) → [B, 64, H, W]
       ↓
Output: [B, mod_order, H, W] Symbol Logits
```

**Parameters:** ~1,177,000

**Key Advantages:**

1. **Direct Symbol Classification**: Outputs symbol logits instead of LLR
2. **Per-Modulation Adapters**: Specialized heads for each QAM order
3. **FiLM Conditioning**: Feature-wise linear modulation for config-awareness
4. **CrossEntropyLoss**: Proper multi-class classification loss

---

## 3. Communication Model Details

### 3.1 Input Preprocessing

The model receives ZF-equalized and constellation-normalized symbols:

```python
# Zero-Forcing Equalization
H_safe = np.where(np.abs(H_grid) > 1e-6, H_grid, 1e-6 + 0j)
eq_symbols = rx_grid / H_safe

# Constellation-Aware Normalization
scale_factors = {
    4:  np.sqrt(2),   # QPSK
    8:  np.sqrt(6),   # 8-QAM (cross)
    16: np.sqrt(10),  # 16-QAM
    64: np.sqrt(42),  # 64-QAM
}
eq_real = eq_symbols.real / scale_factor
eq_imag = eq_symbols.imag / scale_factor
```

### 3.2 FiLM Conditioning

Feature-wise Linear Modulation allows the model to adapt to different configurations:

```python
class FiLMLayer(nn.Module):
    def forward(self, x, cond):
        # cond: [B, cond_dim] → gamma, beta: [B, C, 1, 1]
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
```

### 3.3 Per-Modulation Adapters

Higher-order modulations have deeper adapter heads:

```python
# 4-QAM / 8-QAM Adapter
nn.Sequential(
    nn.Conv2d(128, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, mod_order, 1),  # 1×1 output
)

# 16-QAM / 64-QAM Adapter (Deeper)
nn.Sequential(
    nn.Conv2d(128, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, 128, 3, padding=1),  # Extra layer
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.Conv2d(128, mod_order, 1),
)
```

---

## 4. Training Optimizations

### 4.1 Label Smoothing

**Problem:** At high SNR, targets are very confident (one-hot). This can lead to overconfident predictions that cause errors.

**Solution:** Label smoothing softens the target distribution:

```python
# Default: label_smoothing = 0.1
loss = F.cross_entropy(output, target, label_smoothing=0.1)

# Effect: target [0, 0, 1, 0] → [0.025, 0.025, 0.925, 0.025]
```

**CLI:**
```bash
python AIradar_comm_model_g3.py --mode train_comm --label_smoothing 0.15
```

### 4.2 Mixed Channel Training

Training on both AWGN and Realistic channels improves generalization:

```python
# 50% AWGN (clean channel)
train_ds_awgn = G2DeepDataset(cfg_name, samples//2, enable_rf_impairments=False)

# 50% Realistic (multipath + clutter + CSI error)
train_ds_real = G2DeepDataset(cfg_name, samples//2, enable_rf_impairments=True)

train_ds = ConcatDataset([train_ds_awgn, train_ds_real])
```

### 4.3 Modulation-Specific Training

Mixed-QAM training struggles because different modulations have different decision boundaries. Solution: Train separate models per QAM:

```bash
# Train 4-QAM specific model
python AIradar_comm_model_g3.py --mode train_comm --qam_type 4QAM

# Train 8-QAM specific model
python AIradar_comm_model_g3.py --mode train_comm --qam_type 8QAM

# Train 16-QAM specific model
python AIradar_comm_model_g3.py --mode train_comm --qam_type 16QAM
```

**Checkpoints saved as:** `comm_best_4qam.pt`, `comm_best_8qam.pt`, `comm_best_16qam.pt`

### 4.4 Learning Rate and Scheduler

```python
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

### 4.5 Focal Loss (for LLR-based models)

Optional for v1/v2 models to focus on hard examples:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        # Reduces loss for easy examples, focuses on hard ones
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

---

## 5. Theoretical Limits and DL Performance

### 5.1 Why DL Can't Always Beat Traditional Methods

At **high SNR (25-30 dB)**, traditional Maximum Likelihood (ML) detection approaches Shannon's theoretical limit. The DL model faces fundamental challenges:

| SNR Regime | DL vs Traditional | Explanation |
|------------|------------------|-------------|
| **Low (0-10 dB)** | DL often better | Complex channel effects; DL learns patterns |
| **Medium (10-20 dB)** | Competitive | Both methods perform well |
| **High (25-30 dB)** | Traditional better | ML detection is near-optimal; very few errors to learn from |

### 5.2 The BER Floor Problem

**Observation:** DL model BER stops improving at ~1% even with increasing SNR.

**Causes:**
1. **Training Data Imbalance:** At high SNR, 99%+ of predictions are correct. The model doesn't see enough errors.
2. **Gradient Saturation:** CrossEntropyLoss on highly confident predictions has tiny gradients.
3. **Quantization Effects:** Finite precision models can't distinguish symbols separated by tiny margins.

### 5.3 Theoretical BER for AWGN Channel

For M-QAM with Gray coding over AWGN:

```
BER ≈ (4/log₂M) × (1 - 1/√M) × Q(√(3×Eb/N0×log₂M / (M-1)))
```

At SNR = 30 dB for 16-QAM:
- **Theoretical BER:** ~10⁻⁸
- **Practical Traditional:** ~0.9%
- **Best DL Achieved:** ~1.3%

### 5.4 Where DL Excels

DL is most valuable in scenarios with:

1. **Non-ideal Channels:** Multipath, fading, interference
2. **Imperfect CSI:** When channel estimation is noisy
3. **Hardware Impairments:** Phase noise, IQ imbalance, PA nonlinearity
4. **Low-to-Medium SNR:** Where traditional methods struggle

**Our Results Confirm This:**
- 16-QAM at SNR=15dB (Realistic): DL **66%** better than Traditional
- 16-QAM at SNR=30dB (Realistic): DL **31%** better
- 16-QAM at SNR=30dB (AWGN): Traditional 83% better

---

## 6. Usage Guide

### 6.1 Training Commands

```bash
# Recommended: v3 model with modulation-specific training
python AIradar_comm_model_g3.py \
    --mode train_comm \
    --qam_type 16QAM \
    --model_version v3 \
    --epochs 80 \
    --train_samples 500 \
    --label_smoothing 0.1

# Full training for all modulations
for qam in 4QAM 8QAM 16QAM; do
    python AIradar_comm_model_g3.py \
        --mode train_comm --qam_type $qam --epochs 80
done
```

### 6.2 Evaluation

```bash
# Comprehensive evaluation (all QAM types, both channels)
python AIradar_comm_model_g3.py --mode eval_comprehensive
```

### 6.3 Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_version` | `v3` | v1 (MLP), v2 (Attention), v3 (AdaptiveCommNet) |
| `--qam_type` | `all` | 4QAM, 8QAM, 16QAM, or all |
| `--epochs` | `30` | Training epochs |
| `--train_samples` | `300` | Samples per config |
| `--label_smoothing` | `0.1` | CrossEntropyLoss smoothing |
| `--high_snr_focus` | `False` | More high-SNR training samples |

---

## 7. Results and Analysis

### 7.1 16-QAM Performance Summary

| Metric | AWGN (Clean) | Realistic |
|--------|--------------|-----------|
| Best DL BER | 7.78×10⁻³ @ 30dB | 6.60×10⁻³ @ 30dB |
| DL beats Traditional | SNR ≤ 25dB | SNR ≤ 30dB |
| Max Improvement | 65.8% @ 15dB | 66.0% @ 15dB |

### 7.2 Model Comparison

| Model | Params | Training Time | 16-QAM BER Floor | Recommended Use |
|-------|--------|---------------|------------------|-----------------|
| v1 (MLP) | 189K | Fast | ~22% | Not recommended |
| v2 (Attention) | 3.8M | Slow | ~15% | Research only |
| v3 (Adaptive) | 1.2M | Medium | ~1% | **Production** |

### 7.3 Future Improvements

1. **Curriculum Learning:** Start with low SNR, gradually increase
2. **SNR-Aware Training:** Separate models for different SNR ranges
3. **Ensemble Methods:** Combine models for different regimes
4. **Transformer Architecture:** Self-attention over symbol grid
5. **Channel Coding Integration:** Joint detection and decoding

---

## Appendix A: File Structure

```
AIRadar/
├── AIradar_comm_model_g3.py      # Main training/evaluation script
├── AIradar_comm_model_g2c.py     # G2C models (AdaptiveCommNet source)
├── AIradar_comm_dataset_g2.py    # Dataset generation
└── data/AIradar_comm_model_g3/
    ├── comm_best_4qam.pt         # 4-QAM checkpoint
    ├── comm_best_8qam.pt         # 8-QAM checkpoint
    ├── comm_best_16qam.pt        # 16-QAM checkpoint
    └── eval/
        ├── evaluation_report.md  # Evaluation results
        └── ber_vs_snr_all_qam.png
```

---

*End of Document*
