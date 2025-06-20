import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split, ConcatDataset
from AIRadarLib.pretrain_dataset import SyntheticRadarDataset
#pip install scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score
import math

# === ConvBlock for radar model ===
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# === RadarNet ===
# Enhanced encoder-decoder for multi-target detection from range-Doppler maps
# with improved target extraction and velocity estimation
class RadarNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1, detect_threshold=0.5, max_targets=10):
        """
        Initialize the RadarNet module for multi-target detection.
        
        Args:
            in_channels: Number of input channels (typically 2 for complex data)
            num_classes: Number of output classes (1 for detection map)
            detect_threshold: Threshold for target detection (0.0-1.0)
            max_targets: Maximum number of targets to detect
        """
        super().__init__()
        # Store configuration parameters
        self.detect_threshold = detect_threshold
        self.max_targets = max_targets
        
        # Encoder blocks with increasing channel depth
        self.enc1 = ConvBlock(in_channels, 32)     # Input: [B, 2, D, R] -> Output: [B, 32, D, R]
        self.enc2 = ConvBlock(32, 64)              # Input: [B, 32, D/2, R/2] -> Output: [B, 64, D/2, R/2]
        self.enc3 = ConvBlock(64, 128)             # Input: [B, 64, D/4, R/4] -> Output: [B, 128, D/4, R/4]
        self.enc4 = ConvBlock(128, 256)            # Input: [B, 128, D/8, R/8] -> Output: [B, 256, D/8, R/8]

        # Decoder blocks with skip connections
        self.dec3 = ConvBlock(256 + 128, 128)      # Input: [B, 256+128, D/4, R/4] -> Output: [B, 128, D/4, R/4]
        self.dec2 = ConvBlock(128 + 64, 64)        # Input: [B, 128+64, D/2, R/2] -> Output: [B, 64, D/2, R/2]
        self.dec1 = ConvBlock(64 + 32, 32)         # Input: [B, 64+32, D, R] -> Output: [B, 32, D, R]

        # Output layers for detection and velocity estimation
        self.detection_head = nn.Conv2d(32, num_classes, kernel_size=1)  # [B, 32, D, R] -> [B, 1, D, R]
        self.velocity_head = nn.Conv2d(32, 2, kernel_size=1)            # [B, 32, D, R] -> [B, 2, D, R]
        self.snr_head = nn.Conv2d(32, 1, kernel_size=1)                 # [B, 32, D, R] -> [B, 1, D, R]
        
        # Activation functions
        self.sigmoid = nn.Sigmoid()  # For detection probability
        self.tanh = nn.Tanh()        # For normalized velocity components

        # Pooling and upsampling operations
        self.pool = nn.MaxPool2d(2)  # Downsampling by factor of 2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Forward pass of the RadarNet module.
        
        Args:
            x: Input range-Doppler map with shape [B, 2, D, R]
               B: batch size, 2: complex (real/imag), D: Doppler bins, R: range bins
               
        Returns:
            Dictionary containing:
            - detection_map: Target detection probability map [B, 1, D, R]
            - velocity_map: Normalized velocity map (vx, vy) [B, 2, D, R]
            - snr_map: Signal-to-noise ratio map [B, 1, D, R]
            - target_list: List of detected targets with their properties
        """
        # Input shape validation
        h, w = x.shape[2:]  # [B, 2, D, R] -> D, R
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"RadarNet input height and width must be divisible by 8, got {h}x{w}")

        # === Encoder Path (Contracting) ===
        e1 = self.enc1(x)         # [B, 2, D, R] -> [B, 32, D, R]
        p1 = self.pool(e1)        # [B, 32, D, R] -> [B, 32, D/2, R/2]
        e2 = self.enc2(p1)        # [B, 32, D/2, R/2] -> [B, 64, D/2, R/2]
        p2 = self.pool(e2)        # [B, 64, D/2, R/2] -> [B, 64, D/4, R/4]
        e3 = self.enc3(p2)        # [B, 64, D/4, R/4] -> [B, 128, D/4, R/4]
        p3 = self.pool(e3)        # [B, 128, D/4, R/4] -> [B, 128, D/8, R/8]
        e4 = self.enc4(p3)        # [B, 128, D/8, R/8] -> [B, 256, D/8, R/8]

        # === Decoder Path (Expanding) with Skip Connections ===
        u3 = self.upsample(e4)    # [B, 256, D/8, R/8] -> [B, 256, D/4, R/4]
        # Ensure dimensions match before concatenation
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))  # [B, 256+128, D/4, R/4] -> [B, 128, D/4, R/4]

        u2 = self.upsample(d3)    # [B, 128, D/4, R/4] -> [B, 128, D/2, R/2]
        # Ensure dimensions match before concatenation
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))  # [B, 128+64, D/2, R/2] -> [B, 64, D/2, R/2]

        u1 = self.upsample(d2)    # [B, 64, D/2, R/2] -> [B, 64, D, R]
        # Ensure dimensions match before concatenation
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))  # [B, 64+32, D, R] -> [B, 32, D, R]

        # === Multi-task Output Heads ===
        # Detection probability map
        detection_logits = self.detection_head(d1)  # [B, 32, D, R] -> [B, 1, D, R]
        detection_map = self.sigmoid(detection_logits)  # [B, 1, D, R] - probability in range [0,1]
        
        # Velocity components map (normalized to [-1, 1] range)
        velocity_logits = self.velocity_head(d1)    # [B, 32, D, R] -> [B, 2, D, R]
        velocity_map = self.tanh(velocity_logits)    # [B, 2, D, R] - normalized velocity in range [-1,1]
        
        # SNR estimation map (positive values)

        snr_map = F.relu(self.snr_head(d1))                   # [B, 32, D, R] -> [B, 1, D, R] - positive values only
        
        # === Extract Target List ===
        # Process each item in the batch to extract detected targets
        batch_targets = []
        batch_size = x.shape[0]  # B
        doppler_bins = x.shape[2]  # D
        range_bins = x.shape[3]  # R
        
        for b in range(batch_size):
            # Find peaks in the detection map above threshold
            det_map_b = detection_map[b, 0]  # [D, R] - extract detection map for this batch item
            peaks = (det_map_b > self.detect_threshold)  # Boolean mask of detections above threshold
            
            # Get coordinates of detected targets
            doppler_indices, range_indices = torch.where(peaks)  # Get indices where detection > threshold
            
            # Limit to max_targets if needed
            if len(doppler_indices) > self.max_targets:
                # Sort by detection probability (descending) and keep top max_targets
                probs = det_map_b[doppler_indices, range_indices]  # [num_detections]
                _, top_indices = torch.topk(probs, self.max_targets)  # Get indices of top probabilities
                doppler_indices = doppler_indices[top_indices]  # Filter doppler indices
                range_indices = range_indices[top_indices]  # Filter range indices
            
            # Extract properties for each target
            targets = []
            for d_idx, r_idx in zip(doppler_indices, range_indices):
                # Create a dictionary with target properties
                target = {
                    'batch_idx': b,  # Batch index
                    'doppler_bin': d_idx.item(),  # Doppler bin index
                    'range_bin': r_idx.item(),  # Range bin index
                    'probability': detection_map[b, 0, d_idx, r_idx].item(),  # Detection probability
                    'velocity': (  # Normalized velocity components (vx, vy)
                        velocity_map[b, 0, d_idx, r_idx].item(),  # vx component
                        velocity_map[b, 1, d_idx, r_idx].item()   # vy component
                    ),
                    'snr_db': 10 * torch.log10(snr_map[b, 0, d_idx, r_idx] + 1e-10).item()  # SNR in dB
                }
                targets.append(target)
            
            batch_targets.append(targets)  # Add targets for this batch item to the list
        
        # Return all outputs in the expected format
        # Note: We keep the tensor dimensions as [B, C, D, R] for consistency with RadarTimeNet output
        return {
            'detection_map': detection_map,  # [B, 1, D, R] - Target detection probability
            'velocity_map': velocity_map,    # [B, 2, D, R] - Normalized velocity components (vx, vy)
            'snr_map': snr_map,              # [B, 1, D, R] - Signal-to-noise ratio
            'target_list': batch_targets     # List of B lists, each containing detected targets with properties
        }

# # === Learnable FFT Block (Used for RadarTimeNet) ===
# class LearnableFFT(nn.Module):
#     def __init__(self, input_len, output_len):
#         super().__init__()
#         self.real = nn.Parameter(torch.randn(input_len, output_len))
#         self.imag = nn.Parameter(torch.randn(input_len, output_len))

#     def forward(self, real_input, imag_input):
#         r_part = torch.matmul(real_input, self.real) - torch.matmul(imag_input, self.imag)
#         i_part = torch.matmul(real_input, self.imag) + torch.matmul(imag_input, self.real)
#         magnitude = torch.sqrt(r_part ** 2 + i_part ** 2)
#         return magnitude



# === Combined end-to-end model ===
# Enhanced end-to-end model chaining RadarTimeNet → RadarNet for full processing pipeline
# from raw IQ inputs to multi-target detection with location, velocity, and SNR estimation
class RadarEndToEnd(nn.Module):
    def __init__(self, time_net, detect_net, detect_threshold=0.5, max_targets=10):
        """
        Initialize the RadarEndToEnd module.
        
        Args:
            time_net: RadarTimeNet instance for processing raw IQ signals
            detect_net: RadarNet instance for target detection
            detect_threshold: Threshold for target detection
            max_targets: Maximum number of targets to detect
        """
        super().__init__()
        self.time_net = time_net
        self.detect_net = detect_net
        self.detect_threshold = detect_threshold
        self.max_targets = max_targets
        
        # Ensure detect_net has the same threshold and max_targets
        if hasattr(detect_net, 'detect_threshold'):
            detect_net.detect_threshold = detect_threshold
        if hasattr(detect_net, 'max_targets'):
            detect_net.max_targets = max_targets

    def forward(self, x, ref_signal=None, is_ofdm=False, modulation=None):
        """
        Forward pass of the RadarEndToEnd module.
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            ref_signal: Optional reference signal for demodulation
            is_ofdm: Whether the input signal is OFDM modulated
            modulation: Modulation scheme for OFDM decoding
            
        Returns:
            Dictionary containing:
            - detection_results: Output from RadarNet including detection_map, velocity_map, etc.
            - rd_map: Range-Doppler map from RadarTimeNet
            - ofdm_map: OFDM map (if is_ofdm=True)
            - decoded_bits: Decoded OFDM bits (if is_ofdm=True)
        """
        # Forward pass through time_net with optional parameters
        # Input: [B, num_rx, num_chirps, samples_per_chirp, 2]
        time_net_output = self.time_net(x, ref_signal=ref_signal, is_ofdm=is_ofdm, modulation=modulation)
        
        # Handle the case where time_net returns both RD map and OFDM outputs
        if is_ofdm and self.time_net.support_ofdm:
            # Unpack the tuple (rd_map, ofdm_map, decoded_bits)
            rd_map, ofdm_map, decoded_bits = time_net_output
            
            # Process the RD map with detect_net
            # Input: [B, 2, D, R], Output: Dict with detection_map, velocity_map, etc.
            detection_results = self.detect_net(rd_map)
            
            # Return comprehensive results
            return {
                'detection_results': detection_results,  # Dict with detection_map, velocity_map, target_list
                'rd_map': rd_map,                        # [B, 2, D, R]
                'ofdm_map': ofdm_map,                    # [B, 2, D, R]
                'decoded_bits': decoded_bits             # [B, num_symbols * num_active_subcarriers * bits_per_symbol]
            }
        else:
            # Standard case: just process the RD map
            # Input: [B, 2, D, R]
            rd_map = time_net_output
            
            # Process with detect_net
            # Output: Dict with detection_map, velocity_map, target_list
            detection_results = self.detect_net(rd_map)
            
            # Return results
            return {
                'detection_results': detection_results,  # Dict with detection_map, velocity_map, target_list
                'rd_map': rd_map                         # [B, 2, D, R]
            }

# === Dice Loss ===
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

#Binary detection map supervision.
#Velocity regression from range-Doppler phase.
# class MultiTaskLoss(nn.Module):
#     def __init__(self, lambda_det=1.0, lambda_vel=0.5):
#         super().__init__()
#         self.lambda_det = lambda_det
#         self.lambda_vel = lambda_vel
#         self.bce = nn.BCELoss()
#         self.mse = nn.MSELoss()

#     def forward(self, pred_det, target_det, pred_vel, target_vel):
#         loss_det = self.bce(pred_det, target_det)
#         loss_vel = self.mse(pred_vel, target_vel)
#         return self.lambda_det * loss_det + self.lambda_vel * loss_vel, loss_det, loss_vel

# Use weighted BCE or FocalLoss for detection
# For velocity regression: compute MSE only at positive target bins

# === MASK-AWARE LOSS + SEQUENCE TRIMMING FOR INFERENCE ===
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_det=1.0, lambda_vel=0.5):
        super().__init__()
        self.lambda_det = lambda_det
        self.lambda_vel = lambda_vel
        self.bce = nn.BCELoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_det, target_det, pred_vel, target_vel, mask=None):
        target_det = target_det.squeeze(1) if target_det.dim() == 5 else target_det
        loss_det = self.bce(pred_det, target_det)
        #loss_det = self.bce(pred_det, target_det)
        target_vel = target_vel.squeeze(1) if target_vel.dim() == 5 else target_vel
        loss_vel = self.mse(pred_vel, target_vel)
        #loss_vel = self.mse(pred_vel, target_vel)
        #ensures both inputs have shape [B, D, R, 1]. (batch, doppler, range, channel)

        if mask is not None:
            loss_det = loss_det * mask
            loss_vel = loss_vel * mask

        loss_det = loss_det.sum() / (mask.sum() + 1e-6) if mask is not None else loss_det.mean()
        loss_vel = loss_vel.sum() / (mask.sum() + 1e-6) if mask is not None else loss_vel.mean()

        total_loss = self.lambda_det * loss_det + self.lambda_vel * loss_vel
        return total_loss, loss_det, loss_vel

# === Mask generation for padded chirps ===
def generate_mask_from_lengths(lengths, max_len):
    B = len(lengths)
    mask = torch.zeros(B, max_len, 1, 1)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
    return mask

# === Sequence trimming for inference ===
def trim_rd_map(rd_map, original_length):
    return rd_map[:, :original_length, ...]

# === Device Setup ===
def get_device(gpuid='0'):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpuid}')
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# === Model Export ===
def export_model_to_onnx(model, dummy_input, export_path="radar_model.onnx"):
    model.eval()
    torch.onnx.export(model, dummy_input, export_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=11)
    print(f"Exported model to {export_path}")

# === TorchScript Export ===
def export_model_to_torchscript(model, dummy_input, export_path="radar_model.pt"):
    model.eval()
    traced = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced, export_path)
    print(f"Saved TorchScript model to {export_path}")

# === Doppler Prior Filter ===
def apply_doppler_filter(rd_map, threshold=0.2):
    doppler_energy = rd_map.norm(dim=-1)  # [B, D, R]
    mask = (doppler_energy > threshold * doppler_energy.max(dim=-1, keepdim=True)[0])
    return mask.float().unsqueeze(-1)


def visualize_training(rd_map, det_map, vel_map, step=0):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(rd_map.squeeze().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.title(f'Range-Doppler Map')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(det_map.squeeze().cpu().numpy(), aspect='auto', cmap='hot')
    plt.title('Detection Map')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(vel_map.squeeze().cpu().numpy(), aspect='auto', cmap='coolwarm')
    plt.title('Velocity Map')
    plt.colorbar()
    plt.suptitle(f'Step {step}')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"data/pretrain_{step}.pdf")

# === UPDATED train_radar_model WITH MASK SUPPORT ===
def train_radar_model(model, device, train_loader, val_loader, epochs=5, batch_size=8):
    model.train().to(device)
    multitask_loss = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for step, (x, y_det, y_vel, _, meta) in enumerate(train_loader):
            x, y_det, y_vel = x.to(device), y_det.to(device), y_vel.to(device)
            lengths = [len(m) for m in meta]
            max_chirps = x.shape[2] if x.dim() == 5 else y_det.shape[2]
            mask = generate_mask_from_lengths(lengths, max_chirps).to(device)

            out = model(x)
            rd_output = model.time_net(x).detach()
            vel_pred = torch.angle(torch.complex(rd_output[..., 0], rd_output[..., 1])) / np.pi

            loss, l_det, l_vel = multitask_loss(out, y_det, vel_pred.unsqueeze(-1), y_vel, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch} Step {step} Train Loss: {loss.item():.4f} | Det: {l_det.item():.4f} | Vel: {l_vel.item():.4f}")
                print(f"  Batch Target Counts: {[len(m) for m in meta]}")
                for m in meta:
                    for t in m:
                        print(f"    Target at Range {t['range_bin']}, Doppler {t['doppler_bin']}, SNR {t['snr_db']:.1f} dB")

                # Visualize predictions vs labels
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(y_det[0].cpu().numpy(), cmap='hot')
                plt.title("Target Label Map")
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(out[0].detach().cpu().numpy(), cmap='viridis')
                plt.title("Prediction Map")
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(mask[0].cpu().numpy().squeeze(), cmap='gray')
                plt.title("Mask")
                plt.colorbar()
                plt.tight_layout()
                fig_path = f"data/debug_epoch{epoch}_step{step}.png"
                plt.savefig(fig_path)
                print(f"Saved debug plot to {fig_path}")
                plt.close()

        model.eval()
        val_loss_total = 0
        perf_by_mod = {}
        all_preds, all_gts = [], []
        vel_errors = []

        with torch.no_grad():
            for x, y_det, y_vel, mod_type, meta in val_loader:
                x, y_det, y_vel = x.to(device), y_det.to(device), y_vel.to(device)
                lengths = [len(m) for m in meta]
                max_chirps = x.shape[2] if x.dim() == 5 else y_det.shape[2]
                mask = generate_mask_from_lengths(lengths, max_chirps).to(device)

                out = model(x)
                rd_output = model.time_net(x)
                vel_pred = torch.angle(torch.complex(rd_output[..., 0], rd_output[..., 1])) / np.pi

                loss, _, _ = multitask_loss(out, y_det, vel_pred.unsqueeze(-1), y_vel, mask)
                val_loss_total += loss.item()

                preds = (out > 0.5).float().cpu().numpy().flatten()
                gts = y_det.cpu().numpy().flatten()
                all_preds.extend(preds)
                all_gts.extend(gts)

                print(f"# Positive predictions: {(out > 0.5).sum().item()} / {out.numel()} | Output stats: min={out.min().item():.3f}, max={out.max().item():.3f}, mean={out.mean().item():.3f}")

                for b in range(len(meta)):
                    for t in meta[b]:
                        r, d = t['range_bin'], t['doppler_bin']
                        vel_pred = vel_pred.squeeze(-1) if vel_pred.dim() == 4 else vel_pred
                        y_vel = y_vel.squeeze(1).squeeze(-1) if y_vel.dim() == 5 else y_vel
                        if d < vel_pred.shape[1] and r < vel_pred.shape[2]:
                            vel_errors.append(abs(vel_pred[b, d, r].item() - y_vel[b, d, r].item()))

                for mt in mod_type:
                    mt = mt if isinstance(mt, str) else mt.item()
                    perf_by_mod[mt] = perf_by_mod.get(mt, 0.0) + loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")
        print("Validation Metrics:")
        print(f"  Precision: {precision_score(all_gts, all_preds, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(all_gts, all_preds, zero_division=0):.4f}")
        print(f"  F1 Score:  {f1_score(all_gts, all_preds, zero_division=0):.4f}")
        print(f"  Mean Vel Error: {np.mean(vel_errors):.4f} (approx Doppler deviation)")

        for mod, lsum in perf_by_mod.items():
            print(f"  Modulation {mod}: avg loss = {lsum / len(val_loader):.4f}")

        visualize_training(rd_output[0, ..., 0], out[0, ..., 0], vel_pred[0])

# === VARIABLE-LENGTH CHIRP SUPPORT + RD MAP VISUALIZATION ===
def pad_chirps_to_tensor(batch, pad_chirps):
    padded_iq, padded_det, padded_vel, mods, metas = [], [], [], [], []
    for iq, det, vel, mod, meta in batch:
        ch = iq.shape[2]
        pad_width = pad_chirps - ch
        if pad_width > 0:
            iq = F.pad(iq, (0, 0, 0, 0, 0, pad_width))  # pad chirp dim
            det = F.pad(det, (0, 0, 0, 0, 0, pad_width))
            vel = F.pad(vel, (0, 0, 0, 0, 0, pad_width))
        padded_iq.append(iq)
        padded_det.append(det)
        padded_vel.append(vel)
        mods.append(mod)
        metas.append(meta)
    return torch.stack(padded_iq), torch.stack(padded_det), torch.stack(padded_vel), mods, metas


# === Variable-length collate function ===
def radar_collate_fn(batch):
    max_chirps = max(x[0].shape[2] for x in batch)
    return pad_chirps_to_tensor(batch, max_chirps)

# def radar_collate_fn(batch):
#     xs, y_dets, y_vels, mods, metas = zip(*batch)
#     return (
#         torch.stack(xs),
#         torch.stack(y_dets),
#         torch.stack(y_vels),
#         list(mods),      # keep as list of strings
#         list(metas)      # keep as list of variable-length target lists
#     )

# === Range-Doppler Map Averaging Visualization ===
def visualize_avg_rd_map(rd_batch, step=0, title="Avg RD Map"):
    rd_mag = torch.sqrt(rd_batch[..., 0]**2 + rd_batch[..., 1]**2)  # [B, D, R]
    avg_rd = rd_mag.mean(dim=0).cpu().numpy()  # [D, R]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.imshow(avg_rd, aspect='auto', cmap='viridis')
    plt.title(f'{title} (Step {step})')
    plt.xlabel('Range Bin')
    plt.ylabel('Doppler Bin')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def main_train_radar(real_data_loader=None, num_rx=2, num_chirps=64, threshold=0.4):
    device = get_device()
    model = RadarEndToEnd(RadarTimeNet(num_rx=num_rx, num_chirps=num_chirps), RadarNet())

    if real_data_loader is not None:
        train_loader, val_loader = real_data_loader['train'], real_data_loader['val']
    else:
        num_per_type = 200
        datasets = [
            SyntheticRadarDataset(num_chirps=num_chirps, num_samples=num_per_type, modulation_type='none'),
            SyntheticRadarDataset(num_chirps=num_chirps, num_samples=num_per_type, modulation_type='sine'),
            SyntheticRadarDataset(num_chirps=num_chirps, num_samples=num_per_type, modulation_type='ofdm')
        ]
        full_dataset = ConcatDataset(datasets)
        train_len = int(0.8 * len(full_dataset))
        val_len = len(full_dataset) - train_len
        train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
        # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=radar_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=radar_collate_fn)

    visualize_synthetic_sample(train_dataset, index=0, save_path="data/train_sample.png")
    visualize_synthetic_batch(train_dataset, indices=[0,1,2,3], save_dir="data/train_sample_batch.png")

    multitask_loss = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(5):
        model.train()
        for step, (x, y_det, y_vel, _, meta) in enumerate(train_loader):
            x, y_det, y_vel = x.to(device), y_det.to(device), y_vel.to(device)
            out = model(x) #x: [8, 1, 64, 64, 2], y_det/y_vel: [8, 1, 64, 64, 1] => [8, 64, 64, 1]
            rd_output = model.time_net(x).detach() #[8, 64, 64, 2]
            vel_pred = torch.angle(torch.complex(rd_output[..., 0], rd_output[..., 1])) / np.pi
            #[8, 64, 64]
            loss, l_det, l_vel = multitask_loss(out, y_det, vel_pred.unsqueeze(-1), y_vel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                print(f"Epoch {epoch} Step {step} Train Loss: {loss.item():.4f} | Det: {l_det.item():.4f} | Vel: {l_vel.item():.4f}")
                print(f"  Batch Target Counts: {[len(m) for m in meta]}")
                for m in meta:
                    for t in m:
                        print(f"    Target at Range {t['range_bin']}, Doppler {t['doppler_bin']}, SNR {t['snr_db']:.1f} dB")

        # Validation with metrics
        model.eval()
        val_loss_total = 0
        perf_by_mod = {}
        all_preds, all_gts = [], []
        vel_errors = []

        with torch.no_grad():
            for x, y_det, y_vel, mod_type, meta in val_loader:
                x, y_det, y_vel = x.to(device), y_det.to(device), y_vel.to(device)
                out = model(x)
                rd_output = model.time_net(x)
                vel_pred = torch.angle(torch.complex(rd_output[..., 0], rd_output[..., 1])) / np.pi

                loss, _, _ = multitask_loss(out, y_det, vel_pred.unsqueeze(-1), y_vel)
                val_loss_total += loss.item()

                print(f"out stats: min={out.min().item():.3f}, max={out.max().item():.3f}, mean={out.mean().item():.3f}")
                print(f"# predicted > {threshold}: {(out > threshold).sum().item()}")
                print(f"# target > {threshold}: {(y_det > threshold).sum().item()}")

                preds = (out > threshold).float().cpu().numpy().flatten() #(32768,)
                gts = y_det.cpu().numpy().flatten() #(32768,)
                all_preds.extend(preds)
                all_gts.extend(gts)

                # Squeeze shapes before loop
                y_vel = y_vel.squeeze(1).squeeze(-1) #[8, 64, 64]
                vel_pred = vel_pred.squeeze(-1) if vel_pred.dim() == 4 else vel_pred #[8, 64, 64]

                for b in range(len(meta)):
                    for t in meta[b]:
                        r, d = t['range_bin'], t['doppler_bin']
                        vel_true = t['snr_db']  # for sim purposes we store SNR here
                        vel_pred_val = vel_pred[b, d, r].item()
                        vel_errors.append(abs(vel_pred_val - y_vel[b, d, r].item()))

                for mt in mod_type:
                    mt = mt if isinstance(mt, str) else mt.item()
                    perf_by_mod[mt] = perf_by_mod.get(mt, 0.0) + loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")
        print("Validation Metrics:")
        print(f"  Precision: {precision_score(all_gts, all_preds):.4f}")
        print(f"  Recall:    {recall_score(all_gts, all_preds):.4f}")
        print(f"  F1 Score:  {f1_score(all_gts, all_preds):.4f}")
        print(f"  Mean Vel Error: {np.mean(vel_errors):.4f} (approx Doppler deviation)")

        for mod, lsum in perf_by_mod.items():
            print(f"  Modulation {mod}: avg loss = {lsum / len(val_loader):.4f}")

        visualize_training(rd_output[0, ..., 0], out[0, ..., 0], vel_pred[0])

if __name__ == "__main__":
    main_train_radar()
