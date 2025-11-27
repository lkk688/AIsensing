import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
        
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, c, rates=(1, 6, 12, 18)):
        super().__init__()
        self.br = nn.ModuleList([nn.Conv2d(c, c//4, 3, padding=r, dilation=r) for r in rates])
        self.proj = nn.Conv2d(c, c, 1)
        
    def forward(self, x):
        xs = [F.relu(b(x)) for b in self.br]
        return F.relu(self.proj(torch.cat(xs, 1)))

def norm_layer(c, use_group=True):
    return nn.GroupNorm(num_groups=8, num_channels=c) if use_group else nn.BatchNorm2d(c)

class ConvBNReLU(nn.Module):
    """Convolution-Normalization-ReLU block with optional SE."""
    def __init__(self, c_in, c_out, k=3, s=1, p=1, use_group=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p)
        self.norm = norm_layer(c_out, use_group)
        self.se   = SEBlock(c_out)
        
    def forward(self, x):
        return self.se(F.relu(self.norm(self.conv(x))))

class Calib(nn.Module):
    """Learnable calibration layer (scale and shift)."""
    def __init__(self): 
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x): 
        return self.a * x + self.b

class RadarCommNet(nn.Module):
    """
    Multi-domain Multi-task Network for ISAC.
    
    Encodes radar data (FMCW or OTFS) and performs:
    1. Radar target detection (heatmap regression).
    2. Communication symbol demapping (via separate heads).
    """
    def __init__(self, in_ch_radar=1, in_ch_comm=2, base=48, use_group_norm=True):
        super().__init__()
        # Shared encoder
        self.enc1 = ConvBNReLU(in_ch_radar, base, use_group=use_group_norm)
        self.enc2 = ConvBNReLU(base, base*2, s=2, use_group=use_group_norm)
        self.enc3 = ConvBNReLU(base*2, base*4, s=2, use_group=use_group_norm)
        self.aspp = ASPP(base*4)

        # Decoder (shared)
        self.dec3 = ConvBNReLU(base*4, base*2, use_group=use_group_norm)
        self.dec2 = ConvBNReLU(base*2, base,   use_group=use_group_norm)
        
        # Task-specific heads
        self.out_fmcw = nn.Conv2d(base, 1, 1)
        self.out_otfs = nn.Conv2d(base, 1, 1)

        # Per-domain calibrations (before sigmoid)
        self.calib_fmcw = Calib()
        self.calib_otfs = Calib()

        # Communication demappers (Fully Convolutional)
        self.dem_ofdm = nn.Sequential(
            nn.Conv2d(in_ch_comm, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 1)  # (B, 2, H, W)
        )
        self.dem_otfs = nn.Sequential(
            nn.Conv2d(in_ch_comm, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 2, 1)  # (B, 2, M, N)
        )

    def forward_radar(self, x, domain="fmcw"):
        # U-Net like path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z  = self.aspp(e3)
        
        d3 = F.interpolate(self.dec3(z), size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = F.interpolate(self.dec2(d3), size=e1.shape[-2:], mode='bilinear', align_corners=False)
        
        logits = self.out_fmcw(d2) if domain=="fmcw" else self.out_otfs(d2)
        logits = (self.calib_fmcw if domain=="fmcw" else self.calib_otfs)(logits)
        return logits  # (B,1,H,W)

    def forward_comm(self, grid, domain="ofdm"):
        return self.dem_ofdm(grid) if domain=="ofdm" else self.dem_otfs(grid)

def calib_reg(model, w=1e-4):
    """Regularization for calibration parameters to keep them close to identity."""
    reg = (model.calib_fmcw.a-1).pow(2) + (model.calib_fmcw.b).pow(2) + \
          (model.calib_otfs.a-1).pow(2) + (model.calib_otfs.b).pow(2)
    return w * reg
