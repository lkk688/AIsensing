import torch
import torch.nn.functional as F

def focal_bce_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification (with logits).
    Useful for class imbalance (e.g., sparse radar targets).
    """
    # logits, targets: (B,1,H,W)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = p*targets + (1-p)*(1-targets)
    w = alpha*targets + (1-alpha)*(1-targets)
    loss = (w * (1-pt).pow(gamma) * bce).mean()
    return loss

def dice_loss_with_logits(logits, targets, eps=1e-6):
    """
    Dice Loss for segmentation tasks.
    """
    probs = torch.sigmoid(logits)
    num = 2 * (probs*targets).sum(dim=(2,3)) + eps
    den = (probs.pow(2)+targets.pow(2)).sum(dim=(2,3)) + eps
    return 1 - (num/den).mean()

def radar_loss(logits, targets, alpha=0.25, gamma=2.0, dice_w=0.5):
    """
    Combined Focal + Dice + BCE loss for radar heatmap prediction.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    # Focal component
    pt  = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    fl  = (alpha*(1-targets)*(pt**gamma)*(-torch.log(1-pt)) + 
           (1-alpha)*targets*((1-pt)**gamma)*(-torch.log(pt))).mean()
           
    dl  = dice_loss_with_logits(logits, targets)
    return fl + bce + dice_w*dl

def comm_loss(logits, bits):
    """
    Binary Cross Entropy loss for communication bit prediction.
    """
    # bits shape (B,2) in {0,1}; logits (B,2)
    return F.binary_cross_entropy_with_logits(logits, bits.float())
