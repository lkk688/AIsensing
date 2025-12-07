#!/usr/bin/env python
"""
AIradar_comm_diag.py

Diagnostics for:
  - Existing joint_dump-based deep datasets
  - Trained JointNet generalized model

What it does:
  * Loads existing dumps (no regeneration)
  * Loads trained model from checkpoint
  * For each selected config & split:
      - Prints comm/radar tensor shapes & stats
      - Computes SER on one batch
      - For 16-QAM configs, prints 16x16 confusion matrix
"""

import os
import argparse
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

import AIradar_comm_model_g2 as g2  # adjust if your file name is different


# ---------------------------------------------------------------------
# Helper: pretty printing
# ---------------------------------------------------------------------
def tensor_stats(name, x: torch.Tensor, max_print_dim: int = 4):
    x_np = x.detach().cpu().float()
    print(f"  {name}: shape={tuple(x_np.shape)}, dtype={x_np.dtype}")
    flat = x_np.view(-1)
    print(f"    mean={flat.mean():.4e}, std={flat.std():.4e}, "
          f"min={flat.min():.4e}, max={flat.max():.4e}")


def print_target_stats(name, tgt: torch.Tensor, mod_order: int):
    tgt_np = tgt.detach().cpu().view(-1).numpy()
    print(f"  {name}: shape={tuple(tgt.shape)}")
    print(f"    unique labels (up to 20): {sorted(Counter(tgt_np).items())[:20]}")
    print(f"    label range: min={tgt_np.min()}, max={tgt_np.max()}, mod_order={mod_order}")


def confusion_matrix(pred: np.ndarray, tgt: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(tgt, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray):
    num_classes = cm.shape[0]
    print("  Confusion matrix (rows = true, cols = pred):")
    print("    counts:")
    for i in range(num_classes):
        row = " ".join(f"{v:5d}" for v in cm[i])
        print(f"    {i:2d}: {row}")
    print("    row-normalized:")
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sum, where=row_sum > 0)
    for i in range(num_classes):
        row = " ".join(f"{v:5.2f}" for v in norm[i])
        print(f"    {i:2d}: {row}")


# ---------------------------------------------------------------------
# Build a loader for one config + split
# ---------------------------------------------------------------------
def build_loader_for_config(cfg_name: str,
                            split: str,
                            data_root: str,
                            batch_size: int,
                            num_workers: int = 0):
    """
    Build a DataLoader for a single config + split using the same
    dump convention as AIradar_comm_model_g2.py.

    Assumptions (matching g2):
      - Dumps are saved under:  <data_root>/<cfg_name>/joint_dump_<split>.npy
      - RadarCommDeepDataset __init__ looks like:
            RadarCommDeepDataset(dump_path, config_name, config_id,
                                 max_samples=None, cache_in_memory=False)
    """

    DatasetCls = g2.RadarCommDumpDataset #RadarCommDeepDataset  # your deep dataset

    # Path to the existing dump (no regeneration here)
    #dump_dir = os.path.join(data_root, cfg_name)
    #dump_path = os.path.join(dump_dir, f"joint_dump_{split}.npy"
    
    dump_dir  = os.path.join(data_root, split, cfg_name)
    dump_path = dump_dir #os.path.join(dump_dir, "joint_dump.npy")

    if not os.path.exists(dump_path):
        raise FileNotFoundError(
            f"[Diag] Dump not found for cfg={cfg_name}, split={split}:\n"
            f"        {dump_path}\n"
            f"        Please run AIradar_comm_model_g2.py once to generate it."
        )

    # Map config name -> integer ID (same as in g2)
    if hasattr(g2, "CONFIG_ID_MAP"):
        cfg_id = g2.CONFIG_ID_MAP[cfg_name]
    elif hasattr(g2, "CONFIG_NAME_TO_ID"):
        cfg_id = g2.CONFIG_NAME_TO_ID[cfg_name]
    else:
        raise RuntimeError("Cannot find CONFIG_ID_MAP or CONFIG_NAME_TO_ID in g2.")

    # Instantiate dataset WITHOUT 'split=' (that's what caused the error)
    ds = DatasetCls(
        save_path=dump_path,
        config_name=cfg_name,
        #config_id=cfg_id,
        # adjust these if your __init__ signature is slightly different:
        num_samples=100,
        #cache_in_memory=False,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return ds, loader


# ---------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------
def load_trained_model(ckpt_path: str, device: torch.device):
    """
    Try to load the saved model.

    Two cases:
      1) The checkpoint is a full model object (torch.save(model, path))
      2) The checkpoint is a state_dict or a dict with 'model_state'
    """
    print(f"[Model] Loading checkpoint from: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)

    # ------------------------------------------------------------------
    # Case 1: you saved the full model (not a dict)
    # ------------------------------------------------------------------
    if not isinstance(obj, dict):
        print("  Detected non-dict object in checkpoint, assuming full model.")
        model = obj.to(device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Case 2: dict checkpoint: pure state_dict or {'model_state': ...}
    # ------------------------------------------------------------------
    state_dict = obj.get("model_state", obj)

    # Number of configs (must match training)
    # In your g2 script you have CONFIG_ID_MAP / CONFIG_NAME_TO_ID
    if hasattr(g2, "CONFIG_ID_MAP"):
        num_configs = len(g2.CONFIG_ID_MAP)
    elif hasattr(g2, "CONFIG_NAME_TO_ID"):
        num_configs = len(g2.CONFIG_NAME_TO_ID)
    else:
        raise RuntimeError("Cannot find CONFIG_ID_MAP or CONFIG_NAME_TO_ID in g2.")

    # If you used non-default hyperparams in training, mirror them here.
    # Otherwise, defaults (max_mod_order=64, base_ch=48, num_blocks=4, cond_dim=16)
    # should match what you used in AIradar_comm_model_g2.py.
    model = g2.JointRadarCommNet(
        num_configs=num_configs,
        max_mod_order=64,   # change if you used another in training
        base_ch=48,         # change if you used another in training
        num_blocks=4,       # change if you used another in training
        cond_dim=16,        # change if you used another in training
    )

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [load_trained_model] Missing keys ({len(missing)}):")
        for k in missing:
            print("    -", k)
    if unexpected:
        print(f"  [load_trained_model] Unexpected keys ({len(unexpected)}):")
        for k in unexpected:
            print("    -", k)

    model.to(device)
    model.eval()
    print("  Model loaded and set to eval().")
    return model


# ---------------------------------------------------------------------
# Diagnostics for one config + split
# ---------------------------------------------------------------------
def run_diag_for_config_split(model,
                              cfg_name: str,
                              split: str,
                              data_root: str,
                              batch_size: int,
                              device: torch.device):
    print(f"\n=== Diagnostics: cfg={cfg_name}, split={split} ===")
    ds, loader = build_loader_for_config(
        cfg_name=cfg_name,
        split=split,
        data_root=data_root,
        batch_size=batch_size,
    )

    print(f"  Dataset length: {len(ds)} samples")

    # Take the first batch
    batch_iter = iter(loader)
    try:
        radar_in, radar_tgt, comm_in, comm_tgt, meta = next(batch_iter)
    except StopIteration:
        print("  [WARN] Empty loader for this split.")
        return

    # Get mod_order for this config
    mod_order = g2.RADAR_COMM_CONFIGS[cfg_name].get("mod_order", 4)

    # Basic stats
    tensor_stats("radar_in", radar_in)
    tensor_stats("comm_in", comm_in)
    print_target_stats("radar_tgt", radar_tgt, mod_order=2)  # typically binary
    print_target_stats("comm_tgt", comm_tgt, mod_order=mod_order)

    # Show meta info (config_id etc.)
    print("  meta example:", meta[0])

    # Only run model for val/test (train is mostly for shape checking)
    if model is None or split == "train":
        return

    model.eval()
    radar_in = radar_in.to(device, non_blocking=True)
    comm_in = comm_in.to(device, non_blocking=True)
    # meta is list[dict]; extract config_ids
    cfg_ids = torch.tensor(
        [m["config_id"] for m in meta],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        out = model(radar_in, comm_in, cfg_ids)
        # Adjust depending on how your model returns outputs
        # Example: (radar_logits, comm_logits) = out
        if isinstance(out, (tuple, list)) and len(out) == 2:
            radar_pred, comm_logits = out
        else:
            raise RuntimeError("Model output format not recognized; "
                               "expecting (radar_pred, comm_logits).")

    # Flatten comm logits to [N_tot, num_classes]
    if comm_logits.dim() == 3:
        B, S, C = comm_logits.shape
        comm_logits_flat = comm_logits.reshape(B * S, C)
    elif comm_logits.dim() == 2:
        comm_logits_flat = comm_logits
        B = None
    else:
        raise RuntimeError(f"Unexpected comm_logits dim: {comm_logits.shape}")

    comm_tgt_flat = comm_tgt.view(-1).to(device)

    # SER for this batch
    pred_ints = comm_logits_flat.argmax(dim=-1)
    num_total = comm_tgt_flat.numel()
    num_err = (pred_ints != comm_tgt_flat).sum().item()
    ser = num_err / float(num_total)
    print(f"  [Deep] SER on this {split} batch: {ser:.4e} "
          f"({num_err}/{num_total} symbols wrong)")

    # Confusion matrix for moderate num_classes
    num_classes = comm_logits_flat.shape[-1]
    if num_classes <= 64:  # only print for reasonable size
        cm = confusion_matrix(
            pred=pred_ints.detach().cpu().numpy(),
            tgt=comm_tgt_flat.detach().cpu().numpy(),
            num_classes=num_classes,
        )
        print_confusion_matrix(cm)
    else:
        print(f"  [Info] num_classes={num_classes} too large for confusion matrix printout.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Diagnostics for AIradar_comm_model_g2 datasets & model"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/AIradar_comm_model_g2",
        help="Base directory where train/val/test dumps reside.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="data/AIradar_comm_model_g2/joint_net_generalized_best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=[
            "CN0566_TRADITIONAL",
            "XBand_10GHz_MediumRange",
            "Automotive_77GHz_LongRange",
            "AUTOMOTIVE_TRADITIONAL",
        ],
        help="Config names to diagnose.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to check (train/val/test).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for diagnostics (small is fine).",
    )
    parser.add_argument(
        "--no_model",
        action="store_true",
        help="If set, do not load model (only data stats).",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load model (or not)
    model = None
    if not args.no_model:
        model = load_trained_model(args.ckpt, device)

    for cfg_name in args.configs:
        if cfg_name not in g2.RADAR_COMM_CONFIGS:
            print(f"[WARN] Unknown config name in RADAR_COMM_CONFIGS: {cfg_name}")
            continue
        for split in args.splits:
            run_diag_for_config_split(
                model=model,
                cfg_name=cfg_name,
                split=split,
                data_root=args.data_root,
                batch_size=args.batch_size,
                device=device,
            )


if __name__ == "__main__":
    main()