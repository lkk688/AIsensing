from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler

from AIradar_comm_models_v2 import ISACFoundationModel
from AIradar_comm_model_g6_comm import (
    AIRadar_Comm_Dataset_G3,
    UNIFIED_CONFIGS,
    CONFIG_ID_MAP,
)


# =============================================================================
# Utilities
# =============================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path



def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def load_checkpoint_if_exists(model: nn.Module, path: Optional[str], device: torch.device) -> bool:
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        return True
    return False



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def modulation_to_num_bits(mod_order: int) -> int:
    return int(round(math.log2(mod_order)))



def symbol_to_bits(symbols: torch.Tensor, mod_order: int) -> torch.Tensor:
    num_bits = modulation_to_num_bits(mod_order)
    B, H, W = symbols.shape
    bits = torch.zeros(B, num_bits, H, W, device=symbols.device, dtype=torch.float32)
    for i in range(num_bits):
        bits[:, i] = ((symbols >> i) & 1).float()
    return bits



def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "l2": float(torch.norm(x).item()),
    }



def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            g = p.grad.detach()
            total += float(torch.sum(g * g).item())
    return float(math.sqrt(total))



def named_grad_norms(model: nn.Module, name_filters: List[str]) -> Dict[str, float]:
    out = {}
    for key in name_filters:
        total = 0.0
        for name, p in model.named_parameters():
            if key in name and p.grad is not None:
                g = p.grad.detach()
                total += float(torch.sum(g * g).item())
        out[key] = float(math.sqrt(total))
    return out



def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = float(torch.norm(a).item() * torch.norm(b).item()) + eps
    if denom <= eps:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


# =============================================================================
# Config groups
# =============================================================================


COMM_CONFIG_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "OFDM": {
        "4QAM": ["Automotive_77GHz_LongRange"],
        "8QAM": ["8QAM_MediumRange"],
        "16QAM": ["CN0566_TRADITIONAL", "XBand_10GHz_MediumRange", "AUTOMOTIVE_TRADITIONAL"],
        "64QAM": ["5G_ISAC_HighBandwidth"],
    },
    "OTFS": {
        "4QAM": ["CN0566_OTFS_ISAC", "AUTOMOTIVE_OTFS_ISAC"],
        "16QAM": ["OTFS_HighMobility_Wideband"],
    },
}


# =============================================================================
# Run config
# =============================================================================


@dataclass
class RunConfig:
    mode: str = "train_comm_debug"
    out_dir: str = "data/g7_debug"
    data_root: str = "data/g7_debug"
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 4
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_samples: int = 120
    val_samples: int = 24
    test_samples: int = 24
    num_workers: int = 0
    comm_type: str = "all"
    qam_type: str = "all"
    comm_ckpt: Optional[str] = None
    label_smoothing: float = 0.02
    use_focal_loss: bool = False
    eval_snr_list: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    early_stop_patience: int = 8
    mixed_channel_train: bool = True
    unseen_holdout_fraction: float = 0.34
    comm_model: str = "isac"
    radar_model: str = "isac"
    radar_type: str = "all"
    radar_ckpt: Optional[str] = None
    channel_mode: str = "realistic"
    radar_pos_weight: float = 5.0
    radar_sigma: float = 3.0
    eval_cnr_list: Tuple[int, ...] = (0, 5, 10, 15, 20)
    eval_rcs_list: Tuple[int, ...] = (5, 10, 15, 20, 25)
    grad_clip: float = 5.0
    log_first_n_batches: int = 5
    save_batch_examples: bool = True
    track_activations_every: int = 1
    track_confusion_every: int = 1
    enable_gradient_probe: bool = True
    gradient_probe_batches: int = 2
    otfs_focus_start_epoch: int = 0
    otfs_focus_batches_per_epoch: int = 0
    otfs_loss_boost: float = 1.0
    ofdm_loss_boost: float = 1.0
    qam16_loss_boost: float = 1.0
    qam64_loss_boost: float = 1.0
    model_selection_metric: str = "seen"
    unseen_metric_weight: float = 0.25
    unseen_eval_every: int = 1
    enable_controlled_otfs_emphasis: bool = False
    otfs_ramp_epochs: int = 10


# =============================================================================
# Dataset wrappers
# =============================================================================


class CommUnifiedDataset(Dataset):
    def __init__(
        self,
        config_name: str,
        num_samples: int,
        save_root: str,
        split: str,
        fixed_snr: Optional[float] = None,
        enable_clutter: bool = True,
        enable_imperfect_csi: bool = True,
        enable_rf_impairments: bool = True,
        auto_generate: bool = True,
    ):
        super().__init__()
        self.config_name = config_name
        self.config = UNIFIED_CONFIGS[config_name]
        self.config_id = CONFIG_ID_MAP[config_name]
        self.save_path = ensure_dir(os.path.join(save_root, split, config_name))
        self.base_ds = AIRadar_Comm_Dataset_G3(
            config_name=config_name,
            num_samples=num_samples,
            save_path=self.save_path,
            drawfig=False,
            fixed_snr=fixed_snr,
            enable_clutter=enable_clutter,
            enable_imperfect_csi=enable_imperfect_csi,
            enable_rf_impairments=enable_rf_impairments,
            auto_generate=auto_generate,
        )

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base_ds[idx]
        grid_h = int(item["comm_info"]["grid_h"])
        grid_w = int(item["comm_info"]["grid_w"])
        tx_ints = np.asarray(item["comm_info"]["tx_ints"], dtype=np.int64)
        target = torch.tensor(tx_ints.reshape(grid_h, grid_w), dtype=torch.long)
        return {
            "comm_in": item["isac_comm_tensor"],
            "comm_target": target,
            "config_name": item["config_name"],
            "config_id": item["config_id"],
            "config_tensor": item["config_tensor"],
            "waveform_id": item["waveform_id"],
            "mod_id": item["mod_id"],
            "mod_order": int(item["comm_info"]["mod_order"]),
            "snr_db": float(item["comm_info"]["snr_db"]),
            "raw_sample": item,
        }


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, config_name: str):
        self.dataset = dataset
        self.config_name = config_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        item["dataset_config_name"] = self.config_name
        return item


# =============================================================================
# Collate and sampler
# =============================================================================



def comm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "comm_in": torch.stack([b["comm_in"] for b in batch]),
        "comm_target": torch.stack([b["comm_target"] for b in batch]),
        "config_name": [b["config_name"] for b in batch],
        "config_id": torch.stack([b["config_id"] for b in batch]),
        "config_tensor": torch.stack([b["config_tensor"] for b in batch]),
        "waveform_id": torch.stack([b["waveform_id"] for b in batch]),
        "mod_id": torch.stack([b["mod_id"] for b in batch]),
        "mod_order": torch.tensor([b["mod_order"] for b in batch], dtype=torch.long),
        "snr_db": torch.tensor([b["snr_db"] for b in batch], dtype=torch.float32),
        "raw_sample": [b["raw_sample"] for b in batch],
        "dataset_config_name": [b.get("dataset_config_name", b["config_name"]) for b in batch],
    }


class HomogeneousBatchSampler(Sampler[List[int]]):
    def __init__(self, datasets: List[Dataset], batch_size: int, shuffle: bool = True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches: List[List[int]] = []
        offset = 0
        for ds in datasets:
            n = len(ds)
            indices = list(range(offset, offset + n))
            if shuffle:
                random.shuffle(indices)
            for i in range(0, n, batch_size):
                self.batches.append(indices[i:i + batch_size])
            offset += n

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for b in self.batches:
            yield b

    def __len__(self) -> int:
        return len(self.batches)


# =============================================================================
# Build config splits and loaders
# =============================================================================



def choose_comm_configs(comm_type: str, qam_type: str) -> List[str]:
    if comm_type == "all":
        merged: Dict[str, List[str]] = defaultdict(list)
        for family in COMM_CONFIG_GROUPS.values():
            for qam, cfgs in family.items():
                merged[qam].extend(cfgs)
        groups = merged
    else:
        groups = COMM_CONFIG_GROUPS[comm_type]

    if qam_type == "all":
        flat: List[str] = []
        for cfgs in groups.values():
            flat.extend(cfgs)
        return sorted(list(dict.fromkeys(flat)))
    return list(groups.get(qam_type, []))


def is_otfs_config(config_name: str) -> bool:
    meta = UNIFIED_CONFIGS.get(config_name, {}).get("meta", {})
    comm_waveform = str(meta.get("comm_waveform", "")).upper()
    return comm_waveform == "OTFS" or "OTFS" in config_name.upper()



def split_seen_unseen(configs: List[str], holdout_fraction: float) -> Tuple[List[str], List[str]]:
    if len(configs) <= 1:
        return configs, []
    n_unseen = max(1, int(round(len(configs) * holdout_fraction)))
    unseen = configs[-n_unseen:]
    seen = configs[:-n_unseen]
    return seen, unseen



def build_train_loader(datasets_by_config: Dict[str, Dataset], batch_size: int, num_workers: int = 0) -> DataLoader:
    wrapped: List[Dataset] = []
    for cfg_name, ds in datasets_by_config.items():
        if isinstance(ds, ConcatDataset):
            wrapped.append(ds)
        else:
            wrapped.append(IndexedDataset(ds, cfg_name))
    concat_ds = ConcatDataset(wrapped)
    sampler = HomogeneousBatchSampler(wrapped, batch_size=batch_size, shuffle=True)
    return DataLoader(concat_ds, batch_sampler=sampler, collate_fn=comm_collate, num_workers=num_workers)



def build_eval_loaders(datasets_by_config: Dict[str, Dataset], batch_size: int, num_workers: int = 0) -> Dict[str, DataLoader]:
    return {
        cfg: DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=comm_collate, num_workers=num_workers)
        for cfg, ds in datasets_by_config.items()
    }



def build_comm_datasets(cfg: RunConfig) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset], List[str], List[str]]:
    configs = choose_comm_configs(cfg.comm_type, cfg.qam_type)
    seen_configs, unseen_configs = split_seen_unseen(configs, cfg.unseen_holdout_fraction)
    if not seen_configs:
        seen_configs = configs
        unseen_configs = []

    train_sets: Dict[str, Dataset] = {}
    val_seen_sets: Dict[str, Dataset] = {}
    test_unseen_sets: Dict[str, Dataset] = {}

    for config_name in seen_configs:
        if cfg.mixed_channel_train:
            awgn = IndexedDataset(
                CommUnifiedDataset(
                    config_name=config_name,
                    num_samples=cfg.train_samples // 2,
                    save_root=cfg.data_root,
                    split="train_awgn",
                    enable_clutter=False,
                    enable_imperfect_csi=False,
                    enable_rf_impairments=False,
                ),
                config_name,
            )
            real = IndexedDataset(
                CommUnifiedDataset(
                    config_name=config_name,
                    num_samples=cfg.train_samples - cfg.train_samples // 2,
                    save_root=cfg.data_root,
                    split="train_realistic",
                    enable_clutter=True,
                    enable_imperfect_csi=True,
                    enable_rf_impairments=True,
                ),
                config_name,
            )
            train_sets[config_name] = ConcatDataset([awgn, real])
        else:
            train_sets[config_name] = CommUnifiedDataset(
                config_name=config_name,
                num_samples=cfg.train_samples,
                save_root=cfg.data_root,
                split="train",
                enable_clutter=True,
                enable_imperfect_csi=True,
                enable_rf_impairments=True,
            )

        val_seen_sets[config_name] = CommUnifiedDataset(
            config_name=config_name,
            num_samples=cfg.val_samples,
            save_root=cfg.data_root,
            split="val_seen",
            enable_clutter=True,
            enable_imperfect_csi=True,
            enable_rf_impairments=True,
        )

    for config_name in unseen_configs:
        test_unseen_sets[config_name] = CommUnifiedDataset(
            config_name=config_name,
            num_samples=cfg.test_samples,
            save_root=cfg.data_root,
            split="test_unseen",
            enable_clutter=True,
            enable_imperfect_csi=True,
            enable_rf_impairments=True,
        )

    return train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs


# =============================================================================
# Losses and model wrapper
# =============================================================================


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class ISACCommWrapper(nn.Module):
    def __init__(self, model: ISACFoundationModel):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, config_tensor: torch.Tensor, waveform_id: torch.Tensor, mod_id: torch.Tensor) -> torch.Tensor:
        out = self.model.forward_comm(
            x_comm=x,
            config_tensor=config_tensor,
            waveform_id=waveform_id,
            mod_id=mod_id,
        )
        return out["bit_logits"]

    def forward_debug(
        self,
        x: torch.Tensor,
        config_tensor: torch.Tensor,
        waveform_id: torch.Tensor,
        mod_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.model.forward_comm(
            x_comm=x,
            config_tensor=config_tensor,
            waveform_id=waveform_id,
            mod_id=mod_id,
        )

    @staticmethod
    def bit_logits_to_symbol_logits(bit_logits: torch.Tensor, mod_order: int) -> torch.Tensor:
        temp = ISACFoundationModel(comm_in_channels=5)
        return temp.comm_head.bit_logits_to_symbol_logits(bit_logits, mod_order)



def create_comm_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name.lower() != "isac":
        raise ValueError(f"Unsupported comm model: {model_name}")
    base = ISACFoundationModel(comm_in_channels=5, base_ch=64, cond_dim=64, config_dim=8).to(device)
    return ISACCommWrapper(base).to(device)


# =============================================================================
# Debug analyzers
# =============================================================================


class BatchDebugger:
    def __init__(self, out_dir: str):
        self.out_dir = ensure_dir(out_dir)
        self.batch_logs: List[Dict[str, Any]] = []

    def log_batch(self, epoch: int, batch_idx: int, batch: Dict[str, Any], bit_logits: torch.Tensor, loss: float) -> None:
        x = batch["comm_in"]
        cfg_name = batch["dataset_config_name"][0]
        mod_order = int(batch["mod_order"][0].item())
        entry = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "config_name": cfg_name,
            "shape": list(x.shape),
            "mod_order": mod_order,
            "snr_mean": float(batch["snr_db"].mean().item()),
            "input_stats": tensor_stats(x),
            "logit_stats": tensor_stats(bit_logits.detach().cpu()),
            "loss": float(loss),
        }
        self.batch_logs.append(entry)

    def flush(self) -> None:
        save_json({"batches": self.batch_logs}, os.path.join(self.out_dir, "batch_debug.json"))


class ActivationTracker:
    def __init__(self, out_dir: str):
        self.out_dir = ensure_dir(out_dir)
        self.records: List[Dict[str, Any]] = []

    def track(self, epoch: int, batch: Dict[str, Any], out: Dict[str, torch.Tensor]) -> None:
        rec = {
            "epoch": epoch,
            "config_name": batch["dataset_config_name"][0],
            "shared_f1": tensor_stats(out["shared_f1"].detach().cpu()),
            "shared_f2": tensor_stats(out["shared_f2"].detach().cpu()),
            "shared_f3": tensor_stats(out["shared_f3"].detach().cpu()),
            "comm_feat": tensor_stats(out["comm_feat"].detach().cpu()),
            "bit_logits": tensor_stats(out["bit_logits"].detach().cpu()),
            "cond": tensor_stats(out["cond"].detach().cpu()),
        }
        self.records.append(rec)

    def flush(self) -> None:
        save_json({"activations": self.records}, os.path.join(self.out_dir, "activation_debug.json"))


class MultiConfigProbe:
    """
    Measures whether shared features differ meaningfully across configs and whether
    config conditioning is being used.
    """
    def __init__(self, out_dir: str):
        self.out_dir = ensure_dir(out_dir)
        self.epoch_records: List[Dict[str, Any]] = []

    @torch.no_grad()
    def run_epoch_probe(self, epoch: int, model: ISACCommWrapper, loaders: Dict[str, DataLoader], device: torch.device) -> None:
        model.eval()
        per_cfg_summary = {}
        feature_bank = {}
        cond_bank = {}

        for cfg_name, loader in loaders.items():
            for batch in loader:
                comm_in = batch["comm_in"].to(device)
                config_tensor = batch["config_tensor"].to(device)
                waveform_id = batch["waveform_id"].to(device)
                mod_id = batch["mod_id"].to(device)
                out = model.forward_debug(comm_in, config_tensor, waveform_id, mod_id)

                shared_vec = out["shared_f3"].mean(dim=(2, 3)).detach().cpu()   # [B, C]
                cond_vec = out["cond"].detach().cpu()                             # [B, D]
                feature_bank[cfg_name] = shared_vec.mean(dim=0)
                cond_bank[cfg_name] = cond_vec.mean(dim=0)
                per_cfg_summary[cfg_name] = {
                    "shared_f3_mean_norm": float(torch.norm(shared_vec.mean(dim=0)).item()),
                    "cond_mean_norm": float(torch.norm(cond_vec.mean(dim=0)).item()),
                    "shared_f3_stats": tensor_stats(shared_vec),
                    "cond_stats": tensor_stats(cond_vec),
                }
                break

        similarities = {}
        cfg_names = list(feature_bank.keys())
        for i in range(len(cfg_names)):
            for j in range(i + 1, len(cfg_names)):
                a, b = cfg_names[i], cfg_names[j]
                similarities[f"{a}__{b}"] = {
                    "shared_f3_cosine": cosine_similarity(feature_bank[a], feature_bank[b]),
                    "cond_cosine": cosine_similarity(cond_bank[a], cond_bank[b]),
                }

        self.epoch_records.append({
            "epoch": epoch,
            "per_config": per_cfg_summary,
            "pairwise": similarities,
        })

    def flush(self) -> None:
        save_json({"multi_config_probe": self.epoch_records}, os.path.join(self.out_dir, "multi_config_probe.json"))


# =============================================================================
# Trainer
# =============================================================================


class CommTrainer:
    def __init__(
        self,
        model: ISACCommWrapper,
        device: torch.device,
        debug_dir: str,
        otfs_focus_start_epoch: int = 0,
        otfs_loss_boost: float = 1.0,
        ofdm_loss_boost: float = 1.0,
        qam16_loss_boost: float = 1.0,
        qam64_loss_boost: float = 1.0,
    ):
        self.model = model
        self.device = device
        self.focal_bce = FocalBCELoss()
        self.debug_dir = ensure_dir(debug_dir)
        self.batch_debugger = BatchDebugger(os.path.join(debug_dir, "batch_logs"))
        self.activation_tracker = ActivationTracker(os.path.join(debug_dir, "activations"))
        self.multi_config_probe = MultiConfigProbe(os.path.join(debug_dir, "multi_config_probe"))
        self.otfs_focus_start_epoch = int(max(0, otfs_focus_start_epoch))
        self.otfs_loss_boost = float(max(1e-6, otfs_loss_boost))
        self.ofdm_loss_boost = float(max(1e-6, ofdm_loss_boost))
        self.qam16_loss_boost = float(max(1e-6, qam16_loss_boost))
        self.qam64_loss_boost = float(max(1e-6, qam64_loss_boost))

    def batch_loss_weight(
        self,
        cfg_name: str,
        mod_order: int,
        epoch: int,
        otfs_boost_override: Optional[float] = None,
    ) -> float:
        otfs_boost = self.otfs_loss_boost if otfs_boost_override is None else float(max(1e-6, otfs_boost_override))
        w = otfs_boost if (is_otfs_config(cfg_name) and epoch >= self.otfs_focus_start_epoch) else self.ofdm_loss_boost
        if mod_order == 16:
            w *= self.qam16_loss_boost
        elif mod_order == 64:
            w *= self.qam64_loss_boost
        return float(w)

    def compute_loss(
        self,
        bit_logits: torch.Tensor,
        target_symbols: torch.Tensor,
        mod_order: int,
        label_smoothing: float,
        use_focal_loss: bool,
    ) -> torch.Tensor:
        active_bits = modulation_to_num_bits(mod_order)
        gt_bits = symbol_to_bits(target_symbols, mod_order)[:, :active_bits]
        pred_bits = bit_logits[:, :active_bits]
        if use_focal_loss:
            return self.focal_bce(pred_bits, gt_bits)
        if label_smoothing > 0:
            gt_bits = gt_bits * (1.0 - label_smoothing) + 0.5 * label_smoothing
        return F.binary_cross_entropy_with_logits(pred_bits, gt_bits)

    def to_symbol_logits(self, bit_logits: torch.Tensor, mod_order: int) -> torch.Tensor:
        return self.model.bit_logits_to_symbol_logits(bit_logits, mod_order)

    def gradient_probe(
        self,
        train_loader: DataLoader,
        epoch: int,
        use_focal_loss: bool,
        label_smoothing: float,
    ) -> Optional[Dict[str, Any]]:
        batches = []
        for batch in train_loader:
            batches.append(batch)
            if len(batches) >= 2:
                break
        if len(batches) < 2:
            return None

        probe = {"epoch": epoch, "pairs": []}
        self.model.train()
        for i in range(len(batches) - 1):
            b1 = batches[i]
            b2 = batches[i + 1]
            pair = {}
            grads = []
            names = []
            for batch in [b1, b2]:
                self.model.zero_grad(set_to_none=True)
                comm_in = batch["comm_in"].to(self.device)
                comm_target = batch["comm_target"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                waveform_id = batch["waveform_id"].to(self.device)
                mod_id = batch["mod_id"].to(self.device)
                mod_order = int(batch["mod_order"][0].item())
                bit_logits = self.model(comm_in, config_tensor, waveform_id, mod_id)
                loss = self.compute_loss(bit_logits, comm_target, mod_order, label_smoothing, use_focal_loss)
                loss.backward()
                vecs = []
                for name, p in self.model.named_parameters():
                    if p.grad is not None and ("shared_stem" in name or "cond_encoder" in name):
                        vecs.append(p.grad.detach().reshape(-1).cpu())
                grads.append(torch.cat(vecs) if vecs else torch.zeros(1))
                names.append(batch["dataset_config_name"][0])
            pair = {
                "cfg_a": names[0],
                "cfg_b": names[1],
                "shared_grad_cosine": cosine_similarity(grads[0], grads[1]),
            }
            probe["pairs"].append(pair)
        return probe

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        grad_clip: float = 5.0,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        log_first_n_batches: int = 0,
        track_activations: bool = False,
        otfs_boost_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_ber = 0.0
        total_batches = 0
        per_cfg = defaultdict(lambda: {"loss": 0.0, "ber": 0.0, "n": 0})
        grad_logs = []
        shape_counter = Counter()

        for batch_idx, batch in enumerate(loader):
            comm_in = batch["comm_in"].to(self.device)
            comm_target = batch["comm_target"].to(self.device)
            config_tensor = batch["config_tensor"].to(self.device)
            waveform_id = batch["waveform_id"].to(self.device)
            mod_id = batch["mod_id"].to(self.device)
            mod_order = int(batch["mod_order"][0].item())
            cfg_name = batch["dataset_config_name"][0]
            shape_counter[str(tuple(comm_in.shape))] += 1

            optimizer.zero_grad(set_to_none=True)
            out = self.model.forward_debug(comm_in, config_tensor, waveform_id, mod_id)
            bit_logits = out["bit_logits"]
            base_loss = self.compute_loss(bit_logits, comm_target, mod_order, label_smoothing, use_focal_loss)
            loss_weight = self.batch_loss_weight(cfg_name, mod_order, epoch, otfs_boost_override=otfs_boost_override)
            loss = base_loss * loss_weight
            loss.backward()

            g_total = grad_norm(self.model.parameters())
            g_named = named_grad_norms(self.model, ["model.cond_encoder", "model.shared_stem", "model.comm_input", "model.comm_expert", "model.comm_head"])
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                symbol_logits = self.to_symbol_logits(bit_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                ber = float((pred_bits != gt_bits).float().mean().item())
                pred_hist = torch.bincount(pred.reshape(-1).cpu(), minlength=mod_order).tolist()
                tgt_hist = torch.bincount(comm_target.reshape(-1).cpu(), minlength=mod_order).tolist()

            total_loss += float(loss.item())
            total_ber += ber
            total_batches += 1

            per_cfg[cfg_name]["loss"] += float(loss.item())
            per_cfg[cfg_name]["ber"] += ber
            per_cfg[cfg_name]["n"] += 1

            if batch_idx < log_first_n_batches:
                self.batch_debugger.log_batch(epoch, batch_idx, batch, bit_logits.detach().cpu(), float(loss.item()))

            if track_activations:
                self.activation_tracker.track(epoch, batch, {k: v.detach().cpu() for k, v in out.items()})

            grad_logs.append({
                "batch_idx": batch_idx,
                "config_name": cfg_name,
                "loss": float(loss.item()),
                "base_loss": float(base_loss.item()),
                "loss_weight": float(loss_weight),
                "ber": ber,
                "grad_total": g_total,
                "grad_named": g_named,
                "pred_hist": pred_hist,
                "target_hist": tgt_hist,
            })

        per_cfg_avg = {}
        for cfg_name, stats in per_cfg.items():
            n = max(1, stats["n"])
            per_cfg_avg[cfg_name] = {
                "loss": stats["loss"] / n,
                "ber": stats["ber"] / n,
                "n": stats["n"],
            }

        return {
            "loss": total_loss / max(1, total_batches),
            "ber": total_ber / max(1, total_batches),
            "per_config": per_cfg_avg,
            "grad_logs": grad_logs,
            "shape_histogram": dict(shape_counter),
        }

    def train_focus_batches(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        max_batches: int,
        grad_clip: float = 5.0,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
        extra_loss_boost: float = 1.0,
        otfs_boost_override: Optional[float] = None,
    ) -> Dict[str, float]:
        if max_batches <= 0:
            return {"batches": 0, "loss": 0.0, "ber": 0.0}
        self.model.train()
        total_loss = 0.0
        total_ber = 0.0
        n = 0
        for batch in loader:
            comm_in = batch["comm_in"].to(self.device)
            comm_target = batch["comm_target"].to(self.device)
            config_tensor = batch["config_tensor"].to(self.device)
            waveform_id = batch["waveform_id"].to(self.device)
            mod_id = batch["mod_id"].to(self.device)
            mod_order = int(batch["mod_order"][0].item())
            cfg_name = batch["dataset_config_name"][0]

            optimizer.zero_grad(set_to_none=True)
            bit_logits = self.model(comm_in, config_tensor, waveform_id, mod_id)
            base_loss = self.compute_loss(bit_logits, comm_target, mod_order, label_smoothing, use_focal_loss)
            loss_weight = self.batch_loss_weight(
                cfg_name,
                mod_order,
                epoch,
                otfs_boost_override=otfs_boost_override,
            ) * float(max(1e-6, extra_loss_boost))
            loss = base_loss * loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                symbol_logits = self.to_symbol_logits(bit_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                ber = float((pred_bits != gt_bits).float().mean().item())
            total_loss += float(loss.item())
            total_ber += ber
            n += 1
            if n >= max_batches:
                break
        return {"batches": n, "loss": total_loss / max(1, n), "ber": total_ber / max(1, n)}

    @torch.no_grad()
    def evaluate(self, loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        self.model.eval()
        per_cfg = {}
        all_bers = []
        per_cfg_margin = {}

        for cfg_name, loader in loaders.items():
            cfg_bers = []
            cfg_margins = []
            for batch in loader:
                comm_in = batch["comm_in"].to(self.device)
                comm_target = batch["comm_target"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                waveform_id = batch["waveform_id"].to(self.device)
                mod_id = batch["mod_id"].to(self.device)
                mod_order = int(batch["mod_order"][0].item())

                bit_logits = self.model(comm_in, config_tensor, waveform_id, mod_id)
                symbol_logits = self.to_symbol_logits(bit_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                ber = float((pred_bits != gt_bits).float().mean().item())
                cfg_bers.append(ber)
                all_bers.append(ber)

                top2 = torch.topk(symbol_logits, k=min(2, mod_order), dim=1).values
                if top2.shape[1] >= 2:
                    margin = (top2[:, 0] - top2[:, 1]).mean().item()
                    cfg_margins.append(float(margin))

            per_cfg[cfg_name] = float(np.mean(cfg_bers)) if cfg_bers else 1.0
            per_cfg_margin[cfg_name] = float(np.mean(cfg_margins)) if cfg_margins else 0.0

        return {
            "avg_ber": float(np.mean(all_bers)) if all_bers else 1.0,
            "per_config": per_cfg,
            "per_config_margin": per_cfg_margin,
        }

    @torch.no_grad()
    def evaluate_snr_sweep(
        self,
        config_name: str,
        save_root: str,
        snr_list: List[int],
        num_samples: int,
        channel_mode: str = "realistic",
    ) -> Dict[str, Any]:
        self.model.eval()
        results = {"snr": list(snr_list), "dl_ber": [], "trad_ber": [], "config": config_name, "channel_mode": channel_mode}
        if channel_mode == "awgn":
            enable_clutter = False
            enable_imperfect_csi = False
            enable_rf_impairments = False
        else:
            enable_clutter = True
            enable_imperfect_csi = True
            enable_rf_impairments = False

        for snr_db in snr_list:
            ds = CommUnifiedDataset(
                config_name=config_name,
                num_samples=num_samples,
                save_root=save_root,
                split=f"eval_{channel_mode}_{snr_db}",
                fixed_snr=snr_db,
                enable_clutter=enable_clutter,
                enable_imperfect_csi=enable_imperfect_csi,
                enable_rf_impairments=enable_rf_impairments,
            )
            loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=comm_collate)
            dl_bers = []
            trad_bers = []
            for batch in loader:
                comm_in = batch["comm_in"].to(self.device)
                comm_target = batch["comm_target"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                waveform_id = batch["waveform_id"].to(self.device)
                mod_id = batch["mod_id"].to(self.device)
                mod_order = int(batch["mod_order"][0].item())
                bit_logits = self.model(comm_in, config_tensor, waveform_id, mod_id)
                symbol_logits = self.to_symbol_logits(bit_logits, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                dl_bers.append(float((pred_bits != gt_bits).float().mean().item()))
                trad_bers.append(float(batch["raw_sample"][0]["comm_info"].get("ber", 0.5)))
            results["dl_ber"].append(float(np.mean(dl_bers)) if dl_bers else 1.0)
            results["trad_ber"].append(float(np.mean(trad_bers)) if trad_bers else 1.0)
        return results


# =============================================================================
# Plotting
# =============================================================================



def plot_comm_sweep(results: Dict[str, Any], out_path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(results["snr"], results["dl_ber"], "o-", linewidth=2, label="DL")
    plt.semilogy(results["snr"], results["trad_ber"], "s--", linewidth=2, label="Traditional")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title or f"BER vs SNR - {results['config']}")
    plt.grid(True, alpha=0.3, which="both")
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def plot_comm_sweep_all(snr_reports: Dict[str, Any], out_path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(snr_reports)))
    for i, (key, res) in enumerate(snr_reports.items()):
        plt.semilogy(res["snr"], res["dl_ber"], "o-", color=colors[i], linewidth=2, label=f"{key} (DL)")
        plt.semilogy(res["snr"], res["trad_ber"], "s--", color=colors[i], linewidth=2, alpha=0.6, label=f"{key} (Trad)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title or "BER vs SNR - All Configs")
    plt.grid(True, alpha=0.3, which="both")
    plt.ylim(1e-4, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def plot_history(history: List[Dict[str, Any]], out_path: str) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_ber = [h["train_ber"] for h in history]
    val_ber = [h["val_seen_ber"] for h in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_loss, "o-", label="train loss")
    plt.plot(epochs, train_ber, "s-", label="train BER")
    plt.plot(epochs, val_ber, "^-", label="val seen BER")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Communication Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def plot_per_config_ber(history: List[Dict[str, Any]], out_path: str) -> None:
    cfgs = sorted({cfg for h in history for cfg in h["per_config_val"].keys()})
    plt.figure(figsize=(10, 6))
    for cfg in cfgs:
        xs, ys = [], []
        for h in history:
            if cfg in h["per_config_val"]:
                xs.append(h["epoch"])
                ys.append(h["per_config_val"][cfg])
        plt.plot(xs, ys, marker="o", label=cfg)
    plt.xlabel("Epoch")
    plt.ylabel("Validation BER")
    plt.title("Per-config Validation BER")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()



def plot_grad_norms(history: List[Dict[str, Any]], out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    keys = ["model.cond_encoder", "model.shared_stem", "model.comm_input", "model.comm_expert", "model.comm_head"]
    for key in keys:
        vals = []
        epochs = []
        for h in history:
            grads = h.get("epoch_grad_summary", {})
            if key in grads:
                epochs.append(h["epoch"])
                vals.append(grads[key])
        if vals:
            plt.plot(epochs, vals, marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.title("Module Gradient Norms")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# =============================================================================
# Pipeline
# =============================================================================



def train_comm_pipeline_debug(cfg: RunConfig, device: torch.device) -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(cfg.out_dir, "comm_debug"))
    debug_dir = ensure_dir(os.path.join(out_dir, "debug"))

    train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs = build_comm_datasets(cfg)
    train_loader = build_train_loader(train_sets, cfg.batch_size, cfg.num_workers)
    val_seen_loaders = build_eval_loaders(val_seen_sets, cfg.batch_size, cfg.num_workers)
    test_unseen_loaders = build_eval_loaders(test_unseen_sets, cfg.batch_size, cfg.num_workers)

    model = create_comm_model(cfg.comm_model, device)
    if cfg.comm_ckpt:
        load_checkpoint_if_exists(model, cfg.comm_ckpt, device)

    trainer = CommTrainer(
        model,
        device,
        debug_dir,
        otfs_focus_start_epoch=cfg.otfs_focus_start_epoch,
        otfs_loss_boost=cfg.otfs_loss_boost,
        ofdm_loss_boost=cfg.ofdm_loss_boost,
        qam16_loss_boost=cfg.qam16_loss_boost,
        qam64_loss_boost=cfg.qam64_loss_boost,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"ber": float("inf"), "score": float("inf"), "epoch": -1}
    best_path = os.path.join(out_dir, "comm_best.pt")
    patience = 0
    history: List[Dict[str, Any]] = []
    gradient_probe_records: List[Dict[str, Any]] = []

    dataset_overview = {
        "seen_configs": seen_configs,
        "unseen_configs": unseen_configs,
        "train_sizes": {k: len(v) for k, v in train_sets.items()},
        "val_sizes": {k: len(v) for k, v in val_seen_sets.items()},
        "test_sizes": {k: len(v) for k, v in test_unseen_sets.items()},
        "comm_type": cfg.comm_type,
        "qam_type": cfg.qam_type,
        "mixed_channel_train": cfg.mixed_channel_train,
        "otfs_focus_start_epoch": cfg.otfs_focus_start_epoch,
        "otfs_focus_batches_per_epoch": cfg.otfs_focus_batches_per_epoch,
        "enable_controlled_otfs_emphasis": cfg.enable_controlled_otfs_emphasis,
        "otfs_loss_boost": cfg.otfs_loss_boost,
        "otfs_ramp_epochs": cfg.otfs_ramp_epochs,
        "ofdm_loss_boost": cfg.ofdm_loss_boost,
        "qam16_loss_boost": cfg.qam16_loss_boost,
        "qam64_loss_boost": cfg.qam64_loss_boost,
        "model_selection_metric": cfg.model_selection_metric,
        "unseen_metric_weight": cfg.unseen_metric_weight,
    }
    save_json(dataset_overview, os.path.join(debug_dir, "dataset_overview.json"))

    otfs_train_sets = {k: v for k, v in train_sets.items() if is_otfs_config(k)}
    otfs_focus_loader = build_train_loader(otfs_train_sets, cfg.batch_size, cfg.num_workers) if otfs_train_sets else None

    def epoch_otfs_boost(epoch: int) -> float:
        if not cfg.enable_controlled_otfs_emphasis:
            return 1.0
        if epoch < cfg.otfs_focus_start_epoch:
            return 1.0
        ramp = max(1, int(cfg.otfs_ramp_epochs))
        progress = min(1.0, float(epoch - cfg.otfs_focus_start_epoch + 1) / float(ramp))
        target = max(1.0, float(cfg.otfs_loss_boost))
        return 1.0 + (target - 1.0) * progress

    for epoch in range(1, cfg.epochs + 1):
        dynamic_otfs_boost = epoch_otfs_boost(epoch)
        train_stats = trainer.train_one_epoch(
            train_loader,
            optimizer,
            epoch=epoch,
            grad_clip=cfg.grad_clip,
            label_smoothing=cfg.label_smoothing,
            use_focal_loss=cfg.use_focal_loss,
            log_first_n_batches=cfg.log_first_n_batches if epoch == 1 else 0,
            track_activations=(epoch % cfg.track_activations_every == 0),
            otfs_boost_override=dynamic_otfs_boost,
        )
        focus_stats = {"batches": 0, "loss": 0.0, "ber": 0.0}
        focus_enabled = cfg.enable_controlled_otfs_emphasis
        if focus_enabled and otfs_focus_loader is not None and cfg.otfs_focus_batches_per_epoch > 0 and epoch >= cfg.otfs_focus_start_epoch:
            focus_stats = trainer.train_focus_batches(
                otfs_focus_loader,
                optimizer,
                epoch=epoch,
                max_batches=cfg.otfs_focus_batches_per_epoch,
                grad_clip=cfg.grad_clip,
                label_smoothing=cfg.label_smoothing,
                use_focal_loss=cfg.use_focal_loss,
                extra_loss_boost=max(1.0, dynamic_otfs_boost),
                otfs_boost_override=dynamic_otfs_boost,
            )
        val_seen = trainer.evaluate(val_seen_loaders)
        val_ber = val_seen["avg_ber"]
        unseen_ber = None
        if test_unseen_loaders and (epoch % max(1, cfg.unseen_eval_every) == 0):
            unseen_ber = trainer.evaluate(test_unseen_loaders)["avg_ber"]
        if cfg.model_selection_metric == "seen_plus_unseen" and unseen_ber is not None:
            score = val_ber + cfg.unseen_metric_weight * unseen_ber
        else:
            score = val_ber
        scheduler.step()

        # Summarize gradient norms across epoch
        epoch_grad_summary = defaultdict(list)
        for g in train_stats["grad_logs"]:
            for k, v in g["grad_named"].items():
                epoch_grad_summary[k].append(v)
        epoch_grad_summary = {k: float(np.mean(v)) for k, v in epoch_grad_summary.items()}

        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ber": train_stats["ber"],
            "val_seen_ber": val_ber,
            "per_config_train": train_stats["per_config"],
            "per_config_val": val_seen["per_config"],
            "per_config_margin": val_seen["per_config_margin"],
            "epoch_grad_summary": epoch_grad_summary,
            "shape_histogram": train_stats["shape_histogram"],
            "focus_otfs": focus_stats,
            "dynamic_otfs_boost": dynamic_otfs_boost,
            "unseen_ber_epoch": unseen_ber,
            "selection_score": score,
        }
        history.append(row)

        if cfg.enable_gradient_probe and epoch <= cfg.gradient_probe_batches:
            probe = trainer.gradient_probe(
                train_loader,
                epoch=epoch,
                use_focal_loss=cfg.use_focal_loss,
                label_smoothing=cfg.label_smoothing,
            )
            if probe is not None:
                gradient_probe_records.append(probe)

        trainer.multi_config_probe.run_epoch_probe(epoch, model, val_seen_loaders, device)

        print(
            f"[CommDebug][Epoch {epoch:03d}] "
            f"loss={train_stats['loss']:.4f} "
            f"train_ber={train_stats['ber']:.4e} "
            f"val_seen_ber={val_ber:.4e} "
            f"score={score:.4e}"
        )
        if unseen_ber is not None:
            print(f"  unseen_ber={unseen_ber:.4e}")
        if focus_stats["batches"] > 0:
            print(f"  otfs_focus: batches={focus_stats['batches']} loss={focus_stats['loss']:.4f} ber={focus_stats['ber']:.4e}")
        if cfg.enable_controlled_otfs_emphasis:
            print(f"  dynamic_otfs_boost={dynamic_otfs_boost:.4f}")
        print(f"  Grad summary: {epoch_grad_summary}")
        print(f"  Shape histogram: {train_stats['shape_histogram']}")

        if score < best["score"]:
            best = {"ber": val_ber, "score": score, "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "best": best,
                    "seen_configs": seen_configs,
                    "model_name": cfg.comm_model,
                    "config": asdict(cfg),
                },
                best_path,
            )
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[CommDebug] Early stopping at epoch {epoch}")
                break

    load_checkpoint_if_exists(model, best_path, device)
    final_seen = trainer.evaluate(val_seen_loaders)
    final_unseen = trainer.evaluate(test_unseen_loaders) if test_unseen_loaders else {"avg_ber": 1.0, "per_config": {}, "per_config_margin": {}}

    eval_dir = ensure_dir(os.path.join(out_dir, "eval"))
    snr_reports = {}
    for config_name in seen_configs:
        cfg_eval_dir = ensure_dir(os.path.join(eval_dir, config_name))
        for channel_mode in ["awgn", "realistic"]:
            key = f"{config_name}_{channel_mode}"
            print(f"[CommDebug] Evaluating SNR sweep for {key}")
            res = trainer.evaluate_snr_sweep(
                config_name=config_name,
                save_root=cfg.data_root,
                snr_list=list(cfg.eval_snr_list),
                num_samples=15,
                channel_mode=channel_mode,
            )
            snr_reports[key] = res
            plot_comm_sweep(res, os.path.join(cfg_eval_dir, f"ber_snr_{channel_mode}.png"), title=f"{config_name} - {channel_mode}")

    plot_comm_sweep_all(
        {k: v for k, v in snr_reports.items() if "awgn" in k},
        os.path.join(eval_dir, "comm_ber_snr_awgn_all_configs.png"),
        title="BER vs SNR (AWGN) - All Configs",
    )
    plot_comm_sweep_all(
        {k: v for k, v in snr_reports.items() if "realistic" in k},
        os.path.join(eval_dir, "comm_ber_snr_realistic_all_configs.png"),
        title="BER vs SNR (Realistic) - All Configs",
    )

    plot_history(history, os.path.join(debug_dir, "history.png"))
    plot_per_config_ber(history, os.path.join(debug_dir, "per_config_val_ber.png"))
    plot_grad_norms(history, os.path.join(debug_dir, "grad_norms.png"))

    trainer.batch_debugger.flush()
    trainer.activation_tracker.flush()
    trainer.multi_config_probe.flush()
    save_json({"gradient_probe": gradient_probe_records}, os.path.join(debug_dir, "gradient_probe.json"))

    summary = {
        "checkpoint": best_path,
        "param_count": count_parameters(model),
        "seen_configs": seen_configs,
        "unseen_configs": unseen_configs,
        "history": history,
        "seen": final_seen,
        "unseen": final_unseen,
        "snr_sweeps": snr_reports,
        "model_name": cfg.comm_model,
        "debug_dir": debug_dir,
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def _run_with_radar_support(cfg: RunConfig, device: torch.device) -> Dict[str, Any]:
    from AIradar_comm_model_g6 import RunConfig as G6RunConfig
    from AIradar_comm_model_g6 import train_radar_pipeline as g6_train_radar_pipeline
    from AIradar_comm_model_g6 import run_full_evaluation as g6_run_full_evaluation

    g6_keys = set(G6RunConfig.__dataclass_fields__.keys())
    payload = {k: v for k, v in asdict(cfg).items() if k in g6_keys}
    g6_cfg = G6RunConfig(**payload)

    if cfg.mode in ["train_comm_debug", "train_comm"]:
        return {"comm": train_comm_pipeline_debug(cfg, device)}
    if cfg.mode == "train_radar":
        return {"radar": g6_train_radar_pipeline(g6_cfg, device, model_name=cfg.radar_model)}
    if cfg.mode == "train_both":
        radar_summary = g6_train_radar_pipeline(g6_cfg, device, model_name=cfg.radar_model)
        comm_summary = train_comm_pipeline_debug(cfg, device)
        combo = {"radar": radar_summary, "comm": comm_summary}
        save_json(combo, os.path.join(cfg.out_dir, "summary_train_both.json"))
        return combo
    if cfg.mode == "eval_all":
        return g6_run_full_evaluation(g6_cfg, device, radar_model_name=cfg.radar_model, comm_model_name=cfg.comm_model)
    raise ValueError(f"Unsupported mode: {cfg.mode}")


# =============================================================================
# CLI
# =============================================================================



def parse_args() -> RunConfig:
    import argparse

    parser = argparse.ArgumentParser(description="Detailed debug communication training pipeline")
    parser.add_argument("--mode", choices=["train_comm_debug", "train_comm", "train_radar", "train_both", "eval_all"], default="train_comm_debug")
    parser.add_argument("--out_dir", default="data/g7_debug")
    parser.add_argument("--data_root", default="data/g7_debug")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=120)
    parser.add_argument("--val_samples", type=int, default=24)
    parser.add_argument("--test_samples", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--comm_type", choices=["OFDM", "OTFS", "all"], default="all")
    parser.add_argument("--qam_type", choices=["4QAM", "8QAM", "16QAM", "64QAM", "all"], default="all")
    parser.add_argument("--comm_ckpt", default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--unseen_holdout_fraction", type=float, default=0.34)
    parser.add_argument("--comm_model", choices=["isac"], default="isac")
    parser.add_argument("--radar_model", choices=["isac"], default="isac")
    parser.add_argument("--radar_type", choices=["FMCW", "OTFS", "all"], default="all")
    parser.add_argument("--radar_ckpt", default=None)
    parser.add_argument("--channel_mode", choices=["awgn", "realistic"], default="realistic")
    parser.add_argument("--radar_pos_weight", type=float, default=5.0)
    parser.add_argument("--radar_sigma", type=float, default=3.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--log_first_n_batches", type=int, default=5)
    parser.add_argument("--track_activations_every", type=int, default=1)
    parser.add_argument("--enable_gradient_probe", action="store_true")
    parser.add_argument("--gradient_probe_batches", type=int, default=2)
    parser.add_argument("--otfs_focus_start_epoch", type=int, default=0)
    parser.add_argument("--otfs_focus_batches_per_epoch", type=int, default=0)
    parser.add_argument("--otfs_loss_boost", type=float, default=1.0)
    parser.add_argument("--ofdm_loss_boost", type=float, default=1.0)
    parser.add_argument("--qam16_loss_boost", type=float, default=1.0)
    parser.add_argument("--qam64_loss_boost", type=float, default=1.0)
    parser.add_argument("--model_selection_metric", choices=["seen", "seen_plus_unseen"], default="seen")
    parser.add_argument("--unseen_metric_weight", type=float, default=0.25)
    parser.add_argument("--unseen_eval_every", type=int, default=1)
    parser.add_argument("--enable_controlled_otfs_emphasis", action="store_true")
    parser.add_argument("--otfs_ramp_epochs", type=int, default=10)
    args = parser.parse_args()
    return RunConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    _run_with_radar_support(cfg, device)

"""
python AIRadar/AIradar_comm_model_g6_commtrain2.py \
  --comm_type all \
  --qam_type all \
  --epochs 3 \
  --train_samples 20 \
  --val_samples 8 \
  --test_samples 8 \
  --batch_size 2 \
  --out_dir data/g7_debug_smokev2 \
  --data_root data/g7_debug_smokev2 \
  --enable_gradient_probe

# Full Comm Training
python AIRadar/AIradar_comm_model_g6_commtrain2.py \
    --mode train_comm --comm_type all \
    --qam_type all  \
    --out_dir data/g6_comm2 --data_root data/g6_comm2 \
    --train_samples 1000 --val_samples 200 \
    --test_samples 200 --epochs 50

#Balanced + controlled OTFS emphasis ON :
python AIRadar/AIradar_comm_model_g6_commtrain2.py \
  --mode train_comm \
  --comm_type all --qam_type all \
  --out_dir data/g6_comm3 --data_root data/g6_comm3 \
  --train_samples 1000 --val_samples 200 --test_samples 200 \
  --epochs 60 --batch_size 4 \
  --enable_controlled_otfs_emphasis \
  --otfs_focus_start_epoch 20 \
  --otfs_ramp_epochs 12 \
  --otfs_focus_batches_per_epoch 60 \
  --otfs_loss_boost 1.35 \
  --ofdm_loss_boost 1.0 \
  --qam16_loss_boost 1.05 \
  --qam64_loss_boost 1.15 \
  --model_selection_metric seen_plus_unseen \
  --unseen_metric_weight 0.25

#Turn OTFS emphasis OFF (ablation / baseline):
python AIRadar/AIradar_comm_model_g6_commtrain2.py \
  --mode train_comm \
  --comm_type all --qam_type all \
  --out_dir data/g6_comm3_base --data_root data/g6_comm3_base \
  --train_samples 1000 --val_samples 200 --test_samples 200 \
  --epochs 60 --batch_size 4

#Radar training from the same script :
python AIRadar/AIradar_comm_model_g6_commtrain2.py \
  --mode train_radar \
  --radar_type all \
  --comm_type all --qam_type all \
  --radar_model isac \
  --out_dir data/g6_comm3 --data_root data/g6_comm3 \
  --train_samples 1000 --val_samples 200 --test_samples 200 \
  --epochs 50
"""
