import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class JointRadarCommDisk(Dataset):
    def __init__(self, base_dir, configs=('CN0566_TRADITIONAL','Automotive_77GHz_LongRange','XBand_10GHz_MediumRange','AUTOMOTIVE_TRADITIONAL'), normalize=True):
        self.samples = []
        self.normalize = normalize
        for cfg in configs:
            npy_path = os.path.join(base_dir, cfg, 'joint_dump.npy')
            if os.path.exists(npy_path):
                arr = np.load(npy_path, allow_pickle=True)
                for item in arr:
                    self.samples.append(item)
        if len(self.samples) == 0:
            raise RuntimeError('No joint_dump.npy found')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rdm = s['range_doppler_map']
        ofdm_map = s.get('ofdm_map', None)
        comm = s.get('comm_info', None)
        cfar_list = s['cfar_detections']
        dr, rr = rdm.shape
        mask = np.zeros((dr, rr), dtype=np.float32)
        for d in cfar_list:
            di = int(d['doppler_idx']); ri = int(d['range_idx'])
            if 0 <= di < dr and 0 <= ri < rr:
                mask[di, ri] = 1.0
        rdm_in = rdm
        if self.normalize:
            rdm_in = rdm_in - np.max(rdm_in)
            rdm_in = np.clip(rdm_in, -60, 0)
        rdm_in = (rdm_in + 60.0) / 60.0
        rdm_in = rdm_in.astype(np.float32)
        if ofdm_map is not None:
            om = ofdm_map
            if self.normalize:
                om = om - np.max(om)
                om = np.clip(om, -60, 0)
            om = (om + 60.0) / 60.0
            ofdm_in = om.astype(np.float32)
        else:
            ofdm_in = None
        sample = {
            'rdm': torch.from_numpy(rdm_in)[None, ...],
            'mask': torch.from_numpy(mask)[None, ...],
            'has_ofdm': ofdm_in is not None,
        }
        if ofdm_in is not None:
            sample['ofdm'] = torch.from_numpy(ofdm_in)[None, ...]
        if comm is not None:
            sample['comm_ber'] = float(comm.get('ber', 0.0))
            mod = int(comm.get('mod_order', 4))
            tx_ints = comm.get('tx_ints', None)
            if ofdm_in is not None and tx_ints is not None and mod == 4:
                arr = np.asarray(tx_ints).astype(np.int32)
                n_syms = 8*32
                if arr.size < n_syms:
                    pad = np.zeros(n_syms - arr.size, dtype=np.int32)
                    arr = np.concatenate([arr, pad], axis=0)
                else:
                    arr = arr[:n_syms]
                bits = ((arr[:, None] >> np.array([1,0], dtype=np.int32)) & 1).astype(np.float32)
                bits = bits.T.reshape(2, 8, 32)
                sample['tx_bits'] = torch.from_numpy(bits)
                sample['tx_valid'] = torch.tensor(1.0, dtype=torch.float32)
            else:
                sample['tx_valid'] = torch.tensor(0.0, dtype=torch.float32)
        return sample

def _pad_tensor(x, target_h, target_w):
    h, w = x.shape[-2], x.shape[-1]
    if h == target_h and w == target_w:
        return x
    out = torch.zeros((x.shape[0], target_h, target_w), dtype=x.dtype)
    out[:, :h, :w] = x
    return out

def collate_joint(batch):
    max_h = max(item['rdm'].shape[-2] for item in batch)
    max_w = max(item['rdm'].shape[-1] for item in batch)
    rdm = torch.stack([_pad_tensor(item['rdm'], max_h, max_w) for item in batch], dim=0)
    mask = torch.stack([_pad_tensor(item['mask'], max_h, max_w) for item in batch], dim=0)
    have_ofdm = all(item.get('has_ofdm', False) for item in batch)
    out = {'rdm': rdm, 'mask': mask}
    if have_ofdm:
        oh = max(item['ofdm'].shape[-2] for item in batch)
        ow = max(item['ofdm'].shape[-1] for item in batch)
        ofdm = torch.stack([_pad_tensor(item['ofdm'], oh, ow) for item in batch], dim=0)
        out['ofdm'] = ofdm
    # Always collate tx_bits with zeros fallback and a validity mask
    any_tx = any('tx_bits' in item for item in batch)
    tx_bits_list = []
    tx_valid_list = []
    for item in batch:
        if 'tx_bits' in item:
            tx_bits_list.append(item['tx_bits'])
            tx_valid_list.append(item.get('tx_valid', torch.tensor(1.0)))
        else:
            tx_bits_list.append(torch.zeros((2,8,32), dtype=torch.float32))
            tx_valid_list.append(torch.tensor(0.0, dtype=torch.float32))
    out['tx_bits'] = torch.stack(tx_bits_list, dim=0)
    out['tx_valid'] = torch.stack(tx_valid_list, dim=0)
    if any('comm_ber' in item for item in batch):
        vals = []
        for item in batch:
            v = item.get('comm_ber', 0.0)
            vals.append(float(v))
        out['comm_ber'] = torch.tensor(vals, dtype=torch.float32)
    return out

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.b1 = ConvBNAct(ch, ch)
        self.b2 = ConvBNAct(ch, ch)
    def forward(self, x):
        return x + self.b2(self.b1(x))

class SharedBackbone(nn.Module):
    def __init__(self, in_ch=1, base=32, depth=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, base)
        blocks = []
        ch = base
        for _ in range(depth):
            blocks += [ResidualBlock(ch), ConvBNAct(ch, ch*2, s=2)]
            ch *= 2
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = ch
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

class RadarHeader(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(ConvBNAct(in_ch, in_ch), nn.Conv2d(in_ch, 1, 1))
    def forward(self, x):
        return self.head(x)

class OFDMHeader(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((8, 32))
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(in_ch*8*32, 512), nn.ReLU(True), nn.Linear(512, 2*8*32))
    def forward(self, x):
        z = self.pool(x)
        y = self.mlp(z)
        return y.view(x.size(0), 2, 8, 32)

class JointNet(nn.Module):
    def __init__(self, base=32, depth=3):
        super().__init__()
        self.backbone = SharedBackbone(in_ch=1, base=base, depth=depth)
        self.radar_head = RadarHeader(self.backbone.out_ch)
        self.ofdm_head = OFDMHeader(self.backbone.out_ch)
    def forward(self, rdm=None, ofdm=None):
        out = {}
        if rdm is not None:
            f = self.backbone(rdm)
            out['radar_logits'] = self.radar_head(f)
        if ofdm is not None:
            g = self.backbone(ofdm)
            out['ofdm_bits'] = self.ofdm_head(g)
        return out

def loss_fn(outputs, batch):
    device = next(iter(outputs.values())).device if len(outputs)>0 else torch.device('cpu')
    loss = torch.tensor(0.0, device=device)
    if 'radar_logits' in outputs:
        logits = outputs['radar_logits']
        target = batch['mask'].to(logits.device)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
        pos = torch.clamp(target.sum(), min=1.0)
        neg = torch.clamp(torch.tensor(target.numel(), device=target.device) - target.sum(), min=1.0)
        pos_weight = neg / pos
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        loss = loss + bce
    if 'ofdm_bits' in outputs and 'ofdm' in batch:
        bits_pred = outputs['ofdm_bits']
        target = batch.get('tx_bits', None)
        valid = batch.get('tx_valid', None)
        if target is not None and valid is not None:
            pred_sig = torch.sigmoid(bits_pred)
            diff = (pred_sig - target.to(bits_pred.device))**2
            per_sample = diff.mean(dim=(1,2,3))
            weights = valid.to(bits_pred.device)
            denom = torch.clamp(weights.sum(), min=1.0)
            mse = (per_sample * weights).sum() / denom
        else:
            pseudo = torch.zeros_like(bits_pred)
            mse = F.mse_loss(bits_pred, pseudo)
        loss = loss + mse
    return loss

def train_one_epoch(model, loader, opt, device):
    model.train()
    tot = 0.0; n = 0
    for batch in loader:
        rdm = batch['rdm'].to(device)
        ofdm = batch.get('ofdm'); ofdm = ofdm.to(device) if ofdm is not None else None
        outputs = model(rdm=rdm, ofdm=ofdm)
        loss = loss_fn(outputs, batch)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); n += 1
    return tot/max(n,1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    pr = []
    ber_dl = []
    ber_base = []
    for batch in loader:
        rdm = batch['rdm'].to(device)
        ofdm = batch.get('ofdm', None)
        ofdm = ofdm.to(device) if ofdm is not None else None
        outputs = model(rdm=rdm, ofdm=ofdm)
        logits = outputs['radar_logits']
        mask = batch['mask'].numpy()
        if list(logits.shape[-2:]) != list(mask.shape[-2:]):
            logits = F.interpolate(logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        probs = torch.sigmoid(logits).cpu().numpy()
        tp = np.sum((probs>0.5)*(mask>0.5)); fp = np.sum((probs>0.5)*(mask<=0.5)); fn = np.sum((probs<=0.5)*(mask>0.5))
        precision = tp/(tp+fp) if tp+fp>0 else 0.0
        recall = tp/(tp+fn) if tp+fn>0 else 0.0
        pr.append((precision, recall))
        if ofdm is not None and 'tx_bits' in batch:
            pred = torch.sigmoid(outputs['ofdm_bits']).cpu().numpy()
            gt = batch['tx_bits'].numpy()
            vmask = batch.get('tx_valid', None)
            pb = (pred>0.5).astype(np.float32).flatten()
            gb = gt.flatten()
            if gb.size == pb.size and gb.size > 0:
                if vmask is not None:
                    vs = vmask.numpy().flatten()
                    if np.sum(vs) > 0:
                        # Compute only on valid samples
                        bsz = pred.shape[0]
                        per_sample = []
                        for i in range(bsz):
                            if vs[i] > 0.5:
                                pbi = (pred[i]>0.5).astype(np.float32).flatten()
                                gbi = gt[i].flatten()
                                per_sample.append(np.mean(pbi != gbi))
                        if len(per_sample)>0:
                            ber_dl.append(float(np.mean(per_sample)))
                else:
                    ber_dl.append(np.mean(pb != gb))
        if 'comm_ber' in batch:
            val = batch['comm_ber']
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            ber_base.append(np.mean(np.asarray(val)))
    pr_mean = np.mean(pr, axis=0) if len(pr)>0 else (0.0,0.0)
    ber_dl_mean = float(np.mean(ber_dl)) if len(ber_dl)>0 else 0.0
    ber_base_mean = float(np.mean(ber_base)) if len(ber_base)>0 else 0.0
    return pr_mean[0], pr_mean[1], ber_dl_mean, ber_base_mean

def visualize_batch(model, batch, save_dir, suffix=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    rdm = batch['rdm']
    logits_t = model(rdm=rdm.to(next(model.parameters()).device))['radar_logits']
    if list(logits_t.shape[-2:]) != list(rdm.shape[-2:]):
        logits_t = F.interpolate(logits_t, size=rdm.shape[-2:], mode='bilinear', align_corners=False)
    logits = logits_t.cpu().detach().numpy()[0,0]
    gt = batch['mask'][0,0].numpy()
    inp = rdm[0,0].numpy()
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1); plt.imshow(inp, aspect='auto', origin='lower', cmap='viridis'); plt.colorbar()
    plt.subplot(1,3,2); plt.imshow(logits, aspect='auto', origin='lower', cmap='magma'); plt.colorbar()
    plt.subplot(1,3,3); plt.imshow(gt, aspect='auto', origin='lower', cmap='gray'); plt.colorbar()
    name = 'dl_vs_gt.png' if suffix is None else f'dl_vs_gt_epoch_{suffix}.png'
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, name)); plt.close()

def compare_with_traditional(sample, model, save_dir, suffix=None):
    os.makedirs(save_dir, exist_ok=True)
    rdm = sample['rdm'][0,0].numpy()
    gt = sample['mask'][0,0].numpy()
    logits_t = model(rdm=sample['rdm'].to(next(model.parameters()).device))['radar_logits']
    if list(logits_t.shape[-2:]) != list(sample['rdm'].shape[-2:]):
        logits_t = F.interpolate(logits_t, size=sample['rdm'].shape[-2:], mode='bilinear', align_corners=False)
    logits = logits_t.cpu().detach().numpy()[0,0]
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.imshow(rdm, aspect='auto', origin='lower', cmap='viridis'); plt.colorbar()
    plt.subplot(2,2,2); plt.imshow(gt, aspect='auto', origin='lower', cmap='gray'); plt.colorbar()
    plt.subplot(2,2,3); plt.imshow(logits, aspect='auto', origin='lower', cmap='magma'); plt.colorbar()
    plt.subplot(2,2,4); plt.imshow((logits>0.5).astype(np.float32), aspect='auto', origin='lower', cmap='gray'); plt.colorbar()
    name = 'compare_dl_vs_traditional.png' if suffix is None else f'compare_dl_vs_traditional_epoch_{suffix}.png'
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, name)); plt.close()

def visualize_comm(sample, model, save_dir, suffix=None):
    os.makedirs(save_dir, exist_ok=True)
    dev = next(model.parameters()).device
    ofdm = sample.get('ofdm', None)
    if ofdm is None:
        return
    out = model(ofdm=ofdm.to(dev))
    pred = torch.sigmoid(out['ofdm_bits']).cpu().detach().numpy()[0]
    txb = sample.get('tx_bits', None)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1); plt.imshow(sample['ofdm'][0,0].numpy(), aspect='auto', origin='lower', cmap='viridis'); plt.colorbar()
    plt.subplot(1,2,2); plt.imshow(pred[0], aspect='auto', origin='lower', cmap='magma'); plt.colorbar()
    name = 'comm_epoch.png' if suffix is None else f'comm_epoch_{suffix}.png'
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, name)); plt.close()

def main():
    base_dir = 'data/AIradar_comm_dataset_g1'
    configs = ('CN0566_TRADITIONAL','Automotive_77GHz_LongRange','XBand_10GHz_MediumRange','AUTOMOTIVE_TRADITIONAL')
    dataset = JointRadarCommDisk(base_dir, configs=configs)
    n_train = int(0.8*len(dataset))
    idxs = list(range(len(dataset)))
    train_subset = torch.utils.data.Subset(dataset, idxs[:n_train])
    val_subset = torch.utils.data.Subset(dataset, idxs[n_train:])
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=collate_joint)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=collate_joint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointNet(base=32, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best_sum = 0.0
    out_dir = os.path.join(base_dir, 'dl_joint')
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(10):
        loss = train_one_epoch(model, train_loader, opt, device)
        p, r, ber_dl, ber_base = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={loss:.4f} precision={p:.3f} recall={r:.3f} ber_dl={ber_dl:.4e} ber_base={ber_base:.4e}")
        if p+r > best_sum:
            best_sum = p+r
            torch.save(model.state_dict(), os.path.join(out_dir,'best.pt'))
        batch = next(iter(val_loader))
        visualize_batch(model, batch, save_dir=out_dir, suffix=str(epoch))
        compare_with_traditional(batch, model, save_dir=out_dir, suffix=str(epoch))
        visualize_comm(batch, model, save_dir=out_dir, suffix=str(epoch))
    print(out_dir)

if __name__ == '__main__':
    main()
