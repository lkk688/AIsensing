import argparse
from pathlib import Path
from .config import SystemParams
from .dataset import build_big_dataset
from .train import train_radar_model, train_multidomain_multitask
from .evaluate import evaluate_and_visualize, run_ber_sweep_and_plot
from .models.unet import UNetLite
import torch

def main():
    parser = argparse.ArgumentParser(description="ISAC Experiment Runner")
    parser.add_argument("--mode", type=str, default="train_radar", 
                        choices=["gen_data", "train_radar", "train_mdmt", "eval", "ber_sweep"],
                        help="Operation mode")
    parser.add_argument("--out_dir", type=str, default="./output/isac_experiment", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()
    
    sp = SystemParams()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "gen_data":
        print("Generating dataset...")
        build_big_dataset(out_dir, sp, n_train=1000, n_val=200)
        
    elif args.mode == "train_radar":
        print("Training Radar UNet...")
        train_radar_model(out_dir, out_dir/"checkpoints", epochs=args.epochs)
        
    elif args.mode == "train_mdmt":
        print("Training Multi-Domain Multi-Task Network...")
        train_multidomain_multitask(out_dir, sp, epochs=args.epochs, out_root=out_dir)
        
    elif args.mode == "eval":
        print("Evaluating...")
        # Load model
        net = UNetLite().to("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = out_dir/"checkpoints"/"radar_unet_best.pt"
        if ckpt.exists():
            net.load_state_dict(torch.load(ckpt))
            print("Loaded checkpoint.")
        else:
            print("No checkpoint found, using random weights.")
            
        evaluate_and_visualize(out_dir/"eval_results", sp, net)
        
    elif args.mode == "ber_sweep":
        print("Running BER Sweep...")
        run_ber_sweep_and_plot(
            out_dir/"ber_sweep.png",
            np.arange(0, 21, 2),
            ofdm_cfg=dict(Nfft=256, cp_len=32, n_ofdm_sym=400),
            otfs_cfg=dict(M=64, N=256, cp_len=32)
        )

if __name__ == "__main__":
    import numpy as np
    main()
