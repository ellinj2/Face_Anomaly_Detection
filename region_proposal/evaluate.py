import os
import json
import numpy as np
import matplotlib.pyplot as plt    

def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a region proposal network.")

    parser.add_argument("--history_path", type=str, required=True, help="Path to model training history folder.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to output plots too.")

    return parser

def main(args):
    hist_path = args.history_path
    out_path = args.out_path

    with open(hist_path, "r") as f:
        hist = json.load(f)

    N = len(hist["train_loss"])
    plt.plot(np.arange(1, len(hist["train_loss"])+1), hist["train_loss"])
    plt.title("FCOS+Resnet18: Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(np.arange(1, N+1))
    plt.savefig(os.path.join(out_path, "loss_plot.png"))

    plt.plot(np.arange(1, N+1), hist["train_map_50"], label="Train")
    plt.plot(np.arange(1, N+1), hist["valid_map_50"], label="Validation")
    plt.title("FCOS+Resnet18: AP with IoU Threshold at 0.5")
    plt.xlabel("Epochs")
    plt.ylabel("AP@IoU:0.5")
    plt.xticks(np.arange(1, N+1))
    plt.legend()
    plt.savefig(os.path.join(out_path, "ap_plot.png"))

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
