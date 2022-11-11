import os
import sys
import json
from glob import glob
import multiprocessing as mp

from face_dataset import WFFaceDataset
from region_proposal_network import RegionProposalNetwork

from torch.optim import Adam
import torch

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train the region proposal network.")
    parser.add_argument("--data_path", type=str, help="Path to folder with a train and validation subfolders.")
    parser.add_argument("--model_type", type=str, help="String representing the desired object detection model. Avaiable options are 'retinanet' and 'fcos'.")
    parser.add_argument("--backbone_type", type=str, help="String representing the desired resnet backbone for detector. Avaiable options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.")
    parser.add_argument("--epochs", type=int, help="Number of training iterations over the data.")
    parser.add_argument("--batch_size", type=int, help="Number of images to batch for training and evaluaton.")
    parser.add_argument("--learning_rate", type=float, help="Float reprisenting the learning rate.")
    parser.add_argument("--save_path", type=str, help="Path to save model checkpoints and training history.")
    parser.add_argument("--checkpoints", type=int, default=0, help="Integer N reprisenting after every N epochs to create a model checkpoint. If 0, only save the best model. (Default: 0)")
    parser.add_argument("-c", "--cuda", action="store_true", help="Set flag if model should be loaded and trained on a GPU. By default the model will run on cpu.")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="Number of workers to load data. (Defualt multiprocessing.cpu_count())")

    return parser

def main(args): 
    data_folder = args.data_path
    model_type = args.model_type
    backbone_type = args.backbone_type
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    save_path = args.save_path
    checkpoints = args.checkpoints
    cuda = args.cuda

    save_path = f"{save_path}/{model_type}/{backbone_type}"

    # check command line args...
    if epochs <= 0:
        print(f"Epochs must be a postive integer but was {epochs}.")
        sys.exit()
    
    if batch_size <= 0:
        print(f"Batch size must be a postive integer but was {batch_size}.")
        sys.exit()

    if checkpoints < 0:
        print(f"Checkpoints must be a non-negative integer but was {checkpoints}.")
        sys.exit()

    if not os.path.isdir(data_folder):
        print(f"{data_folder} is not a valid directory.")
        sys.exit()

    train_data_path = os.path.join(data_folder, "train")
    if not os.path.isdir(train_data_path):
        print(f"{data_folder} is missing a train subdirectory.")
        sys.exit()

    train_txt = glob(os.path.join(train_data_path, "*.txt"))
    if len(train_txt) != 1:
        print(f"{train_data_path} should contain one txt file.")
        sys.exit()
    train_txt = train_txt[0]

    train_mat = glob(os.path.join(train_data_path, "*.mat"))
    if len(train_mat) != 1:
        print(f"{train_data_path} should contain one txt file.")
        sys.exit()
    train_mat = train_mat[0]

    valid_data_path = os.path.join(data_folder, "validation")
    if not os.path.isdir(valid_data_path):
        print(f"{data_folder} is missing a validation subdirectory.")
    valid_txt = glob(os.path.join(valid_data_path, "*.txt"))

    if len(valid_txt) != 1:
        print(f"{valid_data_path} should contain one txt file.")
        sys.exit()
    valid_txt = valid_txt[0]

    valid_mat = glob(os.path.join(valid_data_path, "*.mat"))
    if len(valid_mat) != 1:
        print(f"{valid_data_path} should contain one txt file.")
        sys.exit()
    valid_mat = valid_mat[0]

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # load datasets.
    print("Loading Training Dataset...")
    train_dataset = WFFaceDataset(train_txt, train_mat, train_data_path)
    print("Loading Validation Dataset...")
    valid_dataset = WFFaceDataset(valid_txt, valid_mat, valid_data_path)

    model = RegionProposalNetwork(model_type, backbone_type)
    if cuda:
        model.to("cuda:0")

    optim = Adam(model.parameters(), lr)

    hist = model.fit(epochs=epochs, 
                datasets=(train_dataset, valid_dataset), 
                batch_size=batch_size, 
                optimizer=optim, 
                save_path=save_path, 
                checkpoints=checkpoints, 
                progress=True,
                num_workers=args.num_workers)

    # save training results.
    with open(os.path.join(save_path, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=1)

    print("Done")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = get_arg_parser().parse_args()
    main(args)
