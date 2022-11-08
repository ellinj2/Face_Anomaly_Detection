import os
import sys
import json
from glob import glob

from face_dataset import WFFaceDataset
from region_proposal_network import RegionProposalNetwork

from torch.optim import Adam

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train the region proposal network.")
    parser.add_argument("data_path", type=str, help="String representing the desired object detection model. Avaiable options are 'retinanet' and 'fcos'.")
    parser.add_argument("model_type", type=str, help="String representing the desired resnet backbone for detector. Avaiable options are '18', '34', '50', '101', and '152'.")
    parser.add_argument("backbone_type", type=str, help="Path to folder with a train and validation subfolders.")
    parser.add_argument("epochs", type=int, help="Number of training iterations over the data.")
    parser.add_argument("batch_size", type=int, help="Number of images to batch for training and evaluaton.")
    parser.add_argument("learning_rate", type=float, help="Float reprisenting the learning rate.")
    parser.add_argument("save_path", type=str, help="Path to save model checkpoints.")
    parser.add_argument("checkpoints", type=int, default=0, help="Integer N reprisenting after every N epochs to create a model checkpoint. If 0, only save the best model. (Default: 0)")

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

    valid_data_path = os.path.join(data_folder, "validation")
    if not os.path.isdir(valid_data_path):
        print(f"{data_folder} is missing a validation subdirectory.")

    valid_txt = glob(os.path.join(valid_data_path, "*.txt"))

    if len(valid_txt) != 1:
        print(f"{valid_data_path} should contain one txt file.")
        sys.exit()

    valid_txt = valid_txt[0]

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    print("Loading Training Dataset...")
    train_dataset = WFFaceDataset(train_txt, train_data_path)
    print("Loading Validation Dataset...")
    valid_dataset = WFFaceDataset(valid_txt, valid_data_path)

    model = RegionProposalNetwork(model_type, backbone_type)
    optim = Adam(model.parameters(), lr)

    hist = model.fit(epochs=epochs, 
                datasets=(train_dataset, valid_dataset), 
                batch_size=batch_size, 
                optimizer=optim, 
                save_path=save_path, 
                checkpoints=checkpoints, 
                progress=True)

    with open(os.path.join(save_path, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=1)

    print("Done")

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)
