# Previous run: python train.py --model AE --latent-dim 100 --conv-layers 3 --lin-layers 0 --train-split 0.75 --data-path "../data/real_and_fake_face" --epochs 20 -c --batch 1 --num-workers 0 --images 10 --save-path "../ae_results" --model-path "../best_ae" --reset-images --image-size 3,100,100 --conv-in-channel 32 --conv-out-channel 128 --training-images

from anomaly_detectors import AnomalyDetectorModel
from face_dataset import RFFaceDataset
from utils import getDistribution, histogramify
from autoencoder import Encoder, Decoder, AutoEncoder

import os
import sys
from tqdm import tqdm
from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import torch
import torchvision


def img(string):
    try:
        return tuple([int(val) for val in string.split(',')])
    except Exception as e:
        print(f"WARNING : Improper image shape")
        raise e

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model architecture to use")
    parser.add_argument("--load_model", action="store_true", help="Load model, using --model as file path")
    parser.add_argument("--data-path", type=str, required=True, help="Path to load data from")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity of model")
    parser.add_argument("--train-split", type=float, default=1.0, help="Ratio of data to use for training")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for Ensemble methods and dataloading (if -1, uses all workers available)")
    parser.add_argument("--latent-dim", type=int, default=200, help="Latent space dimension")
    parser.add_argument("--conv-layers", type=int, default=3, help="Number of hidden convolutional layers in encoder and decoder")
    parser.add_argument("--lin-layers", type=int, default=3, help="Number of hidden linear layers in encoder and decoder")
    parser.add_argument("-c", "--cuda", action="store_true", help="Flag to use CUDA if available")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for dataloading")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run")
    parser.add_argument("-i", "--images", type=int, default=0, help="Number of images to generate at end of training")
    parser.add_argument("--save-path", default="", help="Path to save generated images")
    parser.add_argument("--model-path", default="", help="Path to save best models")
    parser.add_argument("--load-model", action="store_true", help="Flag to load the model")
    parser.add_argument("--reset-images", action="store_true", help="Delete all images in save-path")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--image-size", type=img, default="3,100,100", help="Shape of input images (C,W,H)")
    parser.add_argument("--conv-in-channel", type=int, default=3, help="Minimum channel size for convolutional layers")
    parser.add_argument("--conv-out-channel", type=int, default=32, help="Maximum channel size for convolutional layers")
    parser.add_argument("--channel-growth", type=int, default=-1, help="Multiplicative rate for convolutional channels (Default -1 calculates uniform growth)")
    parser.add_argument("--training-images", action="store_true", help="Save images while training (will be saved to save-path/epoch_<epoch>.jpg/)")
    return parser

def neuralNet(data_path, **kwargs):
    latent_dim = kwargs["latent_dim"]
    conv_layers = kwargs["conv_layers"]
    lin_layers = kwargs["lin_layers"]
    split = kwargs["split"]
    batch_size = kwargs["batch_size"]
    epochs = kwargs["epochs"]
    device = kwargs["device"]
    print(f"Running on {device}")
    workers = mp.cpu_count() if kwargs["num_workers"] == -1 else kwargs["num_workers"]
    to_generate = kwargs["images"]
    save_path = kwargs["save_path"]
    model_path = kwargs["model_path"]
    load = kwargs["load_model"]
    reset_imgs = kwargs["reset_images"]
    lr = kwargs["learning_rate"]
    image_size = tuple([1] + list(kwargs["image_size"]))
    conv_in_channel = kwargs["conv_in_channel"]
    conv_out_channel = kwargs["conv_out_channel"]
    channel_growth = kwargs["channel_growth"]
    training_images = kwargs["training_images"]

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if reset_imgs:
        for file in os.listdir(save_path):
            os.remove(f"{save_path}/{file}")

    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    autoencoder = AutoEncoder(latent_dim, conv_layers, lin_layers, image_shape=image_size, conv_in_channel=conv_in_channel, conv_out_channel=conv_out_channel, channel_growth=channel_growth, verbose=True)
    # encoder = Encoder(latent_dim, conv_layers=conv_layers, lin_layers=lin_layers, input_shape=image_size, verbose=True)
    # decoder = Decoder(latent_dim, conv_layers=conv_layers, lin_layers=lin_layers, lin_out_dim=encoder.lin_in_shape[1], first_conv_size=encoder.last_conv_size, image_shape=image_size, verbose=True)
    if load:
        autoencoder.encoder = torch.load(f"{model_path}/best_encoder.pt")
        autoencoder.decoder = torch.load(f"{model_path}/best_decoder.pt")
        # encoder.model = torch.load(f"{model_path}/best_encoder.pt")
        # decoder.model = torch.load(f"{model_path}/best_decoder.pt")
    # encoder.to(device)
    # decoder.to(device)
    autoencoder.to(device)

    print("Loading data...")
    train_real_dataset = RFFaceDataset(data_path, label=1, train_test_split=split, image_size=tuple(list(image_size)[2:]), batch="train")
    valid_real_dataset = RFFaceDataset(data_path, label=1, train_test_split=split, image_size=tuple(list(image_size)[2:]), batch="validation")

    train_fake_dataset = RFFaceDataset(data_path, label=0, train_test_split=split, image_size=tuple(list(image_size)[2:]), batch="train")
    valid_fake_dataset = RFFaceDataset(data_path, label=0, train_test_split=split, image_size=tuple(list(image_size)[2:]), batch="validation")

    train_loader = torch.utils.data.DataLoader(train_real_dataset, batch_size, shuffle=True, num_workers=workers, persistent_workers=workers>0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_real_dataset, batch_size, shuffle=True, num_workers=workers, persistent_workers=workers>0, pin_memory=True)
    optim = torch.optim.Adam(list(autoencoder.encoder.parameters()) + list(autoencoder.decoder.parameters()), lr)
    criterion = torch.nn.MSELoss()

    losses = []
    v_losses = []
    t = tqdm(range(epochs))
    loss = np.inf
    best_loss = np.inf
    t.set_description(f"Training Loss: {loss}")
    for epoch in t:
        loss = 0
        for images, _, _, _ in tqdm(train_loader, leave=False):
            images = images.to(device)
            optim.zero_grad()
            decoded = autoencoder(images)
            # encoded = encoder(images)
            # decoded = decoder(encoded)

            train_loss = criterion(decoded, images)
            train_loss.backward()

            optim.step()
            loss += train_loss.item()

        loss = loss / len(train_loader)
        losses.append(loss)

        valid_loss = 0
        for images, _, _, _ in tqdm(valid_loader, leave=False):
            images = images.to(device)
            with torch.no_grad():
                decoded = autoencoder(images)
                # encoded = encoder(images)
                # decoded = decoder(encoded)
                valid_loss += criterion(decoded, images).item()
        valid_loss /= len(valid_loader)

        if valid_loss < best_loss:
            torch.save(autoencoder.encoder, f"{model_path}/best_encoder.pt")
            torch.save(autoencoder.encoder, f"{model_path}/epoch_{epoch+1}_encoder.pt")
            torch.save(autoencoder.decoder, f"{model_path}/best_decoder.pt")
            torch.save(autoencoder.decoder, f"{model_path}/epoch_{epoch+1}_decoder.pt")
            # torch.save(encoder.model, f"{model_path}/best_encoder.pt")
            # torch.save(encoder.model, f"{model_path}/epoch_{epoch+1}_encoder.pt")
            # torch.save(decoder.model, f"{model_path}/best_decoder.pt")
            # torch.save(decoder.model, f"{model_path}/epoch_{epoch+1}_decoder.pt")

            best_loss = valid_loss

        if training_images:
            image, _, _, _ = next(iter(valid_loader))
            image = image.to(device)
            decoded = autoencoder(image)
            torchvision.utils.save_image(decoded, f"{save_path}/epoch_{epoch}.jpg")

        v_losses.append(valid_loss)
        t.set_description(f"Training Loss: {loss}, Validation loss: {valid_loss}")

    print(pd.DataFrame({"Epoch" : [i for i in range(epochs)], "Training Loss" : losses, "Validation Loss" : v_losses}))

    decodings = []
    encodings = []

    autoencoder.encoder = torch.load(f"{model_path}/best_encoder.pt")
    autoencoder.decoder = torch.load(f"{model_path}/best_decoder.pt")
    # encoder.model = torch.load(f"{model_path}/best_encoder.pt")
    # decoder.model = torch.load(f"{model_path}/best_decoder.pt")

    generation_loader = iter(torch.utils.data.DataLoader(valid_real_dataset, 1, shuffle=True))
    for i in tqdm(range(to_generate//2)):
        image, _, _, _ = next(generation_loader)
        image = image.to(device)
        img_name = valid_real_dataset.image_names[i]
        torchvision.utils.save_image(image, f"{save_path}/original_{img_name}")

        decoded = autoencoder(image)
        # encoded = encoder(image)
        # encodings.append(encoded)
        # decoded = decoder(encoded)
        # decodings.append(decoded)
        
        torchvision.utils.save_image(decoded, f"{save_path}/decoded_{img_name}")

    generation_loader = iter(torch.utils.data.DataLoader(valid_fake_dataset, 1, shuffle=True))
    for i in tqdm(range(to_generate//2)):
        image, _, _, _ = next(generation_loader)
        image = image.to(device)
        img_name = valid_fake_dataset.image_names[i]
        torchvision.utils.save_image(image, f"{save_path}/original_{img_name}")

        decoded = autoencoder(image)
        # encoded = encoder(image)
        # encodings.append(encoded)
        # decoded = decoder(encoded)
        # decodings.append(decoded)

        torchvision.utils.save_image(decoded, f"{save_path}/decoded_{img_name}")

    # for e, d in zip(encodings, decodings):
    #     print(d)

def main(args):
    data_folder = args.data_path
    n_jobs = args.num_workers if args.num_workers != -1 else mp.cpu_count()
    device = "cuda:0" if args.cuda and torch.cuda.is_available() else "cpu"

    real_data_path = os.path.join(data_folder, "training_real")
    if not os.path.isdir(real_data_path):
        print(f"{data_folder} is missing a real subdirectory.")
        sys.exit()

    fake_data_path = os.path.join(data_folder, "training_fake")
    if not os.path.isdir(fake_data_path):
        print(f"{data_folder} is missing a fake subdirectory.")
        sys.exit()

    kwargs = dict()
    if args.model == "isolation forest":
        kwargs["n_jobs"] = args.num_workers
        kwargs["verbose"] = args.verbose
    if args.model in ["CNN", "AE"]:
        kwargs["latent_dim"] = args.latent_dim
        kwargs["conv_layers"] = args.conv_layers
        kwargs["lin_layers"] = args.lin_layers
        kwargs["device"] = device
        kwargs["split"] = args.train_split
        kwargs["batch_size"] = args.batch_size
        kwargs["epochs"] = args.epochs
        kwargs["num_workers"] = args.num_workers
        kwargs["images"] = args.images
        kwargs["save_path"] = args.save_path
        kwargs["model_path"] = args.model_path
        kwargs["load_model"] = args.load_model
        kwargs["reset_images"] = args.reset_images
        kwargs["learning_rate"] = args.learning_rate
        kwargs["image_size"] = args.image_size
        kwargs["conv_in_channel"] = args.conv_in_channel
        kwargs["conv_out_channel"] = args.conv_out_channel
        kwargs["channel_growth"] = args.channel_growth
        kwargs["training_images"] = args.training_images

    if args.model in ["CNN", "AE"]:
        return neuralNet(data_folder, **kwargs)

    print("Loading data...")
    train_dataset = RFFaceDataset(args.data_path)

    model = AnomalyDetectorModel(args.model, **kwargs)

    print("Building training set...")
    X_real = []
    X_fake = []
    for (x, y, _, _) in tqdm(train_dataset):
        if not y:
            X_fake.append(x.numpy().reshape(-1))
            continue
        X_real.append(x.numpy().reshape(-1))
    
    X_real_train, X_real_test = train_test_split(X_real, train_size=args.train_split)
    X_fake_train, X_fake_test = train_test_split(X_fake, train_size=args.train_split)

    if args.model == "local outlier factor":
        print("Fitting and estimating...")
        result = model.fit_predict(X_real + X_fake)
        real_scores = result[:len(X_real)]
        fake_scores = result[len(X_real):]
        print(real_scores, fake_scores, sep='\n')
    else:
        print("Fitting model...")
        t1 = time()
        result = model.fit(X_real_train)
        t2 = time()
        print(f"It took {t2-t1} seconds to train")
        print(result)

        print("Scoring real...")
        real_scores = [model.detector.score_samples([x])[0] for x in tqdm(X_real_test)]
        print("Scoring fake...")
        fake_scores = [model.detector.score_samples([x])[0] for x in tqdm(X_fake_test)]

    print("Getting distributions...")
    real_distribution = getDistribution(real_scores)
    fake_distribution = getDistribution(fake_scores)

    print("Making histogram...")
    _model = '_'.join(args.model.split(' '))
    histogramify(real_scores, fake_scores, f"Real and Fake Images (Flattened) Through {args.model.title()}", f"RaFF_flat_{_model}_all")

if __name__ == "__main__":
    main(get_arg_parser().parse_args())