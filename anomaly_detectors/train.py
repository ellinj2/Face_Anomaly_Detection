from anomaly_detectors import AnomalyDetectorModel
from face_dataset import RFFaceDataset
from utils import getDistribution, histogramify

import os
import sys
from tqdm import tqdm
from time import time

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model architecture to use")
    parser.add_argument("--load_model", action="store_true", help="Load model, using --model as file path")
    parser.add_argument("--data-path", type=str, required=True, help="Path to load data from")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity of model")
    return parser

def main(args):
    data_folder = args.data_path

    real_data_path = os.path.join(data_folder, "training_real")
    if not os.path.isdir(real_data_path):
        print(f"{data_folder} is missing a real subdirectory.")
        sys.exit()

    fake_data_path = os.path.join(data_folder, "training_fake")
    if not os.path.isdir(fake_data_path):
        print(f"{data_folder} is missing a fake subdirectory.")
        sys.exit()

    print("Loading data...")
    train_dataset = RFFaceDataset(args.data_path)

    model = AnomalyDetectorModel(args.model, **{"verbose" : args.verbose})

    print("Building training set...")
    X_real = []
    X_fake = []
    for (x, y, _, _) in tqdm(train_dataset):
        if not y:
            X_fake.append(x.numpy().reshape(-1))
            continue
        X_real.append(x.numpy().reshape(-1))
        
    print("Fitting model...")
    t1 = time()
    result = model.fit(X_real)
    t2 = time()
    print(f"It took {t2-t1} seconds to train")
    print(result)

    print("Scoring real...")
    real_scores = model.detector.score_samples(X_real)
    print("Scoring fake...")
    fake_scores = model.detector.score_samples(X_fake)
    print("Getting distributions...")
    real_distribution = getDistribution(real_scores)
    fake_distribution = getDistribution(fake_scores)

    print("Making histogram...")
    histogramify(real_distribution, fake_distribution, "Real and Fake Images (Flattened) Through OneClassSVM", "RaFF_flat_ocSVM_100")

if __name__ == "__main__":
	main(get_arg_parser().parse_args())