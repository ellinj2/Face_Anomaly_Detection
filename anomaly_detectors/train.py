from anomaly_detectors import AnomalyDetectorModel
from face_dataset import RFFaceDataset
from utils import getDistribution, histogramify

import os
import sys
from tqdm import tqdm
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
import multiprocessing as mp

def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model architecture to use")
    parser.add_argument("--load_model", action="store_true", help="Load model, using --model as file path")
    parser.add_argument("--data-path", type=str, required=True, help="Path to load data from")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity of model")
    parser.add_argument("--train-split", type=float, default=1.0, help="Ratio of data to use for training")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers for Ensemble methods (if -1, uses all workers available)")
    return parser

def main(args):
    data_folder = args.data_path
    n_jobs = args.num_workers if args.num_workers != -1 else mp.cpu_count()

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