import os
import sys
from tqdm import tqdm

import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ToPILImage

from adifi import ADIFI
from region_proposal.utils import get_image_paths

import pandas as pd

def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a ADIFI.")

    parser.add_argument("--model-path", type=str, required=True, help="Path to model folder.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data folder.")
    parser.add_argument("--out-path", type=str, required=True, help="Path to output results too.")
    parser.add_argument("--epsilon", type=float, default=0.5, help="Negative log likelihood threshold")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of images to batch while running inference. (Default: 1)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Set flag if images are in subfolders under data_path.")
    parser.add_argument("-i", "--images", action="store_true", help="Set flag if image with bounding boxes is desired. Worker is recommended for large number of images as script speed decresses substationally.")
    parser.add_argument("-c", "--cuda", action="store_true", help="Set flag if model should be loaded and trained on a GPU. By default the model will run on cpu.")

    return parser

def main(args):
    data_path = args.data_path
    model_path = args.model_path
    out_path = args.out_path
    batch_size = args.batch_size
    recursive = args.recursive
    save_images = args.images
    cuda = args.cuda

    if not os.path.isdir(data_path):
        print(f"{data_path} is not a valid directory.")
        sys.exit()

    if not os.path.isdir(model_path):
        print(f"{model_path} is not a valid directory.")
        sys.exit()
    
    if batch_size <= 0:
        print(f"Batch size must be a postive integer but was {batch_size}.")
        sys.exit()

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    else:
        for img in tqdm(os.listdir(out_path)):
            os.remove(f"{out_path}/{img}")

    # load model 
    model = ADIFI(load_path=model_path, epsilon=args.epsilon)
    device = "cpu"
    if cuda:
        device = "cuda:0"
        model.to("cuda:0") 

    # colors=["green" if l==1 else "red" for l in y["labels"]]
    colors = [
            (0, 255, 0),
            (255, 0, 0),
        ]

    image_paths = get_image_paths(data_path, recursive)
    N_batchs = int(len(image_paths) // batch_size)
    toPIL = ToPILImage() # used if images flag was set.

    results_df = pd.DataFrame(columns=["Image Path", "label", "score", "x1", "y2", "x2", "y2"])

    for i in tqdm(range(N_batchs+1)):
        if (i+1)*batch_size < len(image_paths):
            image_p_batch = image_paths[i*batch_size:(i+1)*batch_size]
        else:
            image_p_batch = image_paths[i*batch_size:]

        X_batch = model.preprocess(image_p_batch)
        y_batch = model.detect(X_batch)
        for image_path, X, y in zip(image_p_batch, X_batch, y_batch):
            if save_images:
                X = (X * 255).type(torch.uint8)
                y["boxes"] = y["boxes"].type(torch.int16)

                color = [colors[l] for l in y["labels"]]
                # Real is 0, Fake is 1
                label = [f"Real: {score:.4f}" if l==0 else f"Fake: {score:.4f}" for l, score in zip(y["labels"], -torch.log(y["scores"]))]

                X_boxed = draw_bounding_boxes(X, y["boxes"], labels=label, colors=color, width=2)
                toPIL(X_boxed).save(os.path.join(out_path, os.path.basename(image_path)))
            
            for bbox, label, score in zip(y["boxes"], y["labels"], y["scores"]):
                results_df.loc[len(results_df.index)] = [image_path, str(label), float(score), int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    results_df.to_csv(os.path.join(out_path, "results.csv"), index=False)

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)