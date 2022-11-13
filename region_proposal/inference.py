import os
import cv2
from glob import glob
import pandas as pd
from tqdm import tqdm

import torch
from torchvision.utils import draw_bounding_boxes
from region_proposal_network import RegionProposalNetwork


def get_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a region proposal network.")
    parser.add_argument("--model_path", type=str, required=True, help="") 
    parser.add_argument("--data_path", type=str, required=True, help="")
    parser.add_argument("--out_path", type=str, required=True, help="Path to output results too.")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("-r", "--recursive", action="store_true", help="")
    # parser.add_argument("-w", "--worker", action="store_true", help="Set flag if dataloading worker is wanted. Defaults to false.") # will be implmented soon.
    parser.add_argument("-i", "--images", action="store_true", help="Set flag if image with bounding boxes is desired. Worker is recommended for large number of images as script speed decresses substationally.")
    parser.add_argument("-c", "--cuda", action="store_true", help="Set flag if model should be loaded and trained on a GPU. By default the model will run on cpu.")

    return parser

def main(args):
    data_path = args.data_path
    model_path = args.model_path
    out_path = args.out_path
    batch_size = args.batch_size
    recursive = args.recursive
    # worker = args.worker
    save_images = args.images
    cuda = args.cuda

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # load model 
    model = RegionProposalNetwork(load_path=model_path)
    if cuda:
        model.to("cuda:0") 
    

    # get image paths.
    image_extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
    image_paths = []
    if recursive:
        for ie in image_extensions:
            image_paths = image_paths + glob(os.path.join(data_path, "**", ie), recursive=True)
    else:
        for ie in image_extensions:
            image_paths = image_paths + glob(os.path.join(data_path, ie), recursive=True)


    N_batchs = int(len(image_paths) // batch_size)
    rem = len(image_paths) % batch_size

    results_df = pd.DataFrame(columns=["Image Path", "x1", "y2", "x2", "y2"])

    for i in tqdm(range(N_batchs+1)):
        if i != N_batchs or rem == 0:
            image_p_batch = image_paths[i*batch_size:(i+1)*batch_size]
        else:
            image_p_batch = image_paths[i*batch_size:]

        X_batch = model.preprocess(image_p_batch)
        y_batch = model.propose(X_batch)
        if save_images:
            for image_path, X, y in zip(image_p_batch, X_batch, y_batch):
                X = (X * 255).type(torch.uint8)
                y["boxes"] = y["boxes"].type(torch.int16)
                X_boxed = draw_bounding_boxes(X, y["boxes"], colors="green", width=2)
                X_boxed = torch.moveaxis(X_boxed, 0, -1).numpy()
                X_boxed = cv2.cvtColor(X_boxed, cv2.COLOR_RGB2BGR) # RGB to BGR
                cv2.imwrite(os.path.join(out_path, os.path.basename(image_path)), X_boxed)

                for bbox in y["boxes"]:
                    results_df.loc[len(results_df.index)] = [image_path, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] 
        else:
            for image_path, y in zip(image_p_batch, y_batch):
                for bbox in y["boxes"]:
                    results_df.loc[len(results_df.index)] = [image_path, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    results_df.to_csv(os.path.join(out_path, "results.csv"), index=False)

if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)