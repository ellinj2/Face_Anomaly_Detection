import os
import sys
from glob import glob
import multiprocessing as mp
from tqdm import tqdm

import torch
import pandas as pd
from torchvision.ops import batched_nms, box_iou

from region_proposal_network import RegionProposalNetwork
from face_dataset import WFFaceDataset
from utils import get_image_paths

def correct_incorrect_missing(y_trues, y_preds):
    iou_scores = box_iou(y_trues["boxes"], y_preds["boxes"])
    sum_value = torch.sum(iou_scores >= 0.5, dim=1)

    tp = int(torch.sum(sum_value > 0))
    fp = int(torch.sum(torch.where(sum_value > 1, sum_value-1, 0)))
    fn = int(torch.sum(sum_value == 0))

    return (tp, fp, fn)

def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, required=True, help=".")
    parser.add_argument("--data_path", type=str, required=True, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers to load data. Set to -1 to use all cpu cores. (Default: 0)")
    parser.add_argument("-r", "--recursive", action="store_true", help="")
    parser.add_argument("-c", "--cuda", action="store_true", help="Set flag if model should be loaded and trained on a GPU. By default the model will run on cpu.")

    return parser

def main(args):
    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    recursive = args.recursive
    cuda = args.cuda
    num_workers = args.num_workers

    if num_workers < -1:
        print(f"Number of workers must be postive integer or -1 but was {num_workers}.")

    if num_workers == -1:
        num_workers= mp.cpu_count()

    if not os.path.isdir(model_path):
        print(f"{model_path} is an invalid directory.")
        sys.exit()
    
    if not os.path.isdir(data_path):
        print(f"{data_path} is an invalid directory.")
        sys.exit()

    txt_file = glob(os.path.join(data_path, "*.txt"))
    if len(txt_file) != 1:
        print(f"{data_path} should contain one txt file.")
        sys.exit()
    txt_file = txt_file[0]

    mat_file = glob(os.path.join(data_path, "*.mat"))
    if len(mat_file) != 1:
        print(f"{data_path} should contain one txt file.")
        sys.exit()
    mat_file = mat_file[0]

    if batch_size <= 0:
        print(f"Batch size must be a postive integer but was {batch_size}.")
        sys.exit()

    image_paths = get_image_paths(data_path, recursive)

    if len(image_paths) == 0 and not recursive:
        print(f"{data_path} does not contain any images.")
        sys.exit()
    elif len(image_paths) == 0 and recursive:
        print(f"{data_path} and its subdirectorys does not contain any images.")
        sys.exit()
    
    model = RegionProposalNetwork(load_path=model_path)
    model.update_nms_thresholds(1, 0.01)

    if cuda:
        model.to("cuda:0")

    dataset = WFFaceDataset(txt_file, mat_file, data_path)
    dataloader = model.build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)

    y_batchs_trues, y_batchs_preds = ([], [])
    for X, y in tqdm(dataloader):
        if cuda:
            y = y = [{"boxes": t.to("cuda:0"), "labels": torch.ones(len(t), dtype=torch.int64).to("cuda:0")} for t in y]
        y_batchs_trues.append(y)
        y_batchs_preds.append(model.propose(X))

    result_df = pd.DataFrame(columns=["iou_threshold", "score_threshold", "f1@IoU:0.5", "precision@IoU:0.5", "recall@IoU:0.5", "tp", "fp", "fn"])
    thresholds = [i/10 for i in range(1, 10)]
    for iou_threshold in tqdm(thresholds):
        for score_threshold in tqdm(thresholds, leave=False):
            true_pos, false_pos, false_neg = (0, 0, 0)
            for y_batch_trues, y_batch_preds in zip(y_batchs_trues, y_batchs_preds):
                for y_true, y_pred in zip(y_batch_trues, y_batch_preds):
                    keep_idxs = torch.where(y_pred["scores"] > score_threshold)
                    y_pred = {k: v[keep_idxs] for k, v in y_pred.items()}
                    
                    keep_idxs = batched_nms(y_pred["boxes"], y_pred["scores"], y_pred["labels"], iou_threshold)
                    y_pred = {k: v[keep_idxs] for k, v in y_pred.items()}

                    tp, fp, fn = correct_incorrect_missing(y_true, y_pred)
                    true_pos += tp
                    false_pos += fp
                    false_neg += fn

            precision = true_pos / (true_pos + false_pos) if true_pos + false_pos != 0 else 0
            recall = true_pos / (true_pos + false_neg) if true_pos + false_neg != 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

            result_df.loc[len(result_df.index)] = [iou_threshold, score_threshold, f1_score, precision, recall, true_pos, false_pos, false_neg]
    
    idx_max = result_df["f1@IoU:0.5"].idxmax()
    best_iou_threshold = result_df["iou_threshold"][idx_max]
    best_score_threshold = result_df["score_threshold"][idx_max]
    model.update_nms_thresholds(best_iou_threshold, best_score_threshold)

    model.save(os.path.join(model_path))
    result_df.to_csv(os.path.join(model_path, "threshold_optimization.csv"), index=False)
    print("Done")
            
if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    main(args)