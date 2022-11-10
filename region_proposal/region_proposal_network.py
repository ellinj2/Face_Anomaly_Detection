import json
import os
import copy
from hashlib import sha256

import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from torch import nn
from functools import partial

from torchvision.ops import misc, feature_pyramid_network, box_iou
from torchvision.models import resnet
from torchvision.models.detection import RetinaNet, FCOS, backbone_utils
from torchvision.models.detection.retinanet import RetinaNetHead, _default_anchorgen

from torchmetrics.detection import MeanAveragePrecision


def dataset_formatting(data):
    """
    Formats data from dataset object to model compatiable data structure.
    """
    images, targets = [], []
    
    for d in data:
        images.append(d[0])
        targets.append(d[1])
        
    return images, targets

class RegionProposalNetwork():
    """    
    Class that implments the region proposal network.
	Attributes:
        iou_threshold [float]: Value in the range (0,1] which respensted the maximum detection overlap.
        score_threshold [float]: Value in the range (0,1) which respensted the minimum detection score.
        device [str]: The device that model is store and computations occur.
	Methods:
        parameters: Get model parameters.
        to: Move model and run all future computations to a spesfied device.
        save: Saves model weights to a file in a spesified directroy.
        load: Loads model weights from spesified file. 
        fit: Trains the model on a given input, evaluating after every epoch.
        propose: Runs inference on the model given an input.
        evaluate: Evaluates the model on a given input and returns metrics.
	"""
    def __init__(self, 
                model_type, 
                backbone_type,
                load_path=None,
                trainable_backbone_layers=None, 
                pretrained_backbone=True, 
                iou_threshold=0.5,
                score_threshold=0.5,
                progress=False, 
                **kwargs):
        """
        Initializes an instance of RegionProposalNetwork.

        Parameters:
			model_type [str]: String representing the desired object detection model.
                                Avaiable options are 'retinanet' and 'fcos'.
            backbone_type [str]: String representing the desired resnet backbone for detector.
                                    Avaiable options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.
            load_path [str]: Optional path to file containing model weights to load into model. (Default: None)
            trainable_backbone_layers [int]: Intiger between 0 and 5 indicating the number of trainable layers in 
                                                backbone. 5 means all layers are trainable. If backbone is not 
                                                pretrained do not use this argument. If backbone is pretrained 
                                                and argument is not spesified it defaults to 3.
            pretrained_backbone [bool]: If True the backbone is loaded with pretrained weights on imagenet dataset. (Default: True)
            iou_threshold [float]: Value in the range (0,1] which respensted the maximum detection overlap.
            score_threshold [float]: Value in the range (0,1) which respensted the minimum detection score.
            progress [bool]: If True the backbone pretrained weights download progress bar is displayed. (Default: False) 
			kwargs: Dictionary of the arguments to be passed to the detector API.
        """
        if pretrained_backbone and trainable_backbone_layers is None:
            trainable_backbone_layers = 3

        backbone = self.__resnet_backbone(model_type, 
                                            backbone_type, 
                                            trainable_backbone_layers, 
                                            pretrained_backbone, 
                                            progress)

        if model_type == "retinanet":
            self._model = self.__retinanet(backbone, **kwargs)
        elif model_type == "fcos":
            self._model = self.__fcos(backbone, **kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' is not an available model type. Avaiable options are 'retinanet' and 'fcos'.")

        self._model.float()
        self.to(torch.device('cpu')) # defaults to CPU.

        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        
        self._model_metadata = {"model": model_type, 
                                "backbone": backbone_type,
                                "iou_threshold": iou_threshold,
                                "score_threshold": score_threshold,
                                "parameters": kwargs} 

        if load_path:
            self.load(load_path)

    def __retinanet(self, backbone, **kwargs):
        """
        Builds RetinaNet model.

        Parameters:
			backbone [TBD]: Model to be used as the backbone for the RetinaNet object detector.
            kwargs: Dictionary of arguments to be passed to the RetinaNet detector API.
                    API Docs: https://pytorch.org/vision/0.12/_modules/torchvision/models/detection/retinanet.html

        Returns:
            [TBD]: RetinaNet model with one object detection.
        """
        anchor_generator = _default_anchorgen()

        head = RetinaNetHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes=2,
            norm_layer=partial(nn.GroupNorm, 32),
        )

        head.regression_head._loss_type = "giou"
        return RetinaNet(backbone, 
                        num_classes=2, 
                        anchor_generator=anchor_generator, 
                        head=head, 
                        # score_thresh=0.05,
                        # nms_thresh=1,
                        **kwargs)

    def __fcos(self, backbone, **kwargs):
        """
        Builds FCOS model.

        Parameters:
			backbone [TBD]: Model to be used as the backbone for the FCOS object detector.
            kwargs: Dictionary of arguments to be passed to the FCOS detector API.
                    API Docs: https://pytorch.org/vision/main/_modules/torchvision/models/detection/fcos.html

        Returns:
            [TBD]: FCOS model with one object detection.
        """
        return FCOS(backbone, 
                    num_classes=2, 
                    # score_thresh=0.05,
                    # nms_thresh=1,
                    **kwargs)

    def __resnet_backbone(self, object_detector, resnet_type, trainable_backbone_layers=None, pretrained_backbone=True, progress=False):
        """
        Builds ResNet backbones. 

        Parameters:
			object_detector [str]: String representing the desired object detection model.
                                    Avaiable options are 'retinanet' and 'fcos'.
            resnet_type [str]: String representing the desired resnet backbone for detector.
                                    Avaiable options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.
            trainable_backbone_layers [int]: Intiger between 0 and 5 indicating the number of trainable layers in 
                                                backbone. 5 means all layers are trainable. If backbone is not 
                                                pretrained do not use this argument. If backbone is pretrained 
                                                and argument is not spesified it defaults to 3.
            pretrained_backbone [bool]: If True the backbone is loaded with pretrained weights on imagenet dataset. (Default: True)
            progress [bool]: If True the backbone pretrained weights download progress bar is displayed. (Default: False) 
        
        Code inspired and influenced by:
                https://github.com/pytorch/vision/blob/ce257ef78b9da0430a47d387b8e6b175ebaf94ce/torchvision/models/detection/fcos.py#L676-L769
                https://github.com/pytorch/vision/blob/ce257ef78b9da0430a47d387b8e6b175ebaf94ce/torchvision/models/detection/retinanet.py#L826-L895
        """
        if pretrained_backbone and trainable_backbone_layers is None:
            trainable_backbone_layers = 3

        if resnet_type == "resnet18":
            weights = resnet.ResNet18_Weights.IMAGENET1K_V1
            channel_out = 512
            backbone = resnet.resnet18
        elif resnet_type == "resnet34":
            weights = resnet.ResNet34_Weights.IMAGENET1K_V1
            channel_out = 512
            backbone = resnet.resnet34
        elif resnet_type == "resnet50":
            weights = resnet.ResNet50_Weights.IMAGENET1K_V2
            channel_out = 2048
            backbone = resnet.resnet50
        elif resnet_type == "resnet101":
            weights = resnet.ResNet101_Weights.IMAGENET1K_V2
            channel_out = 2048
            backbone = resnet.resnet101
        elif resnet_type == "resnet152":
            weights = resnet.ResNet152_Weights.IMAGENET1K_V2
            channel_out = 2048
            backbone = resnet.resnet152
        else:
            raise ValueError("The provided resnet type does is not supported. Avaiable options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'.")
        
        backbone_weights = None
        if pretrained_backbone:
            backbone_weights = weights

        trainable_backbone_layers = backbone_utils._validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

        if object_detector == "retinanet":
            backbone = backbone(weights=backbone_weights, progress=progress)
            extra_block = feature_pyramid_network.LastLevelP6P7(channel_out, 256)
        elif object_detector == "fcos":
            norm_layer = misc.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
            backbone = backbone(weights=backbone_weights, progress=progress, norm_layer=norm_layer)
            extra_block = feature_pyramid_network.LastLevelP6P7(256, 256)
        else:
            raise ValueError("The provided object detector type is not supported. Avaiable options are retinanet and fcos.")

        backbone = backbone_utils._resnet_fpn_extractor(
            backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=extra_block)

        return backbone

    def parameters(self):
        """
        Returns the model parameters.

        Returns:
            [TBD]: Object detector parameters.
        """
        return self._model.parameters()

    def to(self, device):
        """
        Loads and performs computations on the model and input data to specified device.

        Parameters:
            device [str]: Name of the device to load model too.
        """
        try:
            self._model.to(device)
            self.device = device
        except Exception as e:
            raise Exception(e)  

    def save(self, save_path, save_name):
        """
        Save the model to specified path.
        
        Parameters:
            save_path [str]: Path to directory to save model weights too.
            save_name [str]: Name of the model folder.

        Returns:
            [str]: Path to the folder the model files were saved too.
        """
        save_dir = os.path.join(save_path, save_name)
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # save model weights.
        model_weights = self._model.state_dict()
        torch.save(model_weights, os.path.join(save_dir, f"{save_name}_weights.pth"))
        
        # save model metadata.
        self._model_metadata["weight_hash"] = sha256(str(model_weights).encode()).hexdigest()
        with open(os.path.join(save_dir, f"{save_name}_metadata.json"), "w") as f:
            json.dump(self._model_metadata, f, indent=1)

        return save_dir

    def load(self, load_path):
        """
        Load the model from spesified path.

        Parameters:
            load_path [str]: Path to file to load model weights from. Note that this instance of RegionProposalNetwork have been 
                                initialized with the same model configuration as the weight file.
        """
        self._model.load_state_dict(torch.load(load_path, map_location=self.device))

    def fit(self, epochs, datasets, batch_size, optimizer, save_path, checkpoints=0, progress=True, num_workers=0):
        """
        Fits the model to the training dataset and evaluates on the validation dataset.

        Parameters:
            epochs [int]: Number of training iterations over the data.
            datasets [tuple]: A tuple containing the training dataset as the first element and the validation dataset 
                                as the second element.
            batch_size [int]: Number of images to batch for training and evaluaton.
            optimizer [TBD|tuple]: Either an optimizer or a tuple contain a learning rate scheduler as the first element 
                                    and an optimizer as the second element.
            save_path [str]: Path to save model checkpoints.
            checkpoints [int]: Integer N reprisenting after every N epochs to create a model checkpoint. If 0, only save the best model. (Default: 0)
            progress [bool]: Report training progress. (Default: True)
            num_workers [int]: Number of workers to use for DataLoaders. (Default: 0)

        Returns:
            [dict]: Dictionary of model training history.
        """
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        train_dataset, valid_dataset = datasets
        

        train_loader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers, ### try to change these settings if dataloading is slow.
                                    collate_fn=dataset_formatting,
                                    pin_memory=True,
                                    persistent_workers=num_workers>0)
        valid_loader = DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=dataset_formatting,
                                    pin_memory=True,
                                    persistent_workers=num_workers>0)

        sched, optim = optimizer if isinstance(optimizer, tuple) else (None, optimizer)

        train_hist = {
            "train_loss": [],
            "train_map": [],
            "train_map_50": [],
            "train_map_75": [],
            "valid_map": [],
            "valid_map_50": [],
            "valid_map_75": []
        }

        best_acc = -np.inf
        best_model_wts = None

        for e in range(epochs):
            self._model.train()

            if progress:
                print(f"Epoch: {e+1}")

                train_e_loader = tqdm(train_loader)
                train_e_loader.set_description(desc=f"Training loss: {np.nan}")
            else:
                train_e_loader = train_loader
            
            loss = 0
            for i, (X, y) in enumerate(train_e_loader):
                X = [x.to(self.device) for x in X]
                y = [{"boxes": t.to(self.device), "labels": torch.ones(len(t), dtype=torch.int64).to(self.device)} for t in y]
                try:
                    loss_dict = self._model(X, y)
                except Exception as e:
                    ys = [_y['boxes'] for _y in y]
                    for i in range(len(y)):
                        print(X[i], ys[i])
                    print(e)
                    exit()

                batch_loss = sum(b_loss for b_loss in loss_dict.values())
                loss += batch_loss.item()

                optim.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if progress:
                    train_e_loader.set_description(desc=f"Training loss: {loss/(i+1):.4f}")

            train_metrics = self.evaluate(train_loader, batch_size, progress=progress, num_workers=num_workers)
            valid_metrics = self.evaluate(valid_loader, batch_size, progress=progress, num_workers=num_workers)
            
            if valid_metrics["map"] > best_acc:
                best_acc = valid_metrics["map"]
                best_model_wts = copy.deepcopy(self._model.state_dict())

            train_hist["train_loss"].append(loss/len(train_e_loader))
            train_hist["train_map"].append(train_metrics["map"])
            train_hist["train_map_50"].append(train_metrics["map_50"])
            train_hist["train_map_75"].append(train_metrics["map_75"])
            
            train_hist["valid_map"].append(valid_metrics["map"])
            train_hist["valid_map_50"].append(valid_metrics["map_50"])
            train_hist["valid_map_75"].append(valid_metrics["map_75"])
            
            if progress:
                print("Training Results:")
                print(f"\tmAP@.50::.05::.95 - {train_hist['train_map'][-1]}")
                print(f"\tmAP@.50 - {train_hist['train_map_50'][-1]}")
                print(f"\tmAP@.75 - {train_hist['valid_map_75'][-1]}")
                print("Validation Results:")
                print(f"\tmAP@.50::.05::.95 - {train_hist['valid_map'][-1]}")
                print(f"\tmAP@.50 - {train_hist['train_map_50'][-1]}")
                print(f"\tmAP@.75 - {train_hist['valid_map_75'][-1]}")

            if checkpoints > 0:
                if e % checkpoints == 0:
                    self.save(save_path, f"checkpoint_{e+1}")

            if sched:
                sched.step()
        
        self._model.load_state_dict(best_model_wts)
        file_path = self.save(save_path, "best_model")

        if progress:
            print(f"Saved best model to '{file_path}' which achived a validation mAP@.5:.05:.95 of {best_acc:.4f}.")

        return train_hist

    def propose(self, X):
        """
        Proposes regions on the input data. 

        Parameters:
            X [TBD]: Image data to propose regions on.

        Returns:
            [list]: List of dictionaries with region proposals for each image in X.
        """
        self._model.eval()
        with torch.no_grad():
            X = [x.to(self.device) for x in X]
            y_hats = self._model(X)
            # print(y_hats)
            # y_hats = self.__nms(y_hats, self.iou_threshold, self.score_threshold)
        return y_hats


    def evaluate(self, dataset, batch_size=1, progress=False, num_workers=0):
        """
        Evaluates the model on a dataset.

        Parameters:
            dataset [torch.utils.data.Dataset|torch.utils.data.DataLoader]: Dataset or dataloader to use for model evaluation.
            batch_size [int]: Number of images to batch for each evaluaton. (Default: 1)
            progress [bool]: Report evaluation progress. (Default: True)
            num_workers [int]: Number of workers to use for DataLoaders. (Default: 0)

        Returns:
            [dict]: Dictionary of evaluation results.
        """
        self._model.eval()

        if isinstance(dataset, torch.utils.data.Dataset):
            dataloader = DataLoader(dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True, 
                                        num_workers=num_workers, ### try to change these settings if dataloading is slow.
                                        collate_fn=dataset_formatting,
                                        pin_memory=True,
                                        persistent_workers=num_workers>0)
        else:
            dataloader = dataset

        if progress:
            dataloader = tqdm(dataloader)

        metrics = MeanAveragePrecision()
        with torch.no_grad():
            for X, y in dataloader:
                X = [x.to(self.device) for x in X]
                y = [{"boxes": t.to(self.device), "labels": torch.ones(len(t), dtype=torch.int64).to(self.device)} for t in y]
                
                y_hats = self.propose(X)
                metrics.update(y_hats, y)

        return {k: v.item() for k, v in metrics.compute().items()}
                
    def __nms(self, y_preds, iou_threshold, score_threshold):
        """
        Runs non-maximum suppression on region proposals.

        Parameters:
            y_preds [list]: List of dictionarys of the form:
                            {
                                "boxes": torch.Tensor([N,4])
                                "labels": torch.Tensor([N])
                                "scores": torch.Tensor([N])
                            }
                            Where N is the number of detections.
            iou_threshold [float]: Value in the range (0,1] which respensted the maximum detection overlap.
            score_threshold [float]: Value in the range (0,1) which respensted the minimum detection score.

        Returns:
            [dict]: Final region proposals with the same for as .
        """
        y_hats = []
        for y_pred in y_preds:
            # Keep regions with score at or above the score threshold.
            keep_idxs = torch.where(y_pred["scores"] >= score_threshold)
            y_pred = {k: v[keep_idxs] for k, v in y_pred.items()}

            # Sort regions in descending order of scores.
            sorted_idxs = torch.argsort(y_pred["scores"], descending=True)
            y_pred = {k: v[sorted_idxs] for k, v in y_pred.items()}

            y_hat = {"boxes": torch.zeros((0,4), dtype=torch.float32, device=self.device), 
                     "labels": torch.zeros(0, dtype=torch.int64, device=self.device), 
                     "scores": torch.zeros(0, dtype=torch.float32, device=self.device)}

            # Suppress non-maximum boxes.
            while len(y_pred["boxes"]) != 0: 
                # Take bounding box with highest score.
                m_bbox = y_pred["boxes"][0].reshape(1,4)
                m_label = y_pred["labels"][0].reshape(1)
                m_score = y_pred["scores"][0].reshape(1)

                # Add highest scoring bounding box to final proposal dictionary.
                y_hat["boxes"] = torch.concat((y_hat["boxes"], m_bbox))
                y_hat["labels"] = torch.concat((y_hat["labels"], m_label))
                y_hat["scores"] = torch.concat((y_hat["scores"], m_score))

                # Remove highest scoring bounding box from prediction dictionary.
                y_pred["boxes"] = y_pred["boxes"][1:,:]
                y_pred["labels"] = y_pred["labels"][1:]
                y_pred["scores"] = y_pred["scores"][1:]

                # Compute iou score between all predictions and the highest scoring bounding box.
                m_bboxs = torch.tile(m_bbox, dims=(len(y_pred["boxes"]),1)).to(self.device)
                bboxs = y_pred["boxes"]
                iou_scores = self.__iou(m_bboxs, bboxs)

                # Keep the predictions that have a iou score below the iou threshold.
                keep_idxs = torch.where(iou_scores < iou_threshold)
                y_pred["boxes"] = y_pred["boxes"][keep_idxs]
                y_pred["labels"] = y_pred["labels"][keep_idxs]
                y_pred["scores"] = y_pred["scores"][keep_idxs]

            y_hats.append(y_hat)

        return y_hats

    def __iou(self, boxes1, boxes2):
            """
            Computes the intersection over union score for two sets of boxes.

            Parameters:
                boxes1 [torch.Tensor]: A tensor of shape [N, 4] where the row format is [x1, y1, x2, y2] 
                                        and the components have the condition that 0 <= x1 < x2 and 0 <= y1 < y2.
                boxes1 [torch.Tensor]: A tensor of the same shape and components conditions as boxes1.
            
            Returns:
                [torch.Tensor]: A tensor of size N with the IOU score for each pairwise boxes from set 1 and 2. 
            """
            # Get coordinates of the pairwise bounding box intersection.
            x1_i = torch.max(boxes1[:,0], boxes2[:,0])
            y1_i = torch.max(boxes1[:,1], boxes2[:,1])
            x2_i = torch.min(boxes1[:,2], boxes2[:,2])
            y2_i = torch.min(boxes1[:,3], boxes2[:,3])

            # Compute the area of the intersection.
            w_i = x2_i - x1_i
            h_i = y2_i - y1_i
            i_areas = torch.where(w_i >= 0, w_i, 0) * torch.where(h_i >= 0, h_i, 0)
            
            # Compute the area of the union.
            b1_areas = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
            b2_areas = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
            u_areas = b1_areas + b2_areas - i_areas

            return i_areas/u_areas