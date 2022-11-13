import os
import copy
import json
from hashlib import sha256
from glob import glob

import torch
import numpy as np
from tqdm import tqdm
import time

from torch.utils.data import DataLoader

from torch import nn
from functools import partial

import cv2
import torchvision.transforms as transforms

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
        preprocess: Tranforms images to model compatable format.
        evaluate: Evaluates the model on a given input and returns metrics.
        update_nms_thresholds: Update the non-maximum suppression thresholds.
	"""
    def __init__(self, 
                model_type=None, 
                backbone_type=None,
                load_path=None,
                trainable_backbone_layers=None, 
                pretrained_backbone=True,
                progress=False, 
                **kwargs):
        """
        Initializes an instance of RegionProposalNetwork.

        Parameters:
			model_type [str]: String representing the desired object detection model. Not used if load_path is spesified.
                                Avaiable options are 'retinanet' and 'fcos'. (Default: None)
            backbone_type [str]: String representing the desired resnet backbone for detector. Not used if load_path is spesified.
                                    Avaiable options are 'resnet18', 'resnet34', 'resnet50', 'resnet101', and 'resnet152'. (Default: None)
            load_path [str]: Optional path to file containing model weights to load into model. (Default: None)
            trainable_backbone_layers [int]: Intiger between 0 and 5 indicating the number of trainable layers in 
                                                backbone. 5 means all layers are trainable. If backbone is not 
                                                pretrained do not use this argument. If backbone is pretrained 
                                                and argument is not spesified it defaults to 3.
            pretrained_backbone [bool]: If True the backbone is loaded with pretrained weights on imagenet dataset. (Default: True)
            iou_threshold [float]: Value in the range (0,1] which respensted the maximum detection overlap. (Default: 0.5)
            score_threshold [float]: Value in the range (0,1) which respensted the minimum detection score. (Default: 0.05)
            progress [bool]: If True the backbone pretrained weights download progress bar is displayed. (Default: False) 
			kwargs: Dictionary of the arguments to be passed to the detector API.
        """
        if load_path:
            self.load(load_path)
        else:
            self._model = self.__build_model(model_type, backbone_type, trainable_backbone_layers, pretrained_backbone, progress, **kwargs)
            self._model_metadata = {"model": model_type, 
                                    "backbone": backbone_type,
                                    "parameters": kwargs}  

        self.to(torch.device('cpu')) # defaults to CPU.
               
    def __build_model(self, model_type, backbone_type, trainable_backbone_layers=None, pretrained_backbone=True, progress=False, **kwargs):
        """
        Builds spesified model.

        Reference contructor for paramemter spesifications.
        """
        if pretrained_backbone and trainable_backbone_layers is None:
            trainable_backbone_layers = 3

        backbone = self.__resnet_backbone(model_type, 
                                            backbone_type, 
                                            trainable_backbone_layers, 
                                            pretrained_backbone, 
                                            progress)

        if model_type == "retinanet":
            model = self.__retinanet(backbone, **kwargs)
        elif model_type == "fcos":
            model = self.__fcos(backbone, **kwargs)
        else:
            raise ValueError(f"Model type '{model_type}' is not an available model type. Avaiable options are 'retinanet' and 'fcos'.")

        model.float()

        return model

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
            load_path [str]: Path to folder to load model from. Folder must contain both the .json metadata and .pth weight file.
        """
        try:
            metadata_path = glob(os.path.join(load_path, "*.json"))[0]
            with open(metadata_path, "r") as f:
                self._model_metadata = json.load(f)
        except:
            raise Exception(f"Encountered error while extracting model metadata from {load_path}.")

        self._model = self.__build_model(self._model_metadata["model"],
                                         self._model_metadata["backbone"], 
                                         **self._model_metadata["parameters"])
        try:
            weight_path = glob(os.path.join(load_path, "*.pth"))[0]
            model_weights = torch.load(weight_path)
            
            if sha256(str(model_weights).encode()).hexdigest() != self._model_metadata["weight_hash"]:
                raise Exception("Model weights and metadata is not for the same model.")

            self._model.load_state_dict(model_weights)
        except:
            raise Exception(f"Encountered error while loading model weights from {load_path}.")

    def fit(self, epochs, datasets, batch_size, optimizer, save_path, checkpoints=0, progress=True, num_workers=0):
        """
        Fits the model to the training dataset and evaluates on the validation dataset.

        Parameters:
            epochs [int]: Number of training iterations over the data.
            datasets [tuple]: A tuple containing the training dataset as the first element and the validation dataset 
                                as the second element.
            batch_size [int]: Number of images to batch for training and evaluaton.
            optimizer [torch.optim|tuple]: Either an optimizer or a tuple contain a learning rate scheduler as the first element 
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
        
        train_loader = self.build_dataloader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
        valid_loader = self.build_dataloader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

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
                print(f"\tmAP@.75 - {train_hist['train_map_75'][-1]}")
                print("Validation Results:")
                print(f"\tmAP@.50::.05::.95 - {train_hist['valid_map'][-1]}")
                print(f"\tmAP@.50 - {train_hist['valid_map_50'][-1]}")
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
        return y_hats

    def preprocess(self, image_paths):
        """
        Preprocesses a list of images paths to a list of tensors that are compatable with the model.

        Paramemeters:
            image_paths [list]: List of image paths.

        Return:
            [list]: A list of image tensors loaded to the same device as the model.
        """
        images = []

        transform = transforms.ToTensor()
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image)
            image.to(self.device)
            images.append(image)
        
        return images

    def evaluate(self, dataset, batch_size=1, num_workers=0, progress=False):
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
            dataloader = self.build_dataloader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
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

    def update_nms_thresholds(self, iou_threshold, score_threshold):
        """
        ### Add Doc
        """
        self._model.nms_thresh = iou_threshold
        self._model.score_thresh = score_threshold
        
        self._model_metadata["parameters"]["nms_thresh"] = iou_threshold
        self._model_metadata["parameters"]["score_thresh"] = score_threshold

    def build_dataloader(self, dataset, shuffle=False, batch_size=1, num_workers=0):
        """
        ### Add doc
        """
        return DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            num_workers=num_workers,
                            collate_fn=dataset_formatting,
                            pin_memory=True,
                            persistent_workers=num_workers>0)
