import os
import copy

import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader

from torch import nn
from functools import partial

from torchvision.ops import misc, feature_pyramid_network
from torchvision.models import resnet
from torchvision.models.detection import RetinaNet, FCOS, backbone_utils
from torchvision.models.detection.retinanet import RetinaNetHead, _default_anchorgen

from torchmetrics.detection import MeanAveragePrecision


class RegionProposalNetwork():
    """    
    Class that implments the region proposal network.
	Attributes:
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
                progress=False, 
                **kwargs):
        """
        Initializes an instance of RegionProposalNetwork.

        Parameters:
			model_type [str]: String representing the desired object detection model.
                                Avaiable options are 'retinanet' and 'fcos'.
            backbone_type [str]: String representing the desired resnet backbone for detector.
                                    Avaiable options are '18', '34', '50', '101', and '152'.
            load_path [str]: Optional path to file containing model weights to load into model. (Default: None)
            trainable_backbone_layers [int]: Intiger between 0 and 5 indicating the number of trainable layers in 
                                                backbone. 5 means all layers are trainable. If backbone is not 
                                                pretrained do not use this argument. If backbone is pretrained 
                                                and argument is not spesified it defaults to 3.
            pretrained_backbone [bool]: If True the backbone is loaded with pretrained weights on imagenet dataset. (Default: True)
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

        self.to(torch.device('cpu')) # defaults to CPU.
        
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
        return RetinaNet(backbone, num_classes=2, anchor_generator=anchor_generator, head=head, **kwargs)

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
        return FCOS(backbone, num_classes=2, **kwargs)

    def __resnet_backbone(object_detector, resnet_type, trainable_backbone_layers=None, pretrained_backbone=True, progress=False):
        """
        Builds ResNet backbones. 

        Parameters:
			object_detector [str]: String representing the desired object detection model.
                                    Avaiable options are 'retinanet' and 'fcos'.
            resnet_type [str]: String representing the desired resnet backbone for detector.
                                    Avaiable options are '18', '34', '50', '101', and '152'.
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

        if resnet_type == "18":
            weights = resnet.ResNet18_Weights.IMAGENET1K_V1
            backbone = resnet.resnet18
        elif resnet_type == "34":
            weights = resnet.ResNet34_Weights.IMAGENET1K_V1
            backbone = resnet.resnet34
        elif resnet_type == "50":
            weights = resnet.ResNet50_Weights.IMAGENET1K_V2
            backbone = resnet.resnet50
        elif resnet_type == "101":
            weights = resnet.ResNet101_Weights.IMAGENET1K_V2
            backbone = resnet.resnet101
        elif resnet_type == "152":
            weights = resnet.ResNet152_Weights.IMAGENET1K_V2
            backbone = resnet.resnet152
        else:
            raise ValueError("The provided resnet type does is not supported. Avaiable options are '19', '34', '50', '101', and '152'.")
        
        backbone_weights = None
        if pretrained_backbone:
            backbone_weights = weights

        trainable_backbone_layers = backbone_utils._validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

        if object_detector == "retinanet":
            backbone = backbone(weights=backbone_weights, progress=progress)
            extra_block = feature_pyramid_network.LastLevelP6P7(2048, 256)
        elif object_detector == "fcos":
            norm_layer = misc.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
            backbone = backbone(weights=backbone_weights, progress=progress, norm_layer=norm_layer)
            extra_block = feature_pyramid_network.LastLevelP6P7(2048, 256)
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
            save_name [str]: Name of the model weights file. Must include a '.pth' file extention.

        Returns:
            [str]: Path the file the model weights were saved too.
        """
        save_file = os.path.join(save_path, save_name)
        torch.save(self._model.state_dict(), save_file)

        return save_file

    def load(self, load_path):
        """
        Load the model from spesified path.

        Parameters:
            load_path [str]: Path to file to load model weights from. Note that this instance of RegionProposalNetwork have been 
                                initialized with the same model configuration as the weight file.
        """
        self._model.load_state_dict(torch.load(load_path, map_location=self.device))

    def fit(self, epochs, datasets, batch_size, optimizer, save_path, checkpoints=0, progress=False):
        """
        Fits the model to the training dataset and evaluates on the validation dataset.

        Parameters:
            epochs [int]: CHANGEME
            datasets [tuple]: CHANGEME
            batch_size [int]: CHANGEME
            optimizer [TBD]: CHANGEME
            save_path [str]: CHANGEME
            checkpoints [int]: CHANGEME
            progress [bool]:  CHANGEME

        Returns:
            [dict]: CHANGEME
        """
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        train_dataset, valid_dataset = datasets
        

        train_loader = DataLoader(train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, ### try to change these settings if dataloading is slow.
                                    collate_fn=__dataset_formatting,
                                    pin_memory=False,
                                    persistent_workers=False)

        if isinstance(optimizer, tuple):
            sched, optim = optimizer
        else:
            sched, optim = (None, optimizer)

        train_hist = {
            "train_loss": [],
            "train_map": [],
            "train_map_50": [],
            "train_map_75": [],
            "valid_loss": [],
            "valid_map": [],
            "valid_map_50": [],
            "valid_map_75": []
        }

        best_acc = 0
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
                y = [{k: v.to(self.device) for k, v in t.items()} for t in y]

                loss_dict = self.model(X, y)

                batch_loss = sum(b_loss for b_loss in loss_dict.values())
                loss += batch_loss.item()

                optim.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if progress:
                    train_e_loader.set_description(desc=f"Training loss: {loss/(i+1):.4f}")

            train_metrics = self.evaluate(train_dataset, batch_size, progress=progress)
            valid_metrics = self.evaluate(valid_dataset, batch_size, progress=progress)
            
            if valid_metrics["map"] > best_acc:
                best_acc = valid_metrics["map"]
                best_model_wts = copy.deepcopy(self._model.state_dict())

            ### add loss
            train_hist["train_map"].append(train_metrics["map"])
            train_hist["train_map_50"].append(train_metrics["map_50"])
            train_hist["train_map_75"].append(train_metrics["map_75"])
            
            ### add loss
            train_hist["valid_map"].append(valid_metrics["map"])
            train_hist["valid_map_50"].append(valid_metrics["map_50"])
            train_hist["valid_map_75"].append(valid_metrics["map_75"])
            

            if progress:
                print("Training Results:")
                print(f"\tmAP@.50::.05::.95 - {train_hist['train_map']}")
                print(f"\tmAP@.50 - {train_hist['train_map_50']}")
                print(f"\tmAP@.75 - {train_hist['valid_map_75']}")
                print("Validation Results:")
                print(f"\tmAP@.50::.05::.95 - {train_hist['valid_map']}")
                print(f"\tmAP@.50 - {train_hist['train_map_50']}")
                print(f"\tmAP@.75 - {train_hist['valid_map_75']}")                


            if checkpoints > 0:
                if e % checkpoints == 0:
                    self.save(save_path, f"checkpoint_{e+1}.pth")

            if sched:
                sched.step()
        
        self._model.load_state_dict(best_model_wts)
        file_path = self.save(save_path, "best_model.pth")

        if progress:
            print(f"Saved best model to '{file_path}' which achived a validation mAP@.5:.05:.95 of {best_acc:.4f}.")

        return train_hist

    def propose(self, X):
        """
        Proposes regions on the input data. 

        Parameters:
            X [TBD]: CHANGEME

        Returns:
            [TBD]: CHANGEME
        """
        self._model.eval()
        with torch.no_grad():
            X = [x.to(self.device) for x in X]
            
            y_hat = self._model(X)
        
        return y_hat


    def evaluate(self, dataset, batch_size=1, progress=False):
        """
        Evaluates the model on a dataset.

        Parameters:
            dataset [TBD]: CHANGEME
            batch_size [int]: CHANGEME
            progress [bool]: CHANGEME

        Returns:
            [dict]: CHANGEME
        """
        self._model.eval()

        dataloader = DataLoader(dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, ### try to change these settings if dataloading is slow.
                                    collate_fn=__dataset_formatting,
                                    pin_memory=False,
                                    persistent_workers=False)
        
        if progress:
            dataloader = tqdm(dataloader)

        metrics = MeanAveragePrecision()
        with torch.no_grad():
            for X, y in dataloader:
                y_hat = self.propose(X)
                metrics.update(y_hat, y)

        return metrics.compute()
                

def __dataset_formatting(data):
    """
    Formats data from dataset object to model compatiable data structure.
    """
    images, targets = [], []
    
    for d in data:
        images.append(d[0])
        targets.append(d[1])
        
    return images, targets
