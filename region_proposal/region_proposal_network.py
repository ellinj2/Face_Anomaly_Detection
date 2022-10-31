from torch import nn
from functools import partial

from torchvision.ops import misc, feature_pyramid_network
from torchvision.models.resnet import ResNet50_Weights, resnet50

from torchvision.models.detection import backbone_utils, RetinaNet, FCOS
from torchvision.models.detection.retinanet import RetinaNetHead, _default_anchorgen


class RegionProposalNetowrk():
    """
    Class to interact with the region proposal network.
    """

    def __init__(self, load_path=None):
        pass

    def __retinanet(self, trainable_backbone_layers, pretrained_backbone=True, progress=False, **kwargs):
        """
        Builds RetinaNet model.
        Code inspired by:
            https://github.com/pytorch/vision/blob/ce257ef78b9da0430a47d387b8e6b175ebaf94ce/torchvision/models/detection/retinanet.py#L826-L895
        """
        backbone_weights = None
        if pretrained_backbone:
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2

        trainable_backbone_layers = backbone_utils._validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)
        
        backbone = resnet50(weights=backbone_weights, progress=progress)

        backbone = backbone_utils._resnet_fpn_extractor(
            backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=feature_pyramid_network.LastLevelP6P7(2048, 256))

        anchor_generator = _default_anchorgen()

        head = RetinaNetHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes=2,
            norm_layer=partial(nn.GroupNorm, 32),
        )

        head.regression_head._loss_type = "giou"
        return RetinaNet(backbone, 2, anchor_generator=anchor_generator, head=head, **kwargs)


    def __fcos(self, trainable_backbone_layers, pretrained_backbone=True, progress=False, **kwargs):
        """
        Builds FCOS model.
        Code inspired by:
            https://github.com/pytorch/vision/blob/ce257ef78b9da0430a47d387b8e6b175ebaf94ce/torchvision/models/detection/fcos.py#L676-L769
        """
        backbone_weights = None
        if pretrained_backbone:
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2
        
        trainable_backbone_layers = backbone_utils._validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)
        norm_layer = misc.FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d

        backbone = resnet50(weights=backbone_weights, progress=progress, norm_layer=norm_layer)
        backbone = backbone_utils._resnet_fpn_extractor(
            backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=feature_pyramid_network.LastLevelP6P7(256, 256)
        )

        return FCOS(backbone, num_classes=2, **kwargs)


    def parameters(self):
        """
        Returns the model parameters.
        """
        pass

    def to(self, device):
        """
        Loads and performs computations on the model and input data to specified device.
        """
        pass

    def save(self, save_path, save_name):
        """
        Save the model to specified path.
        """
        pass

    def load(self, load_path):
        """
        Load the model from spesified path.
        """
        pass

    def fit(self, epochs, datasets, batch_size, optimizer, save_path, checkpoints=0, progress=False):
        """
        Fits the model to the training dataset and evaluates on the validation dataset
        """
        pass

    def propose(self, X):
        """
        Proposes regions on the input data. 
        """
        pass

    def evaluate(self, datasets, batch_size=1, progress=False):
        """
        Evaluates the model on a dataset.
        """
        pass
