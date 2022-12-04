import torch
import torchvision
from haroun import Data, Model, ConvPool
from haroun.augmentation import augmentation
from haroun.losses import rmse

class Network(torch.nn.Module):
    """
    Network architecture inspired by https://www.kaggle.com/code/hraouak/anti-spoofing-acc-99/notebook
    """
    def __init__(self, image_size=(1,3,64,64)):
        super(Network, self).__init__()
        x = torch.rand(image_size)
        self.input_norm = torch.nn.BatchNorm2d(3, affine=False)
        x = self.input_norm(x)
        pre = x.shape
        self.layer1 = ConvPool(in_features=3, out_features=8)
        x = self.layer1(x)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        self.layer2 = ConvPool(in_features=8, out_features=16)
        x = self.layer2(x)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        self.layer3 = ConvPool(in_features=16, out_features=32)
        x = self.layer3(x)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        self.layer4 = ConvPool(in_features=32, out_features=64)
        x = self.layer4(x)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        self.layer5 = ConvPool(in_features=64, out_features=128)
        x = self.layer5(x)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        self.layer6 = ConvPool(in_features=128, out_features=256)
        x = self.layer6(x)
        x = x.reshape(x.size(0), -1)
        print(f"CONV\t{pre}->\t{x.shape}")
        pre = x.shape
        
        self.net = torch.nn.Sequential(self.layer1, self.layer2, self.layer3, 
                                       self.layer4, self.layer5, self.layer6)
        

        self.fc1 = torch.nn.Linear(in_features=256, out_features=128)
        x = self.fc1(x)
        print(f"LIN\t{pre}->\t{x.shape}")
        pre = x.shape
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=32)
        x = self.fc2(x)
        print(f"LIN\t{pre}->\t{x.shape}")
        pre = x.shape
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.fc3 = torch.nn.Linear(in_features=32, out_features=8)
        x = self.fc3(x)
        print(f"LIN\t{pre}->\t{x.shape}")
        pre = x.shape
        self.bn3 = torch.nn.BatchNorm1d(8)
        self.fc4 = torch.nn.Linear(in_features=8, out_features=2)
        x = self.fc4(x)
        print(f"LIN\t{pre}->\t{x.shape}")

        self.lin = torch.nn.Sequential(self.fc1, self.bn1, self.fc2, self.bn2,
                                       self.fc3, self.bn3, self.fc4)  


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.input_norm(X)
        X = self.net(X)
        X = X.reshape(X.size(0), -1)
        X = self.lin(X)
        X = torch.nn.functional.elu(X, alpha=1.0, inplace=False)
        return X

class AnomalyClassifier:
    def __init__(self, load_path):
        self.image_size = (None, 3, 64, 64)
        self.transform = torchvision.transforms.Resize(self.image_size[2:])
        self.net = Network()
        self.net.load_state_dict(torch.load(load_path))
        self.device = "cpu"

    def to(self, device):
        self.net.to(device)
        self.device = device

    def preprocess(self, X):
        return self.transform(X.to(self.device))

    def classify(self, X):
        self.net.eval()
        X = self.net(X)
        return torch.nn.Sigmoid()(X)
        # return (((X[:, 1] - X[:, 0]) + 1) / 2).detach()