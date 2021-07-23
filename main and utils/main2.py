''' Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from tqdm import tqdm
from torchsummary import summary
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

import torchvision
class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10/sabeesh3", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label
    
def generate_dataset(train_batch, test_batch):
    transform_train = A.Compose([
                                 A.Cutout(num_holes = 1, p = 0.5, fill_value =(0.4914, 0.4822, 0.4465)),
                                 A.HorizontalFlip(p=0.5),
                                 A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                 ToTensor()
                                 ])

    transform_test = A.Compose([
        A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ToTensor(),
    ])

    train = Cifar10SearchDataset( train=True, download=True, transform=transform_train,)
    test = Cifar10SearchDataset( train=False, download=True, transform=transform_test)
    dataloader_args_train = dict(shuffle=True, batch_size=train_batch, num_workers=4, pin_memory=True)
    dataloader_args_test =dict(shuffle=True, batch_size=test_batch, num_workers=4, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(train, **dataloader_args_train)
    testloader = torch.utils.data.DataLoader(test, **dataloader_args_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    
    return trainloader, testloader


def generate_model(model, input_size = (3,32,32)):
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")
    model_generated = model.to(device)
    print(summary(model_generated, input_size=input_size))
    return model_generated

                
