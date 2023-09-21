import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from lia.utils import split
from lia.utils import find_lr
from lia import train
from lia import eval
from lia.utils.main import subsample_perclass
import numpy as np


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 


def test_train():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    ds = CIFAR10('/workspace/datasets/',train=True,download=True,transform=transform)
    n_classes = len(ds.class_to_idx)
    batch_size = 256
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,n_classes )
    #sampler = subsample_perclass(ds,100) # creates a sample of 100 images per class
    #index_selected = sampler.indices
    # index,counts = np.unique(ds.targets[index_selected],return_counts=True)
    dl_train,dl_valid,dl_test = split(ds,batch_size)#,sample_list=index_selected)
    lr                        = find_lr(model,dl_train)
    train(model,dl_train,dl_valid,device)
    eval(model,dl_test,device)


    
    




if __name__ == '__main__':
    test_train()

