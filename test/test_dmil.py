import torch
import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from sklearn import metrics

from lia.dmil import train
from lia.dmil import predict
from lia.utils import split_dataset_perclass
from lia.utils import split_ytrue_stratified
import lia.dmil as dmil


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu') 

 
def test_loss():
     # prepare model
    refmodel = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model = dmil.loss.ArcFace(refmodel.features)

def test_dmil():
    # prepare data
    if torch.cuda.is_available():
        print('GPU Found!. Using CUDA')
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    ds = CIFAR10('~/datasets/',train=True,download=True,transform=transform)
    n_classes = len(ds.class_to_idx)
    ytrue = torch.tensor(ds.targets)
    index_pos  = torch.arange(ytrue.shape[0])
    batch_size = 256
    sampler    = split_dataset_perclass(ds,sample_size=500)
    vsample    = sampler.indices
    sets       = split_ytrue_stratified(ytrue[vsample],(0.8,0.1,0.1))
    itrain     = index_pos[vsample][sets[0]]
    ival       = index_pos[vsample][sets[1]]
    itest      = index_pos[vsample][sets[2]]
    sampler_train = torch.utils.data.SubsetRandomSampler(itrain)
    sampler_val   = torch.utils.data.SubsetRandomSampler(ival)
    sampler_test  = torch.utils.data.SubsetRandomSampler(itest)

    dl_train = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=sampler_train)
    dl_val   = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=sampler_val)
    dl_test  = torch.utils.data.DataLoader(ds,batch_size=batch_size,sampler=sampler_test)


    # prepare model
    refmodel = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model = dmil.ArcFace(refmodel.features)

    #train
    
    train(model,dl_train,dl_val,device)
    ytrue,pred = predict(model,dl_test,device)
    print(metrics.classification_report(ytrue,pred))

