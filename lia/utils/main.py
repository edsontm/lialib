import torch
from sklearn.model_selection import train_test_split
import numpy as np

__all__ = ['find_lr','split','subsample_stratified','subsample_perclass']

def find_lr(model,dl_train):
    pass
def split(ds,bs=64,split_size=(0.8,0.1,0.1),sample_list = None):
    
    if sample_list is None:
        sample_list_target = np.array(ds.targets)
        sample_list = np.arange(len(ds))
    else:
        sample_list_target = np.array(ds.targets[sample_list])
    test_size1 = 1 - split_size[0]
    test_size2 = split_size[0]/(split_size[0]+split_size[1])
    train_idx, temp_idx = train_test_split(sample_list,test_size=test_size1,shuffle=True,stratify=sample_list_target)
    valid_idx, test_idx = train_test_split(temp_idx,test_size=test_size2,shuffle=True,stratify=sample_list_target[temp_idx])
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler  = torch.utils.data.SubsetRandomSampler(test_idx)
    
    dl_train = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=train_sampler)
    dl_valid = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=valid_sampler)
    dl_test  = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=test_sampler)
    return (dl_train,dl_valid,dl_test)
def split(ds,bs=64,split_size=(0.8,0.1,0.1)):
    sample_list_target = np.array(ds.targets)
    sample_list = np.arange(len(ds))
    test_size1 = 1 - split_size[0]
    test_size2 = split_size[0]/(split_size[0]+split_size[1])
    train_idx, temp_idx = train_test_split(sample_list,test_size=test_size1,shuffle=True,stratify=sample_list_target)
    valid_idx, test_idx = train_test_split(temp_idx,test_size=test_size2,shuffle=True,stratify=sample_list_target[temp_idx])
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    test_sampler  = torch.utils.data.SubsetRandomSampler(test_idx)
    
    dl_train = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=train_sampler)
    dl_valid = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=valid_sampler)
    dl_test  = torch.utils.data.DataLoader(ds,batch_size=bs,sampler=test_sampler)
    return (dl_train,dl_valid,dl_test)


def subsample_stratified(ds,test_size):
    ds.targets = np.array(ds.targets) 
    _,test_idx = train_test_split(np.arange(len(ds)),test_size=test_size,shuffle=True,stratify=ds.targets)
    return torch.utils.data.SubsetRandomSampler(test_idx)

def subsample_perclass(ds,sample_size):
    ds.targets = np.array(ds.targets)
    index_pos=np.arange(0,len(ds))
    lselected = []
    for class_id in np.unique(ds.targets):
        class_set = index_pos[ds.targets==class_id]
        selected = np.random.permutation(len(class_set))[:sample_size]
        lselected += class_set[selected].tolist()
    return torch.utils.data.SubsetRandomSampler(lselected)


