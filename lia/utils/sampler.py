import torch
import numpy as np

__all__ = ['find_lr','split_dataset_stratified','split_dataset_perclass','split_ytrue_stratified']
def split_dataset_stratified(ds,split_size=(0.8,0.1,0.1),return_type='dataloader',bs=64):
    ytrue = np.array(ds.targets)
    sets = split_ytrue_stratified(ytrue,split_size)
    dataloaders = []
    samplers = []
    for set_id in range(len(split_size)):
        sampler = torch.utils.data.SubsetRandomSampler(sets[set_id])
        samplers.append(sampler)
        dataloaders.append(torch.utils.data.DataLoader(ds,batch_size=bs,sampler=sampler))
    if return_type == 'dataloader':
        sets = dataloaders
    elif return_type == 'sampler':
        sets = samplers
    return sets


def split_dataset_perclass(ds, sample_size):
    ds.targets = np.array(ds.targets)
    index_pos=np.arange(0,len(ds))
    lselected = []
    for class_id in np.unique(ds.targets):
        class_set = index_pos[ds.targets==class_id]
        selected = np.random.permutation(len(class_set))[:sample_size]
        lselected += class_set[selected].tolist()
    return torch.utils.data.SubsetRandomSampler(lselected)

def split_ytrue_stratified(ytrue,split_size):
    sets = [np.array([])]*len(split_size)
    ytrue     = np.array(ytrue)
    index_pos = np.arange(0,len(ytrue))
    classes,class_count = np.unique(ytrue,return_counts=True)
    for class_id in classes:

        split_size_quantity = (np.array(split_size) * class_count[class_id]).astype(int)
        remainder = class_count[class_id] - split_size_quantity.sum()
        for i in range(remainder):
            split_size_quantity[i]+=1

        class_set = index_pos[ytrue==class_id]
        i = 0
        for set_id,setsize in enumerate(split_size_quantity):
            sets[set_id]=np.concatenate((sets[set_id],class_set[i:i+setsize].astype(int)))
            i+=setsize
        ret = []
        for vsets in sets:
            ret.append(vsets.astype(int).tolist())
    return ret

