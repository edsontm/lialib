
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import sklearn.neighbors as neighbors
import sklearn.metrics as metrics

from tqdm import tqdm
import numpy as np

import timm
from resnetpass import resnet18_cbam
from pytorch_metric_learning import losses
import wandb

import os

config = {
    "lr":0.0001,
    "momentum":0.9,
    "weight_decay":0.0001,
    "prot_aug":True,
    "subcenter_margin":0.28,
    "subcenters":3,
    "backbone":"rn18",
    "pretrained":False,
    "optimizer":'sgd',
    "checkpoint_file":'checkpoint.pth'
}

wandb.init(
    # set the wandb project where this run will be logged
    project="backbone_training",
    config=config
)

if config['backbone'] == 'efficientnet':
    model = timm.create_model('efficientnet_b3', pretrained=config['pretrained'])
    model.classifier = nn.Linear(model.classifier.in_features, 512)
elif config['backbone'] == 'efficientnetv2':
    model = torchvision.models.efficientnet_v2_s()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 512)
elif config['backbone'] == 'rn18':
    model = resnet18_cbam(pretrained=False)
    model.fc = nn.Linear(512,512)

device = torch.device('cuda:0')
class_order_file = 'class_order.pth'

if os.path.isfile(class_order_file ):
    class_order = torch.load(class_order_file)
else:
    class_order = torch.randperm(100)
    torch.save(class_order,class_order_file)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

ds_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=preprocess)
ds_test  = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)

def instances_from_classes(dataset, class_order):
    subset = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label in class_order:
            subset.append(i)
    return subset

set_train = instances_from_classes(ds_train,class_order[:50])
set_test  = np.array(instances_from_classes(ds_test, class_order[:50]))

sub_sample_train = sorted(np.random.permutation(set_train)[:300])
sub_sample_test  = sorted(np.random.permutation(set_test)[:300])

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(set_train))
dl_test = torch.utils.data.DataLoader(ds_test,  batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(set_test))
dl_sub_train = torch.utils.data.DataLoader(ds_train,  batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(sub_sample_train))
dl_sub_test = torch.utils.data.DataLoader(ds_test,  batch_size=64, sampler=torch.utils.data.SubsetRandomSampler(sub_sample_test))

loss_func = losses.SubCenterArcFaceLoss(num_classes=100,embedding_size=512,margin=config['subcenter_margin'],sub_centers=config['subcenters'])
if config['optimizer'] == 'sgd':
    opt = optim.SGD(model.parameters(),lr=config['lr'],momentum=config['momentum'],weight_decay=config['weight_decay'])
elif config['optimizer'] == 'adam':
    opt = optim.Adam(model.parameters(),lr=config['lr'])

scheduler = optim.lr_scheduler.MultiStepLR(opt,milestones=[45,90],gamma=0.1)

# given a batch of instances create a surrounding sample of instances
def create_surrounding_batch(batch,batch_y, num_surrounding=100, size=0.001):
    surrounding_batch = []
    surrounding_batch_y = []
    for i in range(len(batch)):
        
        instances_sorrounding = batch[i]+(torch.randn(num_surrounding,512)*size).to(batch.device)
        surrounding_batch.append(instances_sorrounding)
        surrounding_batch_y.append(batch_y[i].repeat(num_surrounding))
    return torch.cat(surrounding_batch), torch.cat(surrounding_batch_y)

def get_embeddings(lmodel,dl):
    lmodel.to(device)
    loop = tqdm(dl)
    lpred = []
    ly = []
    with torch.no_grad():
        for x,y in loop:
            x = x.to(device)
            y = y.to(device)
            pred = lmodel(x)
            lpred = lpred+pred.tolist()
            ly    = ly + y.tolist()
    return lpred,ly

def pred_knn_model(model,dl_local_train,dl_local_test):
    train_emb,train_y = get_embeddings(model,dl_local_train)
    test_emb,test_y   = get_embeddings(model,dl_local_test)
    train_emb  = np.array(train_emb)
    train_y    = np.array(train_y)
    test_emb   = np.array(test_emb)
    test_y     = np.array(test_y)
    clf = neighbors.KNeighborsClassifier(n_neighbors=3,weights='distance')
    clf.fit(train_emb,train_y.T)
    pred = clf.predict(test_emb)
    return pred,test_y

model.to(device)
loss_func.to(device)
loss_train = []
for epoch in range(100):
    model.train()
    loop = tqdm(dl_train)
    lloss = []
    for data, labels in loop:
        data = data.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        embeddings = model(data)
        if config['prot_aug']:
            embeddings,labels = create_surrounding_batch(embeddings,labels)
        loss = loss_func(embeddings, labels)
        loss.backward()
        lloss.append(loss.item())
        opt.step()
    scheduler.step()
    if epoch %10 == 0:
        pred,labels = pred_knn_model(model,dl_sub_train,dl_sub_test)
        f1 = metrics.f1_score(labels,pred,average='macro')
        precision = metrics.precision_score(labels,pred,average='macro')
        recall = metrics.recall_score(labels,pred,average='macro')
        accuracy = metrics.accuracy_score(labels,pred)
        wandb.log({'precision':precision,'recall':recall,'f1':f1,'accuracy':accuracy},step=epoch)
        print(metrics.classification_report(labels.T,pred))
    loss_train.append(np.mean(lloss))
    print(f"epoch {epoch} loss {loss_train[-1]}")
    wandb.log({'loss_train':loss_train[-1]},step=epoch)

pred,labels = pred_knn_model(model,dl_train,dl_test)
f1 = metrics.f1_score(labels,pred,average='macro')
precision = metrics.precision_score(labels,pred,average='macro')
recall = metrics.recall_score(labels,pred,average='macro')
accuracy = metrics.accuracy_score(labels,pred)
wandb.log({'precision':precision,'recall':recall,'f1':f1,'accuracy':accuracy},step=epoch)
print(metrics.classification_report(labels.T,pred))

torch.save({'rn18':model.state_dict(),'class_order':class_order,'loss_train':loss_train,'config':config},config['checkpoint_file'])