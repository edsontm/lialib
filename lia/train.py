import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics as metrics
import lia.hypergrad.baydin as baydin
__all__ = ['train','predict']

class Trainer():
    def __init__(self,model):
        self.model = model
        

def train(model,dl_train,dl_valid,device,patience_time=10,max_epoch=100,recover_checkpoint=None):
    opt = baydin.AdamHD(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    stop = False
    epoch = 0
    lowest_loss_eval = 10000
    last_best_result = 0
    loss_train = []
    loss_eval = []
    eval_accuracy = 0
    if recover_checkpoint is not None:
        try:
            actual_state = torch.load(recover_checkpoint)
        except Exception as e:
            print('Recover model could not be loaded!'+e)
        else:
            opt.load_state_dict(actual_state['opt'])
            model.load_state_dict(actual_state['model'])
            epoch         = actual_state['epoch']
            loss_train    = actual_state['loss_train']
            loss_eval     = actual_state['loss_eval']
            eval_accuracy = actual_state['eval_accuracy']
            print(f'model recovered: epoch {epoch} loss_train {loss_train[-1]} loss_eval {loss_eval[-1]} eval_accuracy {eval_accuracy}')
        
    model.to(device)
    while (not stop):
        model.train()
        lloss = []
        loop = tqdm(dl_train)
        for x,y in loop:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            closs = criterion(pred,y)
            closs.backward()
            opt.step()
            opt.zero_grad()
            lloss.append(closs.item())
        loss_train.append(np.mean(lloss))
        lloss = []
        model.eval()
        lres = []
        ytrue = []
        with torch.no_grad():
            for data,y in dl_valid:
                data = data.to(device)

                pred = model(data)
                closs = criterion(pred.cpu(),y)
                lloss.append(closs.item())
                res  = pred.argmax(dim=1).cpu().tolist()
                lres += res
                ytrue += y
        avg_loss_eval = np.mean(lloss)
        loss_eval.append(avg_loss_eval)
        eval_accuracy = metrics.accuracy_score(ytrue,lres)
        if avg_loss_eval < lowest_loss_eval:
            lowest_loss_eval = avg_loss_eval 
            last_best_result = 0
            print("Best model found! saving...")
            actual_state = {'opt':opt.state_dict(),'model':model.state_dict(),'epoch':epoch,'loss_train':loss_train,'loss_eval':loss_eval,'eval_accuracy':eval_accuracy}
            torch.save(actual_state,'best_model.pth')
        last_best_result += 1
        if last_best_result > patience_time:
            stop = True
        if epoch >= max_epoch:
            stop =  True
        print(f"epoch {epoch}  loss_train {loss_train[-1]:.4f} loss_eval {avg_loss_eval:.4f} eval_acc {eval_accuracy:.4f} last_best {last_best_result} lr {opt.param_groups[0]['lr']}")
        epoch+=1


