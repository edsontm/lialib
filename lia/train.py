import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics as metrics
__all__ = ['train']

class Trainer():
    def __init__(self,model):
        self.model = model
        

def train(model,dl_train,dl_valid,device):
    epochs = 100
    opt = optim.AdamW(model.parameters(),lr=0.004,betas=(0.899,0.999))
    criterion = nn.CrossEntropyLoss()
    patience_time = 10
    stop = False
    epoch = 0
    lowest_loss_eval = 10000
    last_best_result = 0
    loss_train = []
    loss_eval = []
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
            actual_state = {'optim':opt.state_dict(),'model':model.state_dict(),'epoch':epoch,'loss_train':loss_train,'loss_eval':loss_eval}
            torch.save(actual_state,'best_model.pth')
        last_best_result += 1
        if last_best_result > patience_time:
            stop = True
        print(f"epoch {epoch} loss_train {loss_train[-1]:.4f} loss_eval {avg_loss_eval:.4f} eval_acc {eval_accuracy:.4f} last_best {last_best_result}")
        epoch+=1

