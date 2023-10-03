import torch

__all__ = ['predict']
def predict(model,dl_test,device):
    ytrue = []
    lres  = []
    with torch.no_grad():
        for data,y in dl_test:
            data = data.to(device)
            pred = model(data)
            res  = pred.argmax(dim=1).cpu().tolist()
            lres += res
            ytrue += y
    return ytrue,lres

