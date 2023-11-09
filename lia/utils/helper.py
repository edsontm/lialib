import torch

def load_model(recover_file):
    actual_state = torch.load(recover_file)
    return actual_state['model']