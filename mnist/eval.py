import numpy as np
import torch
import torch.nn as nn 

@torch.no_grad()
def eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0
    all_preds, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs # multiply mean loss by batch size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
        
    return total_loss/n, correct/n, np.concatenate(all_preds), np.concatenate(all_labels)
        
    