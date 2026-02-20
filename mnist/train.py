import numpy as np
import torch
import torch.nn as nn 


def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    n = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs # multiply mean loss by batch size
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        n += x.size(0)
    return total_loss/n, correct/n
        

    
