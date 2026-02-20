import torch
import torch.nn as nn 

from mnist.utils import set_seed
from mnist.utils import get_device
from mnist.load_data import get_data
from mnist.model import CNN_Classifier 
from mnist.train import train
from mnist.eval import eval

from pathlib import Path
import json

path_model = "./models"
path_history = "output/history"

def trainer(load_model):
    if load_model:
        model = CNN_Classifier(0.2)
        state = torch.load(Path(path_model, "mnist_cnn.pt"))
        model.load_state_dict(state)
        
        with open(Path(path_history, "train_history.json"), "r") as f:
            history = json.load(f)
            
        with open(Path(path_history, "test_data.json"), "r") as f:
            test_data = json.load(f)
        return model, history, test_data
         
    set_seed(1)
    device = get_device()
    print(f"Device: {device}")
    
    # Hyperparams
    batch_size = 32
    epochs = 5
    lr = 1e-3
    val_size = 10_000
    
    train_loader, val_loader, test_loader, _ = get_data("data", batch_size=batch_size, val_size=val_size, seed=1)
    
    model = CNN_Classifier(0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}    
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss:.2f}, Val Loss: {val_loss:.3f}, Train Acc.: {train_acc}, Val Acc: {val_acc}")
        
        Path("output/history").mkdir(parents=True, exist_ok=True)

        
    test_loss, test_acc, preds, labels = eval(model, test_loader, criterion, device)  
    print(f"Test Loss: {test_loss:.2f}, Test Acc.: {test_acc}") 
    
    # Save test data
    test_data = {"test_loss": test_loss, "test_acc": test_acc, "labels": labels.tolist(), "preds": preds.tolist()}
    with open(Path(path_history, "test_data.json"), "w") as f:
        json.dump(test_data, f, indent=2)
    
    

    # Save history
    Path(path_history).mkdir(parents=True, exist_ok=True)
    with open(Path(path_history, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

        
    # Save model
    Path(path_model).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(path_model, "mnist_cnn.pt"))

    
    return model, history, test_data
        
        