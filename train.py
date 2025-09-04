import torch
from tqdm.auto import tqdm
import time
from datetime import timedelta
#import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from torch import amp




def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epoch, model_path):
    
    best_val_loss = float('inf')
    #wandb.watch(model, log='all', log_graph=True)
    
    model.train()
    
    for epoch in range(num_epoch):
        
        train_loss = 0
        for cat, cont, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch}"):
            cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(cat, cont)
            loss = criterion(predictions, labels)
            loss.backward()
            
            optimizer.step()
            
            
            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, _, _ = evaluate_model(model, val_loader, device, criterion)
        
        scheduler.step(val_loss)

        
        print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    return model

def evaluate_model(model, dataloader, device, criterion):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for cat, cont, labels in dataloader:
            cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
            predictions = model(cat, cont)

            
            loss = criterion(predictions, labels.long())
            total_loss += loss.item()

            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if criterion and len(dataloader) > 0 else 0
    return avg_loss, all_preds, all_labels

def test_and_report(model, test_loader, device, class_names, task_indices=None):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.
    """
    #print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for cat, cont, labels in test_loader:
            cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
            predictions = model(cat, cont)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    #print(f"Final Test Accuracy: {acc*100:.2f}%")
    labels = list(range(len(class_names)))
    #print('--- Classification Report ---')
    #print(classification_report(all_labels, all_preds, labels=labels, target_names=class_names, digits=4))

    
    task_names = [class_names[i] for i in task_indices]
    print("\n--- Per-task classification report (only selected indices) ---\n")
    print(classification_report(
        all_labels,
        all_preds,
        labels=task_indices,
        target_names=task_names,
        digits=4,
        zero_division=0
        ))
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    
    return acc

