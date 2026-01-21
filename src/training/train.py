import torch
from tqdm.auto import tqdm
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from torch import amp
import copy



def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epoch, exp, oracle=False, dataset_name='CICIDS2017'):
    
    best_val = float('inf')
    no_improve = 0
    patience = 10
    if not oracle:
        model_path = f"model/CL_{dataset_name}_Exp_{exp.exp_id}.pth"
        wandb.define_metric(step_metric = "Epoch", name = f"Training/exp_{exp.exp_id}/train_loss")
        wandb.define_metric(step_metric = "Epoch", name = f"Training/exp_{exp.exp_id}/val_loss")
        wandb.define_metric(step_metric = "Epoch", name = f"Training/exp_{exp.exp_id}/lr")
    else:
        model_path = f"model/CL_{dataset_name}_oracle_{exp.exp_id}.pth"
        wandb.define_metric(step_metric = "Epoch", name = f"Training/oracle_{exp.exp_id}/train_loss")
        wandb.define_metric(step_metric = "Epoch", name = f"Training/oracle_{exp.exp_id}/val_loss")
        wandb.define_metric(step_metric = "Epoch", name = f"Training/oracle_{exp.exp_id}/lr")
    
    #wandb.watch(model, log='all', log_graph=True)
    for epoch in range(1, num_epoch + 1):
        model.train()
        tot, n = 0.0, 0
        for (x_categ, x_cont), y in tqdm(train_loader, desc=f"Exp {exp.exp_id} | Epoch {epoch}/{num_epoch}"):
            
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_categ, x_cont)     
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            bs = y.size(0); tot += loss.item() * bs; n += bs # Find average loss per batch
        train_loss = tot / max(1, n)

        val_loss = evaluate_model(model, val_loader, device, criterion)
        #scheduler.step() # CosineAnnealingLR
        scheduler.step(val_loss) # ReduceLROnPlateau
        print(f"[Exp {exp.exp_id}] Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if not oracle:
            wandb.log({
                f"Training/exp_{exp.exp_id}/train_loss": train_loss,
                f"Training/exp_{exp.exp_id}/val_loss": val_loss,
                f"Training/exp_{exp.exp_id}/lr": optimizer.param_groups[0]['lr'],
                "Epoch": epoch
            }, commit=True)
        else:
            wandb.log({
                f"Training/oracle_{exp.exp_id}/train_loss": train_loss,
                f"Training/oracle_{exp.exp_id}/val_loss": val_loss,
                f"Training/oracle_{exp.exp_id}/lr": optimizer.param_groups[0]['lr'],
                "Epoch": epoch
            }, commit=True)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[EarlyStop Exp {exp.exp_id}] best_val={best_val:.6f}")
                break

    return best_state

def evaluate_model(model, val_loader, device, criterion):
    """
    Evaluates the model on a given dataloader.
    Can compute loss and/or return predictions and labels.
    """

    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for (x_categ, x_cont), y in val_loader:
            x_categ, x_cont, y = x_categ.to(device), x_cont.to(device), y.to(device)
            logits = model(x_categ, x_cont)
            loss = criterion(logits, y)
            bs = y.size(0); tot += loss.item() * bs; n += bs
        val_loss = tot / max(1, n)

    return val_loss

def test_and_report(model, test_loader, device, class_names, task_indices=None):
    """
    Evaluates a model on the test set and prints a classification report
    and confusion matrix.
    """
    #print("\n--- Starting Final Test ---")
    model.eval()

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for (cat, cont), labels in test_loader:
            cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
            predictions = model(cat, cont)
            
            preds = torch.argmax(predictions, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            #print(preds)
    acc = accuracy_score(all_labels, all_preds)
 
    
  
    task_names = [class_names[i] for i in task_indices]

    #sys.exit()
    print("\n--- Per-task classification report (only selected indices) ---\n")
    
    # Get classification report with labels parameter to match target_names
    report_str = classification_report(
        all_labels,
        all_preds,
        labels=task_indices,
        target_names=task_names,
        digits=4,
        zero_division=0
    )
    
    # Replace "micro avg" with "accuracy" for consistency
    if "micro avg" in report_str:
        report_str = report_str.replace("micro avg", "accuracy")
    
    print(report_str)
    #print(res)

    res = classification_report(
        all_labels,
        all_preds,
        labels=task_indices,
        target_names=task_names,
        digits=4,
        zero_division=0,
        output_dict=True
        )
    
    # Fix the dictionary: replace "micro avg" key with "accuracy" key
    if "micro avg" in res:
        # Extract f1-score from micro avg dictionary
        res["accuracy"] = res["micro avg"]["f1-score"]
        del res["micro avg"]
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(all_labels, all_preds))
    
    return res, all_labels, all_preds

