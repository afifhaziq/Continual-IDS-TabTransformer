# todo: Currently, the benign classes from both dataset is in separate classes. find out whether to keep it like that or just merge with cic's benign
import yaml, numpy as np, torch, copy, math
from ci_builder import build_dataset_incremental_scenario
from tab_transformer_pytorch import TabTransformer 
from train import test_and_report, train_model
import torch.nn as nn
import torch.optim 
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using the device: ", device)

# --- load config and arrays ONCE ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

 

def load_dataset(config, dataset_name):
    cat_idx_file = f"dataset/{dataset_name}/catfeaturelist.npy"
    train_np = np.load(f"dataset/{dataset_name}/train.npy")
    val_np   = np.load(f"dataset/{dataset_name}/val.npy")
    test_np  = np.load(f"dataset/{dataset_name}/test.npy")
    # global classes (sorted).
    all_classes = tuple(config['datasets'][dataset_name]['classes'])
    num_class   = len(all_classes)
    return (cat_idx_file, train_np, val_np, test_np), all_classes, num_class


CIC_data, CIC_classes, CIC_num_class = load_dataset(config, "CICIDS2017")

UNSW_data, UNSW_classes, UNSW_num_class = load_dataset(config, "UNSWNB15")

combined_data = (CIC_data, UNSW_data)


all_classes = CIC_classes + UNSW_classes

scenario = build_dataset_incremental_scenario(
    combined_data,
    all_classes=all_classes,
    num_class_1 = 8,
    num_class_2 = 10,
    class_order=list(range(len(all_classes))) 
)

#sys.exit()
# --- loop over experiences with your normal PyTorch workflow ---

batch_size = config['batch_size']
max_epochs = config['epochs']
patience   = 3

# define a small helper for loaders
def make_loader(ds, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=16,
                      pin_memory=True, persistent_workers=True)

# init model for the first exp
first_exp = scenario[0]
#print(sec_exp.train_ds.vocab_sizes)
#print(first_exp.train_ds.vocab_sizes)
#print(scenario[1].train_ds.vocab_sizes)
#sys.exit()
model = TabTransformer(
    categories = first_exp.train_ds.vocab_sizes,
    num_continuous = first_exp.train_ds.num_continuous_features,
    dim = 32, depth = 6, heads = 8,
    attn_dropout = 0.1, ff_dropout = 0.1,
    dim_out = CIC_num_class,              # Initial head
).to(device)

criterion = nn.CrossEntropyLoss()     

#print(first_exp.train_ds.vocab_sizes)
#print(first_exp.train_ds.num_continuous_features)

def adjust_model(model, num_class_new_total):
    """
    Expands the model's classifier head to support more classes.
    
    Args:
        model: the model with an existing classification head
        num_class_new_total: the new total number of classes after expansion
    """
    old_head = model.mlp.mlp[-1]
    in_features = old_head.in_features
    num_class_old = old_head.out_features
    if num_class_new_total <= num_class_old:
        return model  # nothing to do

    new_head = nn.Linear(in_features, num_class_new_total, bias=True)

    with torch.no_grad():
        # copy old
        new_head.weight[:num_class_old].copy_(old_head.weight)
        new_head.bias[:num_class_old].copy_(old_head.bias)
        # init new rows with kaiming
        nn.init.kaiming_uniform_(new_head.weight[num_class_old:], a=math.sqrt(5))
        fan_in = new_head.weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        new_head.bias[num_class_old:].uniform_(-bound, bound) # Set the bias to values between -bound and +bound

    model.mlp.mlp[-1] = new_head
    return model
    
import sys
#sys.exit()

def from_report(report_dict):
    acc = float(report_dict.get("accuracy", 0.0))
    macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
    return acc, macro_f1


# ---- W&B ----
wandb.init(project="continual-datasetincremental-Naive", config=dict(
    lr=config["learning_rate"],
    batch_size=config["batch_size"],
    epochs=config["epochs"]
), mode ="disabled")

seen_classes = 0
num_exps = len(scenario)      
#print (num_exps)
#sys.exit()
best_acc_so_far = np.full(num_exps, 0, dtype=np.float32)
best_f1_so_far  = np.full(num_exps, 0, dtype=np.float32)
wandb.define_metric("step_exp")
wandb.define_metric("overall/*", step_metric="step_exp")
wandb.define_metric("exp/*",     step_metric="step_exp")
#wandb.define_metric("Training/*",     step_metric="epoch")
#wandb.watch(model, log='all', log_graph=True)
for exp in scenario:

    exp_id = exp.exp_id
    seen_classes += len(exp.class_ids)
    print(f"\n=== Exp {exp.exp_id} | classes: {exp.class_ids} ===")

    #if exp_id == 0:
        #continue
        
    # Expand head if the number of output increases
    if model.mlp.mlp[-1].out_features < seen_classes:
        model = adjust_model(model, seen_classes)
        model.mlp.mlp[-1] = model.mlp.mlp[-1].to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        print(f"Head Expanded to {model.mlp.mlp[-1].out_features}")

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    
    train_loader = make_loader(exp.train_ds, shuffle=True)
    val_loader   = make_loader(exp.val_ds,   shuffle=False)

    #best_state = copy.deepcopy(model.state_dict())

    best_state = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, max_epochs, exp)

    model.load_state_dict(best_state)

    
    # list of dataset with the classes index. seen_pairs is a tuple (list of datasets, list of classes)
    seen_pairs = [(scenario[i].test_ds, scenario[i].class_ids) 
              for i in range(exp.exp_id + 1)]
    #sys.exit()
    
    # ---- test: current exp only ----
    for i, (single_test, class_group) in enumerate(seen_pairs): # Loop for single task testing
        print(f"\n=== Exp {exp.exp_id} | Test on single experience with classes: {class_group} ===")
        test_loader = make_loader(single_test, shuffle=False)
        #test_and_report(model, test_loader, device, all_classes, task_indices=class_group)

        report_single, y_true, y_pred = test_and_report(
        model, test_loader, device, all_classes,
        task_indices=class_group)
        
        acc, mf1 = from_report(report_single)
        
        fgt_acc = max(0.0, best_acc_so_far[i] - acc)
        fgt_f1  = max(0.0, best_f1_so_far[i]  - mf1)
        best_acc_so_far[i] = max(best_acc_so_far[i], acc)
        best_f1_so_far[i]  = max(best_f1_so_far[i],  mf1)
        #class_names=[all_classes[c] for c in class_group]
        #print(class_names)
   
        # log per-task WITHOUT committing the step yet
        wandb.log({
            f"exp/{i}/acc": acc,
            f"exp/{i}/macro_f1": mf1,
            f"exp/{i}/forgetting_acc": fgt_acc,
            f"exp/{i}/forgetting_macro_f1": fgt_f1,
            "step_exp": exp_id
        }, step=exp_id, commit=False)
    
    seen_classes_list = list(range(seen_classes))
    print(f"\n=== Exp {exp.exp_id} | Test on classes seen so far: {seen_classes_list} ===")

    # ---- test: overall learned exp ----
    seen_test_datasets = [ds for ds, _ in seen_pairs] 

    # Concat the dataset and class_names
    concat_test = torch.utils.data.ConcatDataset(seen_test_datasets)
    
    test_loader = make_loader(concat_test, shuffle=False)

    overall_report, y_true_all, y_pred_all = test_and_report(
    model, test_loader, device, all_classes, task_indices=seen_classes_list)
    overall_acc, overall_macro_f1 = from_report(overall_report)

    wandb.log({
        f"overall/acc": overall_acc,
        f"overall/macro_f1": overall_macro_f1,
        f"overall/{exp_id}/conf_mat": wandb.plot.confusion_matrix(
            y_true=y_true_all, preds=y_pred_all,
            class_names=[all_classes[c] for c in seen_classes_list]
        ),
        "step_exp": exp_id
    }, step=exp_id, commit=True)


        #test_and_report(model, test_loader, device, all_classes, task_indices=seen_classes_list)
   

    #if exp.exp_id == 1:
        #sys.exit()
    # ---- test: seen-so-far exp ----
    
    #sys.exit()
    