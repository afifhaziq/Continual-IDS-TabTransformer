# /// script
# requires-python = "==3.12.4"
# dependencies = []
# ///

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import time
import yaml
from tab_transformer_pytorch import TabTransformer, FTTransformer
from preprocess import TabularDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
from train import train_model, test_and_report
import sys
import gc
# ======================================================================================
# STEP 0: SETUP
# ======================================================================================
start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*30)
print(f"Using device: {device.upper()}")
print("="*30)

def get_dataset_info(config, dataset_name, task=None):
    """Reads dataset-specific information from the config."""
    try:
        dataset_config = config['datasets'][dataset_name]
        
        if task:
            dataset_config = config['datasets'][dataset_name][task]
                
        classes = tuple(dataset_config['classes'])
        num_class = len(classes)
        
        return classes, num_class
        
    except KeyError:
        print(f"Error: Configuration for dataset '{dataset_name}' not found in config.yaml.")
        exit()

def load_dataset(config, dataset_name, classes, all_classes, task=None):

    if task:
        data_path = f"dataset/{dataset_name}/{task}"
    else:
        data_path = f"dataset/{dataset_name}"

    train = np.load(f"{data_path}/train.npy")
    val   = np.load(f"{data_path}/val.npy")
    test  = np.load(f"{data_path}/test.npy")

    train_dataset = TabularDataset(train, data_path+"/catfeaturelist.npy", classes, all_classes, fit=True)
    val_dataset   = TabularDataset(val, data_path+"/catfeaturelist.npy", classes, all_classes, fit=False,
                                   scaler=train_dataset.scaler)
    test_dataset  = TabularDataset(test, data_path+"/catfeaturelist.npy", classes, all_classes, fit=False,
                                   scaler=train_dataset.scaler)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"])

    return train_dataset, train_loader, val_loader, test_loader

def adjust_model(model, num_class_new_total):
    """
    Expands the model's classifier head to support more classes.
    
    Args:
        model: the model with an existing classification head
        num_class_new_total: the new total number of classes after expansion
    """
    old_head = model.mlp.mlp[-1]       # old Linear layer TabTransformer
    #old_head = model.to_logits[-1]      # old Linear layer FTTransformer
    in_features = old_head.in_features
    num_class_old = old_head.out_features     # num_class from previous task

    # define new head with expanded output
    new_head = torch.nn.Linear(in_features, num_class_new_total)

    with torch.no_grad():
        # copy over old weights
        new_head.weight[:num_class_old].copy_(old_head.weight)
        new_head.bias[:num_class_old].copy_(old_head.bias)

    # replace classifier
    model.mlp.mlp[-1] = new_head
    #model.to_logits[-1] = new_head
    return model

def combine_class(classes_old, classes_new):
    combined_class =classes_old + classes_new
    classes = list(dict.fromkeys(combined_class))  # preserves order, removes duplicate
    return classes
    
# --- Load Base Config ---
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
config['dataset_name_A'] = "CICIDS2017"

config['task'] = "Task_A"

classes, num_class = get_dataset_info(config, config['dataset_name_A'], config['task'] )

config['num_class_A'] = num_class
config['classes_A'] = classes
config['model_path_A'] = f"model/Model_{config['dataset_name_A']}_{config['task']}.pth"


print("STEP 1: Loading datasets from .npy files...")

# --- Load Data ---
train_dataset_A, train_loader_A, val_loader_A, test_loader_A = load_dataset(config, config['dataset_name_A'], config['classes_A'], config['classes_A'], config['task'])


print(f"\nDataset A Info: {train_dataset_A.total_features} features, {config['num_class_A']} classes.")
#print(f"Dataset B Info: {num_features_B} features, {num_class_B} classes.\n")

# ======================================================================================
# PHASE 1: Initial Training on Task A (The "Before" State)
# ======================================================================================
print("\n" + "="*30)
print(f"PHASE 1: Initial Training on Task A")
print("="*30)

#print(train_dataset_A.vocab_sizes)
#print(train_dataset_A.num_continuous_features)
# Define the model to handle the larger feature set (A) and class set (A)
model = TabTransformer(
    categories = train_dataset_A.vocab_sizes, 
    num_continuous = train_dataset_A.num_continuous_features,
    dim = 32,
    dim_out = config['num_class_A'],
    depth = 6, 
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1, 
).to(device)

# Dummy for model summary
x_categ_dummy = torch.zeros(config["batch_size"], train_dataset_A.num_categorical_features, dtype=torch.long).to(device)
x_cont_dummy = torch.zeros(config["batch_size"], train_dataset_A.num_continuous_features, dtype=torch.float).to(device)

summary(model, 
        input_data=(x_categ_dummy, x_cont_dummy), 
        device=device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

# --- Training and testing on Dataset A ---
train_model(model, train_loader_A, val_loader_A, device, criterion, optimizer, scheduler, config['epochs'], config['model_path_A'])

config['idx_classes_A'] = np.unique(train_dataset_A.labels)

acc_A = test_and_report(model, test_loader_A, device, config['classes_A'], config['idx_classes_A'])

# --- Evaluate the initial model to establish baseline memory ---

print("\n--- Establishing Baseline Performance ---")
#print(f"Accuracy on Dataset A BEFORE fine-tuning: {acc_A:.2f}%")
print("This is the model's original 'memory' we will test later.")

#del train_A, test_A, val_A, test_dataset_A, val_dataset_A, train_loader_A,test_loader_A, val_loader_A
#gc.collect()
#sys.exit()

# ======================================================================================
# PHASE 2: Naive Fine-Tuning on Task B (The Forgetting Process)
# ======================================================================================
print("\n" + "="*30)
print(f"PHASE 2: Naively Fine-Tuning on Task B")
print("(This is where catastrophic forgetting happens)")
print("="*30)

# --- Load Data ---

    
config['task'] = "Task_B"

classes, num_class = get_dataset_info(config, config['dataset_name_A'], config['task'] )

config['num_class_B'] = num_class
config['classes_B'] = classes
config['model_path_B'] = f"model/Model_{config['dataset_name_A']}_{config['task']}.pth"

config['classes_AB'] = combine_class(config['classes_A'], config['classes_B'])
config['num_class_AB'] = len(config['classes_AB'])

print("STEP 1: Loading datasets from .npy files...")

# --- Load Data ---
train_dataset_B, train_loader_B, val_loader_B, test_loader_B = load_dataset(config, config['dataset_name_A'], config['classes_B'], config['classes_AB'], config['task'])


model = TabTransformer(
    categories = train_dataset_B.vocab_sizes, 
    num_continuous = train_dataset_B.num_continuous_features,
    dim = 32,
    dim_out = config['num_class_A'], # Set to original first adjust_model() will change the output later
    depth = 6, 
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1, 
).to(device)

# Load Task A weights
state_dict = torch.load('model/Model_CICIDS2017_Task_A.pth')
model.load_state_dict(state_dict)


#print("\n--- Accuracy on Task A using Model A (baseline) ---")
acc_A = test_and_report(model, test_loader_A, device, config['classes_A'], config['idx_classes_A'])

# Adjust head for Task AB
model = adjust_model(model, config['num_class_AB']).to(device)


config['idx_classes_B'] = np.unique(train_dataset_B.labels)
#print("Unique labels in Task B train:", config['idx_classes_B'])

#print("Model output dim:", model.mlp.mlp[-1].out_features)
#sys.exit()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
# --- Training Loop on Dataset B ---
train_model(model, train_loader_B, val_loader_B, device, criterion, optimizer, scheduler, config['epochs'], config['model_path_B'])
#state_dict = torch.load('model/Model_CICIDS2017_Task_B.pth')
#model.load_state_dict(state_dict)

print("\n--- Accuracy on Task B using Model B ---")
acc_B = test_and_report(model, test_loader_B, device, config['classes_AB'], config['idx_classes_B'])
#print(f"Accuracy on Task B after fine-tuning: {acc_B:.2f}%")

print("\n--- Accuracy on Task A using Model B ---")
acc_AB = test_and_report(model, test_loader_A, device, config['classes_AB'], config['idx_classes_A'])
sys.exit()

# ======================================================================================
# PHASE 3: Naive Fine-Tuning on Dataset C (The Forgetting Process)
# ======================================================================================

print("\n" + "="*30)
print(f"PHASE 3: Naively Fine-Tuning on Task C")
print("(This is where catastrophic forgetting happens)")
print("="*30)

# --- Load Data ---

    
config['task'] = "Task_C"

classes, num_class = get_dataset_info(config, config['dataset_name_A'], config['task'] )

config['num_class_C'] = num_class
config['classes_C'] = list(classes)
config['model_path_C'] = f"model/Model_{config['dataset_name_A']}_{config['task']}.pth"



#sys.exit()
config['classes_ABC'] = combine_class(config['classes_AB'], config['classes_C'])
config['num_class_ABC'] = len(config['classes_ABC'])

print("STEP 1: Loading datasets from .npy files...")

# --- Load Data ---
train_dataset_C, train_loader_C, val_loader_C, test_loader_C = load_dataset(config, config['dataset_name_A'], config['classes_C'], config['classes_ABC'], config['task'])


model = TabTransformer(
    categories = train_dataset_C.vocab_sizes, 
    num_continuous = train_dataset_C.num_continuous_features,
    dim = 32,
    dim_out = config['num_class_A'], # Set to original first adjust_model() will change the output later
    depth = 6, 
    heads = 8,
    attn_dropout = 0.1,
    ff_dropout = 0.1, 
).to(device)

# Load Task B weights
#state_dict = torch.load('model/Model_CICIDS2017_Task_B.pth')
#model.load_state_dict(state_dict)
config['idx_classes_C'] = np.unique(train_dataset_C.labels)

# Adjust head for Task AB
model = adjust_model(model, config['num_class_ABC']).to(device)

#print("Unique labels in Task B train:", config['idx_classes_C'])

#print("Model output dim:", model.mlp.mlp[-1].out_features)
#sys.exit()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)

# --- Training Loop on Dataset C ---
train_model(model, train_loader_C, val_loader_C, device, criterion, optimizer, scheduler, config['epochs'], config['model_path_C'])

print("\n--- Accuracy on Task C using Model C ---")
acc_C = test_and_report(model, test_loader_C, device, config['classes_ABC'], config['idx_classes_C'])

print("\n--- Accuracy on Task B using Model C ---")
acc_BC = test_and_report(model, test_loader_B, device, config['classes_ABC'], config['idx_classes_B'])

print("\n--- Accuracy on Task A using Model C ---")
acc_AC = test_and_report(model, test_loader_A, device, config['classes_ABC'], config['idx_classes_A'])

sys.exit()
# ======================================================================================
# PHASE 3: Measuring the Damage (The "After" State)
# ======================================================================================
print("\n" + "="*30)
print("PHASE 3: Measuring Catastrophic Forgetting")
print("="*30)

# --- Check if the model REMEMBERS the OLD task (Dataset A) ---

print(f"🔻 Accuracy on original task A AFTER fine-tuning: {acc_AB:.2f}%")

# --- Final Conclusion ---
print("\n--- CONCLUSION ---")
print(f"Initial Accuracy on Task A: {acc_A:.2f}%")
print(f"Final Accuracy on Task A:   {acc_AB:.2f}%")
change = acc_A - acc_AB
print(f"Performance Change: {change:.2f}%")
print("\nThe model's performance on the original task dropped significantly.")
print("This successfully demonstrates CATASTROPHIC FORGETTING.")

end_time = time.time()
print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")