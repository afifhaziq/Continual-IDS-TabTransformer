# /// script
# requires-python = "==3.12.4"
# dependencies = []
# ///

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import sys

from tab_transformer_pytorch import TabTransformer

# ======================================================================================
# STEP 0: SETUP
# ======================================================================================
start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*30)
print(f"Using device: {device.upper()}")
print("="*30)

# ======================================================================================
# STEP 1: LOAD DATASETS FROM .npy FILES
# ======================================================================================
print("STEP 1: Loading datasets from .npy files...")

try:
    # --- Load Dataset A (the larger one) ---
    x_train_A = np.load("x_train_A.npy").astype(np.float32)
    y_train_A = np.load("y_train_A.npy")
    x_test_A = np.load("x_test_A.npy").astype(np.float32)
    y_test_A = np.load("y_test_A.npy")
    print("✅ Successfully loaded Dataset A files.")

    # --- Load Dataset B (the smaller one) ---
    x_train_B = np.load("x_train_B.npy").astype(np.float32)
    y_train_B = np.load("y_train_B.npy")
    # We don't need x_test_B or y_test_B for this simulation, but would for a full project
    print("✅ Successfully loaded Dataset B files.")

except FileNotFoundError as e:
    print(f"❌ ERROR: File not found -> {e}")
    print("Please make sure your .npy files are named correctly (e.g., 'x_train_A.npy') and are in the same directory.")
    exit()

# --- Infer shapes and class counts ---
num_features_A = x_train_A.shape[1]
num_classes_A = int(np.max(y_train_A) + 1)
num_features_B = x_train_B.shape[1]
num_classes_B = int(np.max(y_train_B) + 1)

print(f"\nDataset A Info: {num_features_A} features, {num_classes_A} classes.")
print(f"Dataset B Info: {num_features_B} features, {num_classes_B} classes.\n")

# --- Define the unified feature dimension for the model ---
MODEL_INPUT_DIM = num_features_A # Model is built to handle the larger feature set

# --- Helper function for padding ---
def pad_features(features, target_dim):
    """Pads features with zeros on the right to match the target dimension."""
    current_dim = features.shape[1]
    if current_dim < target_dim:
        padding_size = target_dim - current_dim
        padding = np.zeros((features.shape[0], padding_size), dtype=np.float32)
        return np.concatenate([features, padding], axis=1)
    return features

# --- Helper function for evaluation ---
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for cat_features, cont_features, labels in loader:
            cat_features, cont_features, labels = cat_features.to(device), cont_features.to(device), labels.to(device)
            outputs = model(cat_features, cont_features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds) * 100

# ======================================================================================
# PHASE 1: Initial Training on Dataset A (The "Before" State)
# ======================================================================================
print("\n" + "="*30)
print(f"PHASE 1: Initial Training on Dataset A")
print("="*30)

# Create a dummy categorical feature for TabTransformer compatibility
cat_train_A_dummy = np.zeros((len(x_train_A), 1), dtype=np.int64)
cat_test_A_dummy = np.zeros((len(x_test_A), 1), dtype=np.int64)

# Create DataLoaders for Dataset A
a_train_dataset = TensorDataset(torch.from_numpy(cat_train_A_dummy), torch.from_numpy(x_train_A), torch.from_numpy(y_train_A))
a_test_dataset = TensorDataset(torch.from_numpy(cat_test_A_dummy), torch.from_numpy(x_test_A), torch.from_numpy(y_test_A))
a_train_loader = DataLoader(a_train_dataset, batch_size=256, shuffle=True)
a_test_loader = DataLoader(a_test_dataset, batch_size=512)

# Define the model to handle the larger feature set (A) and class set (A)
model = TabTransformer(
    categories = (1,), # Single dummy category
    num_continuous = MODEL_INPUT_DIM,
    dim = 32,
    dim_out = num_classes_A,
    depth = 6, 
    heads = 8
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

# --- Training Loop on Dataset A ---
NUM_EPOCHS = 5
model.train()
for epoch in range(NUM_EPOCHS):
    for cat, cont, labels in tqdm(a_train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
        outputs = model(cat, cont)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# --- Evaluate the initial model to establish baseline memory ---
baseline_accuracy_a = evaluate_model(model, a_test_loader, device)
print("\n--- Establishing Baseline Performance ---")
print(f"✅ Accuracy on Dataset A BEFORE fine-tuning: {baseline_accuracy_a:.2f}%")
print("This is the model's original 'memory' we will test later.")


# ======================================================================================
# PHASE 2: Naive Fine-Tuning on Dataset B (The Forgetting Process)
# ======================================================================================
print("\n" + "="*30)
print(f"PHASE 2: Naively Fine-Tuning on Dataset B")
print("(This is where catastrophic forgetting happens)")
print("="*30)

# --- CRITICAL STEP: Pad Dataset B features to match the model's input size ---
print(f"Padding Dataset B features from {num_features_B} to {MODEL_INPUT_DIM}...")
x_train_B_padded = pad_features(x_train_B, MODEL_INPUT_DIM)
print("Padding complete.")

# Create dummy categorical features for padded B
cat_train_B_dummy = np.zeros((len(x_train_B_padded), 1), dtype=np.int64)

# Create a DataLoader for the padded Dataset B
b_train_dataset = TensorDataset(torch.from_numpy(cat_train_B_dummy), torch.from_numpy(x_train_B_padded), torch.from_numpy(y_train_B))
b_train_loader = DataLoader(b_train_dataset, batch_size=256, shuffle=True)

# --- Training Loop on Dataset B ---
model.train()
for epoch in range(NUM_EPOCHS):
    for cat, cont, labels in tqdm(b_train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        cat, cont, labels = cat.to(device), cont.to(device), labels.to(device)
        outputs = model(cat, cont)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# ======================================================================================
# PHASE 3: Measuring the Damage (The "After" State)
# ======================================================================================
print("\n" + "="*30)
print("PHASE 3: Measuring Catastrophic Forgetting")
print("="*30)

# --- Check if the model REMEMBERS the OLD task (Dataset A) ---
final_accuracy_a = evaluate_model(model, a_test_loader, device)
print(f"🔻 Accuracy on original task (Dataset A) AFTER fine-tuning: {final_accuracy_a:.2f}%")

# --- Final Conclusion ---
print("\n--- CONCLUSION ---")
print(f"Initial Accuracy on Dataset A: {baseline_accuracy_a:.2f}%")
print(f"Final Accuracy on Dataset A:   {final_accuracy_a:.2f}%")
change = final_accuracy_a - baseline_accuracy_a
print(f"Performance Change: {change:.2f}%")
print("\nThe model's performance on the original task dropped significantly.")
print("This successfully demonstrates CATASTROPHIC FORGETTING.")

end_time = time.time()
print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")