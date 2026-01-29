
import yaml
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim
import wandb
import sys
import argparse
import random
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer

# Local imports
from src.data.ci_builder import build_class_incremental_scenario
from src.training.train import test_and_report, train_model
from src.strategies.replay import build_replay_buffer, print_buffer_distribution, ReplayDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description='Continual learning strategies')
parser.add_argument('--er', action='store_true', help='Enable Experience Replay strategy')
parser.add_argument('--mem', type=int, default=10, help='Memory percentage for Experience Replay (e.g., 10 for 10%)')
parser.add_argument('--scenario', type=int, choices=[1, 2], default=1, help='Scenario type: 1 = Class Incremental (benign only in first exp), 2 = Class-Instance Incremental (benign split across all exps)')
parser.add_argument('--balanced',type=str,choices=['True', 'False'],default='False', help='Enable balanced sampling for the replay buffer (True/False)')
parser.add_argument('--dataset', type=str, default='CICIDS2017', help='Dataset name')
# Atack on buffer
attack_group = parser.add_mutually_exclusive_group()
attack_group.add_argument('--lf', action='store_true', default=False, help='Enable Label Flipping attack')
attack_group.add_argument('--mp', action='store_true', default=False, help='Enable Model Poisoning attack')
parser.add_argument('--poison_rate', type=int, default=0, help='Percentage of poisoned data to use in the buffer (0-100)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--learning_rate', type=float, help='Override learning rate from config')
parser.add_argument('--oracle', action='store_true', default=False, help='Enable oracle training for intransigence calculation')

args = parser.parse_args()

strategy = 'er' if args.er else 'naive'
memory_percentage = args.mem if args.er else 0
scenario = args.scenario
poisoning_lf = args.lf
poisoning_mp = args.mp
balanced = 'True' if args.balanced else 'False'
dataset_name = args.dataset

seed = args.seed if args.seed else random.randint(1, 1000)


print(f"Running experiment with seed: {seed}")

# Set seed for reproducibility

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"Strategy: {strategy}")
print(f"Memory percentage: {memory_percentage}%")
print(f"Scenario: {scenario}")
print(f"Label flipping: {poisoning_lf}")
print(f"Model poisoning: {poisoning_mp}")
print(f"Scenario description: {'Class Incremental Scenario' if scenario == 1 else 'Class-Instance Incremental Scenario'}")
print(f"Dataset name: {dataset_name}")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
#sys.exit()

# --- load config ---
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

if args.learning_rate:
    config['learning_rate'] = args.learning_rate


cat_idx_file = f"dataset/{dataset_name}/catfeaturelist.npy"

if poisoning_mp:
    iat_idx_file = f"dataset/{dataset_name}/iatfeaturelist.npy"

cat_global = np.load(cat_idx_file).tolist()

train_np = np.load(f"dataset/{dataset_name}/train.npy")
val_np   = np.load(f"dataset/{dataset_name}/val.npy")
test_np  = np.load(f"dataset/{dataset_name}/test.npy")

# global classes (sorted).
all_classes = tuple(config['datasets'][dataset_name]['classes'])
num_class   = len(all_classes)

# --- build CI scenario:  e.g., 5 exps for 10 classes (2 classes per exp) ---
scenario = build_class_incremental_scenario(
    train_np, val_np, test_np,
    all_classes=all_classes,
    categorical_indices_file=cat_idx_file,
    n_experiences=num_class//2,                 
    class_order=list(range(num_class)),
    scenario=scenario
)
#print(num_class//2)

#import sys
#sys.exit()
# --- loop over experiences with your normal PyTorch workflow ---

batch_size = config['batch_size']
max_epochs = config['epochs']
patience   = 3

# define a small helper for loaders
def create_dataloader(dataset, shuffle, replay_buffer=None, exp_id=None):
    """
    Creates a DataLoader, optionally combining the dataset with a replay buffer.
    """
    if strategy == 'er' and exp_id is not None and exp_id > 0 and replay_buffer:
        buffer_type = "Training" if shuffle else "Validating"
        print(f"--- ER Strategy: {buffer_type} with {len(replay_buffer)} samples from replay buffer ---")
        replay_ds = ReplayDataset(replay_buffer)
        dataset = torch.utils.data.ConcatDataset([dataset, replay_ds])

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16,
                      pin_memory=True, persistent_workers=True)



# Load model configuration from config.yaml
model_config = config['model']

# init model for the first exp
first_exp = scenario[0]
model = TabTransformer(
    categories = first_exp.train_ds.vocab_sizes,
    num_continuous = first_exp.train_ds.num_continuous_features,
    dim = model_config["dim"], depth = model_config["depth"], heads = model_config["heads"], # best 10,
    attn_dropout = model_config["attn_dropout"], ff_dropout = model_config["ff_dropout"],
    dim_out = model_config["dim_out"],     # Initial head
    mlp_act = nn.GELU()
).to(device)

criterion = nn.CrossEntropyLoss()     

#print(first_exp.train_ds.vocab_sizes)
#print(first_exp.train_ds.num_continuous_features)
#sys.exit()

def adjust_model(model, num_class_new_total, device):
    """
    Expands the model's classifier head to support more classes.
    
    Args:
        model: the model with an existing classification head
        num_class_new_total: the new total number of classes after expansion
        device: device to move the new head to
    """
    old_head = model.mlp.mlp[-1]
    in_features = old_head.in_features
    num_class_old = old_head.out_features
    if num_class_new_total <= num_class_old:
        return model  # nothing to do

    # Match bias setting to the old head
    has_bias = (old_head.bias is not None)
    print(f"has_bias: {has_bias}")

    #sys.exit()
    new_head = nn.Linear(in_features, num_class_new_total, bias=has_bias)
    '''
    with torch.no_grad():
        # Copy old weights and biases (if they exist)
        new_head.weight[:num_class_old].copy_(old_head.weight)
        if has_bias:
            new_head.bias[:num_class_old].copy_(old_head.bias)
        
        # Initialize new rows with Xavier uniform (better for final layer)
        nn.init.xavier_uniform_(new_head.weight[num_class_old:], gain=1.0)
        
        # Initialize new biases to zero (better for softmax)
        if has_bias:
            new_head.bias[num_class_old:].zero_()'''
    
    # Replace the head and move to device
    model.mlp.mlp[-1] = new_head.to(device)
    return model
    
#sys.exit()

def from_report(report_dict):
    acc = float(report_dict.get("accuracy", 0.0))
    macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0))
    return acc, macro_f1


def calculate_bwt_fwt_metrics(accuracy_matrix):
    """
    Calculate Backward Transfer (BWT) and Forward Transfer (FWT) metrics.
    This function processes the final accuracy matrix at the end of all experiences.
    """
    n_tasks = accuracy_matrix.shape[0]
    bwt_scores = []
    fwt_scores = []

    # BWT: For each experience i > 0, calculate the average performance change on all tasks j < i.
    for i in range(1, n_tasks):
        # Accuracy on past tasks j after training on task i, compared to performance right after learning task j.
        previous_task_bwt = [accuracy_matrix[i, j] - accuracy_matrix[j, j] for j in range(i)]
        avg_bwt = np.mean(previous_task_bwt) if previous_task_bwt else 0
        bwt_scores.append(avg_bwt)

    # FWT: For each experience i > 0, calculate the zero-shot accuracy on task i having been trained up to task i-1.
    for i in range(1, n_tasks):
        fwt = accuracy_matrix[i - 1, i]
        fwt_scores.append(fwt)
    
    overall_bwt = np.mean(bwt_scores) if bwt_scores else 0
    overall_fwt = np.mean(fwt_scores) if fwt_scores else 0
    
    return {
        'bwt_scores': bwt_scores,
        'fwt_scores': fwt_scores,
        'overall_bwt': overall_bwt,
        'overall_fwt': overall_fwt
    }


def compute_joint_oracle_metrics(scenario, all_classes, device, config):
    """
    Oracle per task j: train a fresh model on concat(train) of exps 0..j, 
    validate on concat(val) of exps 0..j, test on concat(test) of exps 0..j.
    """
    import torch
    from torch.utils.data import ConcatDataset

    oracle_acc = np.zeros(len(scenario), dtype=np.float32)
    oracle_f1 = np.zeros(len(scenario), dtype=np.float32)

    # Base feature schema from exp 0 (same across exps)
    base_ds = scenario[0].train_ds
    base_vocab = base_ds.vocab_sizes
    num_cont  = base_ds.num_continuous_features

    for j in range(len(scenario)):
        # Classes seen up to j; head must support global label ids
        seen_class_ids = sorted({c for k in range(j + 1) for c in scenario[k].class_ids})
        dim_out = max(seen_class_ids) + 1

        model_o = TabTransformer(
            categories=base_vocab,
            num_continuous=num_cont,
            dim=64, depth=6, heads=8,
            attn_dropout=0.1, ff_dropout=0.1,
            dim_out=dim_out
        ).to(device)

        criterion_o = nn.CrossEntropyLoss()
        optimizer_o = torch.optim.AdamW(model_o.parameters(), lr=config["learning_rate"], weight_decay=1e-2)
        scheduler_o = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_o, mode='min', factor=0.1, patience=3)

        # Concat train sets for exps 0..j (no data leakage)
        train_ds_list = []
        val_ds_list = []
        for k in range(j + 1):
            train_ds_list.append(scenario[k].train_ds)
            val_ds_list.append(scenario[k].val_ds)
        
        combined_train_ds = ConcatDataset(train_ds_list)
        combined_val_ds = ConcatDataset(val_ds_list)

        train_loader_o = create_dataloader(combined_train_ds, shuffle=True)
        val_loader_o   = create_dataloader(combined_val_ds,   shuffle=False)

        best_state_o = train_model(model_o, train_loader_o, val_loader_o, device, criterion_o, optimizer_o, scheduler_o, config['epochs'], scenario[j], oracle=True, dataset_name=dataset_name)
        model_o.load_state_dict(best_state_o)

        # Evaluate on concatenated test sets from exps 0..j
        test_ds_list = []
        for k in range(j + 1):
            test_ds_list.append(scenario[k].test_ds)
        combined_test_ds = ConcatDataset(test_ds_list)
        
        test_loader_o = create_dataloader(combined_test_ds, shuffle=False)
        report_o, _, _ = test_and_report(model_o, test_loader_o, device, all_classes, task_indices=seen_class_ids)
        acc_o, mf1_o = from_report(report_o)
        oracle_acc[j] = acc_o
        oracle_f1[j] = mf1_o

        wandb.log({
            f"exp/{j}/oracle_acc": acc_o,
            f"exp/{j}/oracle_f1": mf1_o
        }, commit=False)

        # Explicitly delete objects to free up memory and help terminate dataloader workers
        del model_o, criterion_o, optimizer_o, scheduler_o, best_state_o
        del train_loader_o, val_loader_o, test_loader_o
        del combined_train_ds, combined_val_ds, combined_test_ds
        del train_ds_list, val_ds_list, test_ds_list
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #np.save("oracle_acc.npy", oracle_acc)
    return oracle_acc, oracle_f1

if poisoning_lf:
    project_name = f"continual-CI-Scenario_{args.scenario}_lf"
elif poisoning_mp:
    project_name = f"continual-CI-Scenario_{args.scenario}_mp"
    iat_global = np.load(iat_idx_file).tolist()
else:
    project_name = f"continual-CI-Scenario_{args.scenario}"


# ---- W&B ----
wandb.init(project=project_name, config=dict(
    strategy=strategy,
    memory_percentage=memory_percentage,
    scenario=args.scenario,
    group=memory_percentage,
    lr=config["learning_rate"],
    batch_size=config["batch_size"],
    epochs=config["epochs"],
    dataset=dataset_name, **model_config,
    seed=seed
), mode ="online")



seen_classes_set = set()
num_exps = len(scenario)      
#print (num_exps)
#sys.exit()
accuracy_matrix = np.zeros((num_exps, num_exps)) # R[i,j] = acc on task j after training on task i
f1_matrix = np.zeros((num_exps, num_exps))
overall_accuracy_matrix = np.zeros(num_exps)
overall_f1_matrix = np.zeros(num_exps)
best_acc_so_far = np.full(num_exps, 0, dtype=np.float32)
best_f1_so_far  = np.full(num_exps, 0, dtype=np.float32)
# Use default WandB step; avoid custom step metrics to prevent step conflicts
#wandb.define_metric("Training/*",     step_metric="epoch")
#wandb.watch(model, log='all', log_graph=True)

# --- Replay Buffer Components (for ER) ---

if strategy == 'er':
    replay_buffer = []
    val_replay_buffer = []


for exp in scenario:

    exp_id = exp.exp_id
    seen_classes_set.update(exp.class_ids)
    seen_classes = len(seen_classes_set)
    print(f"\n=== Exp {exp.exp_id} | classes: {exp.class_ids} ===")

    # Expand head if the number of output increases
    if model.mlp.mlp[-1].out_features < seen_classes:
        model = adjust_model(model, seen_classes, device)
        #model.mlp.mlp[-1] = model.mlp.mlp[-1].to(device)
        
    #optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], nesterov=True, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    
    # --- Dataloader preparation and concatenate with replay buffer (Strategy-dependent) ---

    if strategy == 'er':    
        train_loader = create_dataloader(exp.train_ds, shuffle=True, replay_buffer=replay_buffer, exp_id=exp_id)
        val_loader = create_dataloader(exp.val_ds, shuffle=False, replay_buffer=val_replay_buffer, exp_id=exp_id)
    else:
        train_loader = create_dataloader(exp.train_ds, shuffle=True, exp_id=exp_id)
        val_loader = create_dataloader(exp.val_ds, shuffle=False, exp_id=exp_id)

    best_state = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, max_epochs, exp, dataset_name=dataset_name)
    #sys.exit()
    model.load_state_dict(best_state)

    # --- Update replay buffer (if using ER) ---
    if strategy == 'er':
        # Percentage-based buffer: rebuild from scratch at each experience
        print(f"\nRebuilding replay buffer with {memory_percentage}% of samples from all seen experiences...")
        
        # 1. Group all available samples by class from all seen experiences
        seen_class_ids = sorted(list(seen_classes_set))
        replay_buffer = build_replay_buffer(scenario[:exp_id+1], seen_class_ids, memory_percentage, seed, balanced=balanced)


        # Apply in-memory label flipping on the built replay buffer
        if poisoning_lf and replay_buffer:
            from src.strategies.attack import label_flip
            pre_size = len(replay_buffer)
            replay_buffer = label_flip(replay_buffer, seed, args.poison_rate)
            print(f"Applied label flipping to {int(pre_size * (args.poison_rate/100.0))} samples (rate={args.poison_rate}%).")

        elif poisoning_mp and replay_buffer:
            from src.strategies.attack import model_poisoning
            pre_size = len(replay_buffer)
            replay_buffer = model_poisoning(
                replay_buffer,
                seed,
                args.poison_rate,
                iat_indices=iat_global,
                cat_indices=cat_global,
            )
            print(f"Applied model poisoning to {int(pre_size * (args.poison_rate/100.0))} samples (rate={args.poison_rate}%).")

        print_buffer_distribution(replay_buffer, "training")

        # 3. Populate validation replay buffer
        print("\nRebuilding validation replay buffer...")
        val_replay_buffer = build_replay_buffer(scenario[:exp_id+1], seen_class_ids, memory_percentage, seed, use_validation_set=True, balanced=balanced)

        print_buffer_distribution(val_replay_buffer, "validation")

    
    # list of dataset with the classes index. seen_pairs is a tuple (list of datasets, list of classes)
    seen_pairs = [(scenario[i].test_ds, scenario[i].class_ids) 
              for i in range(exp.exp_id + 1)]

    # --- Test-time backdoor injection (Model Poisoning) ---
    # Starting from the 2nd experience (exp_id >= 1): choose a random non-benign class from buffer
    # and inject a backdoor pattern (IAT features set to [vmin, vmax]) into a percentage of its samples
    # in the current experience's single test set. Because the combined test concatenates the same
    # dataset instances, the same poisoned samples will appear there too.
    target_label = None
    rng = None
    if poisoning_mp and strategy == 'er' and exp_id >= 1 and 'replay_buffer' in locals() and replay_buffer:
        non_benign_labels = sorted({int(lbl.item()) for (_, lbl) in replay_buffer if int(lbl.item()) != 0})
        if non_benign_labels:
            rng = random.Random(seed + exp_id)
            target_label = rng.choice(non_benign_labels)
            print(f"Backdoor target class (non-benign): {target_label}")
            wandb.log({
                f"exp/{exp_id}/backdoor_target_class": target_label
            }, commit=False)
            
    #sys.exit()
    
    # ---- test: current exp only ----
    for i, (single_test, class_group) in enumerate(seen_pairs): # Loop for single task testing
        print(f"\n=== Exp {exp.exp_id} | Test on single experience with classes: {class_group} ===")
        # Apply backdoor only to current experience's test set, if target label is present
        if (
            poisoning_mp and target_label is not None and (target_label in class_group)
        ):
            labels_np = single_test.labels.astype(int)
            candidate_idxs = np.where(labels_np == target_label)[0].tolist()
            if candidate_idxs:
                k = int(len(candidate_idxs) * (args.poison_rate / 100.0))
                k = max(0, min(len(candidate_idxs), k))
                selected = set(rng.sample(candidate_idxs, k)) if (rng is not None and k > 0) else set()
                #print("selected:", selected)
                if selected:
                    sel_arr = np.fromiter(selected, dtype=int)
                    vmin, vmax = 14.0, 16.0
                    rand_gen = np.random.RandomState(seed + exp_id)
                    backdoor_vals = rand_gen.uniform(vmin, vmax, size=(len(sel_arr), len(iat_global)))
                    single_test.features[sel_arr[:, None], iat_global] = backdoor_vals
                    print(f"Injected backdoor into {len(sel_arr)} samples of class {target_label} in single test")

        test_loader = create_dataloader(single_test, shuffle=False)
        #test_and_report(model, test_loader, device, all_classes, task_indices=class_group)

        report_single, y_true, y_pred = test_and_report(
        model, test_loader, device, all_classes,
        task_indices=class_group)
        
        acc, mf1 = from_report(report_single)
        
        # Store accuracy in matrix: R[current_exp, tested_task] = accuracy
        accuracy_matrix[exp.exp_id, i] = acc
        f1_matrix[exp.exp_id, i] = mf1
        
        fgt_acc = max(0.0, best_acc_so_far[i] - acc)
        fgt_f1  = max(0.0, best_f1_so_far[i]  - mf1)
        best_acc_so_far[i] = max(best_acc_so_far[i], acc)
        best_f1_so_far[i]  = max(best_f1_so_far[i],  mf1)
        #class_names=[all_classes[c] for c in class_group]
        #print(class_names)

        print("per-task acc:", acc)
        # log per-task WITHOUT committing
        wandb.log({
            f"exp/{i}/acc": acc,
            f"exp/{i}/macro_f1": mf1,
            f"exp/{i}/forgetting_acc": fgt_acc,
            f"exp/{i}/forgetting_macro_f1": fgt_f1
        }, commit=False)
    
    seen_classes_list = sorted(list(seen_classes_set))
    print(f"\n=== Exp {exp.exp_id} | Test on classes seen so far: {seen_classes_list} ===")

    # ---- test: overall learned exp ----
    seen_test_datasets = [ds for ds, _ in seen_pairs] 

    # Concat the dataset and class_names
    concat_test = torch.utils.data.ConcatDataset(seen_test_datasets)
    
    test_loader = create_dataloader(concat_test, shuffle=False)
    #print("tHIS IS SEEN CLASS names:", all_classes)
    overall_report, y_true_all, y_pred_all = test_and_report(
    model, test_loader, device, all_classes, task_indices=seen_classes_list)
    overall_acc, overall_macro_f1 = from_report(overall_report)
    overall_accuracy_matrix[exp_id] = overall_acc
    overall_f1_matrix[exp_id] = overall_macro_f1

    # Create a contiguous local mapping (0..N-1) for the confusion matrix
    # This avoids issues with sparse indices or string matching in W&B
    local_map = {global_id: local_idx for local_idx, global_id in enumerate(seen_classes_list)}
    
    # Map predictions to local 0..N-1 indices
    y_true_local = [local_map[int(i)] for i in y_true_all]
    y_pred_local = [local_map[int(i)] for i in y_pred_all]
    
    # Disable wandb display names alphabetically for UNSWNB15
    class_names_display = [all_classes[c] for c in seen_classes_list]
    
    if dataset_name == "UNSWNB15":
        class_names_display = [f"{c:02d}_{all_classes[c]}" for c in seen_classes_list]
    else:
        class_names_display = [all_classes[c] for c in seen_classes_list]

    print("Debug: all_classes[c] for c in seen_classes_list:", class_names_display)
    wandb.log({
        "overall/acc": overall_acc,
        "overall/macro_f1": overall_macro_f1,
        f"overall/{exp_id}/conf_mat": wandb.plot.confusion_matrix(
            y_true=y_true_local, preds=y_pred_local,
            class_names=class_names_display
        )
    }, commit=True)
    
    print(f"Completed experience {exp_id} - Overall accuracy: {overall_acc:.4f}, Macro F1: {overall_macro_f1:.4f}")

print("\nAll experiences completed successfully!")
#sys.exit()
# --- Final Analysis ---
# Calculate and log BWT/FWT metrics at the end of all experiences
bwt_fwt_results_acc = calculate_bwt_fwt_metrics(accuracy_matrix)
bwt_fwt_results_f1 = calculate_bwt_fwt_metrics(f1_matrix)

print("Final Accuracy Matrix (R[i,j] = acc on task j after training on task i):")
print(pd.DataFrame(accuracy_matrix).to_string(float_format="%.4f"))
print("\nFinal F1-Score Matrix (R[i,j] = F1 on task j after training on task i):")
print(pd.DataFrame(f1_matrix).to_string(float_format="%.4f"))

print(f"\nOverall BWT (Accuracy): {bwt_fwt_results_acc['overall_bwt']:.4f}")
print(f"Overall FWT (Accuracy): {bwt_fwt_results_acc['overall_fwt']:.4f} (Note: measures zero-shot acc on next task)")

print(f"\nOverall BWT (F1-Score): {bwt_fwt_results_f1['overall_bwt']:.4f}")
print(f"Overall FWT (F1-Score): {bwt_fwt_results_f1['overall_fwt']:.4f} (Note: measures zero-shot F1 on next task)")

# --- Prepare final logs ---
wandb_log = {
    "overall/bwt_acc": bwt_fwt_results_acc['overall_bwt'],
    "overall/fwt_acc": bwt_fwt_results_acc['overall_fwt'],
    "overall/bwt_f1": bwt_fwt_results_f1['overall_bwt'],
    "overall/fwt_f1": bwt_fwt_results_f1['overall_fwt']
}

# Add per-experience BWT and FWT scores to the log
for i, (bwt_acc, bwt_f1) in enumerate(zip(bwt_fwt_results_acc['bwt_scores'], bwt_fwt_results_f1['bwt_scores'])):
    exp_id_for_bwt = i + 1
    wandb_log[f"exp/{exp_id_for_bwt}/bwt_avg_so_far_acc"] = bwt_acc
    wandb_log[f"exp/{exp_id_for_bwt}/bwt_avg_so_far_f1"] = bwt_f1

for i, (fwt_acc, fwt_f1) in enumerate(zip(bwt_fwt_results_acc['fwt_scores'], bwt_fwt_results_f1['fwt_scores'])):
    exp_id_for_fwt = i + 1
    wandb_log[f"exp/{exp_id_for_fwt}/fwt_zero_shot_acc"] = fwt_acc
    wandb_log[f"exp/{exp_id_for_fwt}/fwt_zero_shot_f1"] = fwt_f1

# --- Oracle Section (Conditional) ---
if args.oracle:
    # Oracle via joint training
    print("\n" + "="*60)
    print("ORACLE TRAINING SECTION - Computing Intransigence Metrics")
    print("="*60)
    print("Training oracle models for each task with joint training")
    oracle_acc, oracle_f1 = compute_joint_oracle_metrics(scenario, all_classes, device, config)
    print("Oracle training completed!")

    # Calculate Intransigence
    intransigence_acc = oracle_acc - overall_accuracy_matrix
    overall_intransigence_acc = np.mean(intransigence_acc) if intransigence_acc.size > 0 else 0.0
    intransigence_f1 = oracle_f1 - f1_matrix
    overall_intransigence_f1 = np.mean(intransigence_f1) if intransigence_f1.size > 0 else 0.0

    # Print intransigence summary
    print(f"\nOverall Intransigence (Accuracy): {overall_intransigence_acc:.4f}")
    print(f"Overall Intransigence (F1-Score): {overall_intransigence_f1:.4f}")

    # Add intransigence to log dictionary
    wandb_log["overall/intransigence_acc"] = overall_intransigence_acc
    wandb_log["overall/intransigence_f1"] = overall_intransigence_f1
    
    for i, (intransigence_acc_val, intransigence_f1_val) in enumerate(zip(intransigence_acc, intransigence_f1)):
        exp_id_for_intransigence = i + 1
        wandb_log[f"exp/{exp_id_for_intransigence}/intransigence_acc"] = intransigence_acc_val
        wandb_log[f"exp/{exp_id_for_intransigence}/intransigence_f1"] = intransigence_f1_val

# --- Final Log ---
# Log all collected metrics in a single commit
print("\n" + "="*50)
print("Logging final metrics to W&B...")
wandb.log(wandb_log, commit=True)
print("Done.")

sys.exit()
