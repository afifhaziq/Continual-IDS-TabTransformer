import numpy as np
import random
from torch.utils.data import Dataset

class ReplayDataset(Dataset):
    """A dataset to wrap the experience replay buffer."""
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

def build_replay_buffer(experiences, seen_class_ids, memory_percentage, seed, min_samples_per_class=5, use_validation_set=False):
    """
    Builds a replay buffer from a list of experiences.
    
    Args:
        experiences (list): A list of experience objects, up to the current one.
        seen_class_ids (set): A set of class IDs encountered so far.
        memory_percentage (int): The percentage of samples to keep for each class.
        seed (int): The random seed for reproducibility.
        min_samples_per_class (int): A minimum number of samples to keep for each class.
        use_validation_set (bool): If True, samples from the validation set; otherwise, from the training set.
        
    Returns:
        list: A new replay buffer.
    """
    samples_by_class = {c: [] for c in seen_class_ids}
    
    for exp in experiences:
        # Determine the source dataset based on the flag
        source_ds = exp.val_ds if use_validation_set else exp.train_ds
        labels = source_ds.labels.astype(int)
        
        for class_id in exp.class_ids:
            if class_id in seen_class_ids and class_id in labels:
                indices = np.where(labels == class_id)[0]
                for idx in indices:
                    samples_by_class[class_id].append(source_ds[idx])

    new_buffer = []
    for class_id in seen_class_ids:
        class_samples = samples_by_class[class_id]
        if not class_samples:
            continue
        
        #current_memory_percentage = 5 if class_id == 0 else memory_percentage
        current_memory_percentage = memory_percentage

        num_total_samples = len(class_samples)
        percentage_target = int(num_total_samples * (current_memory_percentage / 100.0))
        target_samples = max(min_samples_per_class, percentage_target)
        quota = min(target_samples, num_total_samples)
        
        # Use a local Random instance with the provided seed for deterministic sampling
        rng = random.Random(seed)
        selected_samples = rng.sample(class_samples, quota)
        new_buffer.extend(selected_samples)
        
    return new_buffer

def print_buffer_distribution(buffer, buffer_name):
    """
    Prints the size and class distribution of a replay buffer.
    """
    print(f"\nFinal {buffer_name} replay buffer size: {len(buffer)}")
    if buffer:
        buffer_class_counts = {}
        for s in buffer:
            class_id = int(s[1].item())
            buffer_class_counts[class_id] = buffer_class_counts.get(class_id, 0) + 1
        
        print(f"Final {buffer_name} buffer class distribution:")
        for c in sorted(buffer_class_counts.keys()):
            print(f"Class {c}: {buffer_class_counts[c]} samples")
