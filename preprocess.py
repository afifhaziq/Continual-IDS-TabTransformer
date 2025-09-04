import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys

class TabularDataset(Dataset):
    def __init__(self, data, categorical_indices_file, task_classes, all_classes, fit=True, scaler=None, max_features=None):
        """
        Args:
            data (numpy.ndarray): Full feature matrix (NumSamples, NumFeatures+1 [last col = label])
            categorical_indices_file (str): Numpy file storing indices of categorical features
            task_classes (list): Class names for this task (order must match the label encoding in `data`)
            all_classes (list): Global combined class list, with 'Benign' always at index 0
        """
        self.features = data[:, :-1]
        self.labels = data[:, -1]

        # Build mapping from task-specific indices -> global indices
        self.label_mapping = {i: all_classes.index(c) for i, c in enumerate(task_classes)}

        # Remap labels to global indices
        self.labels = np.array([self.label_mapping[int(l)] for l in self.labels], dtype=np.int64)
        
        # Load categorical indices
        self.categorical_indices = np.load(categorical_indices_file).tolist()
        
        # Continuous indices = all except categorical
        self.cont_indices = [i for i in range(self.features.shape[1]) if i not in self.categorical_indices]

        # Pad if needed
        if max_features:
            self.features = self.pad_features(max_features)

        # Normalize continuous features only
        self.scaler = scaler
        self.features = self._normalize_continuous(fit=fit)

        # Calculate vocab sizes for categorical columns
        self.vocab_sizes = self._calculate_vocab_sizes()

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        all_features = self.features[idx]
        label = self.labels[idx]

        x_categ = torch.tensor(all_features[self.categorical_indices], dtype=torch.long)
        x_cont = torch.tensor(all_features[self.cont_indices], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return x_categ, x_cont, label

    def _normalize_continuous(self, fit=True):
        if len(self.cont_indices) == 0:
            return self.features  # nothing to normalize

        if fit:
            self.scaler = StandardScaler()
            cont_features = self.scaler.fit_transform(self.features[:, self.cont_indices].astype(float))
        else:
            cont_features = self.scaler.transform(self.features[:, self.cont_indices].astype(float))

        self.features[:, self.cont_indices] = cont_features
        return self.features

    def _calculate_vocab_sizes(self):
        vocab_list = []
        for idx in self.categorical_indices:
            if idx == 0:
                vocab_size = 65536
            else:
                col_data = self.features[:, idx]
                vocab_size = int(len(np.unique(col_data)))
            vocab_list.append(vocab_size)
        return vocab_list

    def pad_features(self, max_features):
        current_dim = self.features.shape[1]

        if current_dim < max_features:
            padding_size = max_features - current_dim
            padding = np.zeros((self.features.shape[0], padding_size), dtype=np.float32)
            new_features = np.concatenate([self.features, padding], axis=1)
        elif current_dim > max_features:
            new_features = self.features[:, :max_features]
        else:
            new_features = self.features

        return new_features
        
    @property
    def num_categorical_features(self):
        return len(self.categorical_indices)
    
    @property
    def num_continuous_features(self):
        return len(self.cont_indices)

    @property
    def total_features(self):
        return len(self.cont_indices) + len(self.categorical_indices)

    def verify(self, batch_size=4):
        """Run diagnostic checks to ensure categorical/continuous features are handled correctly"""
        print("🔍 Running dataset verification...\n")

        # 1. Check vocab sizes
        print("Vocab sizes (categorical features):", self.vocab_sizes)

        # 2. Continuous stats
        cont_data = self.features[:, self.cont_indices].astype(float)
        print("\nContinuous features mean (per column):", np.round(np.mean(cont_data, axis=0), 4))
        print("Continuous features std  (per column):", np.round(np.std(cont_data, axis=0), 4))

        # 3. Single sample check
        sample = self[0]
        assert len(sample) == 3, f"❌ Dataset should return 3 elements (x_categ, x_cont, label), got {len(sample)}"
        x_categ, x_cont, label = sample
        print("\nSingle sample check:")
        print("Categorical:", x_categ, "| dtype:", x_categ.dtype, "| shape:", x_categ.shape)
        print("Continuous:", x_cont, "| dtype:", x_cont.dtype, "| shape:", x_cont.shape)
        print("Label:", label, "| dtype:", label.dtype)

        # ✅ Explicit ordering check
        assert torch.is_floating_point(x_cont), "❌ Second element (continuous) must be float tensor"
        assert x_categ.dtype in (torch.int32, torch.int64), "❌ First element (categorical) must be int tensor"
        assert label.ndim == 0 or label.shape[0] == 1, "❌ Third element (label) must be scalar/1D"
        print("✅ Dataset returns (categorical → continuous → label) in correct order")

        # 4. Batch check
        loader = DataLoader(self, batch_size=batch_size, shuffle=True)
        x_categ, x_cont, labels = next(iter(loader))
        print("\nBatch check:")
        print("Categorical batch:", x_categ.shape, "| dtype:", x_categ.dtype)
        print("Continuous batch:", x_cont.shape, "| dtype:", x_cont.dtype)
        print("Labels batch:", labels.shape, "| dtype:", labels.dtype)

        # 5. Special check for port column (if first cat col is ports)
        if 0 in self.categorical_indices:
            ports = self.features[:, 0].astype(int)
            print("\nPort column check → min:", ports.min(), "max:", ports.max())
            if ports.min() < 0 or ports.max() > 65535:
                print("⚠️ Warning: Ports out of range for embedding lookup!")
            else:
                print("✅ Ports within [0, 65535]")

        print("\n✅ Verification complete.\n")

    


