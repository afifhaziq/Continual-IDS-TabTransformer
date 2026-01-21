import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import random

class AvalancheTabularDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_categ, x_cont, label = self.base[idx]
        return (x_categ, x_cont), label

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
        #print("This is tabular object")
        #if remap:
            #self.targets = self.labels.copy()
            # Build mapping from task-specific indices -> global indices
            #self.label_mapping = {i: all_classes.index(c) for i, c in enumerate(task_classes)}

            # Remap labels to global indices
            #self.targets = np.array([self.label_mapping[int(l)] for l in self.labels], dtype=np.int64)
        
        # Load categorical indices
        self.categorical_indices = np.load(categorical_indices_file).tolist()
        #print(self.categorical_indices)
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
        #print(self.vocab_sizes)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        all_features = self.features[idx]
        label = self.labels[idx]

        x_categ = torch.tensor(all_features[self.categorical_indices], dtype=torch.long)
        x_cont = torch.tensor(all_features[self.cont_indices], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return (x_categ, x_cont), label

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
            #if idx == 0:
                #vocab_size = 65536
            #else:
            col_data = self.features[:, idx]
            vocab_size = int(len(np.unique(col_data)))
            vocab_list.append(vocab_size)
            #print(vocab_list)
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





    


