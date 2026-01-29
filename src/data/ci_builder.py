from dataclasses import dataclass
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Sequence, Optional
from .preprocess import TabularDataset


@dataclass
class CIExperience:
    exp_id: int
    class_ids: List[int]  # global class ids in this experience
    class_names: List[str]  # names aligned to global all_classes
    train_ds: TabularDataset
    val_ds: TabularDataset
    test_ds: TabularDataset

    def make_loaders(
        self, batch_size: int, num_workers: int = 8, shuffle_train: bool = True
    ):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        return train_loader, val_loader, test_loader


class CIScenario:
    """Indexable CI scenario with parallel train/val/test experiences."""

    def __init__(self, exps: List[CIExperience]):
        self._exps = exps

    def __len__(self):
        return len(self._exps)

    def __getitem__(self, i) -> CIExperience:
        return self._exps[i]

    @property
    def train_stream(self):
        return self._exps

    @property
    def valid_stream(self):
        return self._exps

    @property
    def test_stream(self):
        return self._exps


def _split_classes_into_experiences(
    all_class_ids: Sequence[int], n_experiences: int, scenario: int = 2
) -> List[List[int]]:
    """
    Chunk global class ids into n_experiences based on scenario type.

    Args:
        all_class_ids: All class IDs to distribute
        n_experiences: Number of experiences to create
        scenario: 1 (benign only in first exp) or 2 (benign split across all exps)

    Returns:
        List of class groups for each experience
    """
    benign_class = 0
    other_classes = [c for c in all_class_ids if c != benign_class]

    experiences = []

    if scenario == 1:
        # Scenario 1: Benign class only in first experience
        # Exp 0: [0, 1], Exp 1: [2, 3], Exp 2: [4, 5], Exp 3: [6, 7]
        for exp_id in range(n_experiences):
            if exp_id == 0:
                # First experience: benign + first other class
                exp_classes = [benign_class, other_classes[0]]
            else:
                # Other experiences: 2 other classes each
                start_idx = 1 + (exp_id - 1) * 2
                end_idx = start_idx + 2
                if end_idx <= len(other_classes):
                    exp_classes = other_classes[start_idx:end_idx]
                else:
                    # Take remaining classes if less than 2
                    exp_classes = other_classes[start_idx:]

            experiences.append(exp_classes)

    elif scenario == 2:
        # Scenario 2: Benign class split across all experiences
        # Exp 0: [0, 1], Exp 1: [0, 2, 3], Exp 2: [0, 4, 5], Exp 3: [0, 6, 7]
        for exp_id in range(n_experiences):
            if exp_id == 0:
                # First experience: benign + first other class
                exp_classes = [benign_class, other_classes[0]]
            else:
                # Other experiences: benign + 2 other classes
                start_idx = 1 + (exp_id - 1) * 2
                end_idx = start_idx + 2
                if end_idx <= len(other_classes):
                    exp_classes = [benign_class] + other_classes[start_idx:end_idx]
                else:
                    # Take remaining classes if less than 2
                    exp_classes = [benign_class] + other_classes[start_idx:]

            experiences.append(exp_classes)

    else:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 1 or 2")

    return experiences


def _subset_by_classes(
    data_np: np.ndarray, keep_class_ids: Sequence[int]
) -> np.ndarray:
    """Filter rows by labels ∈ keep_class_ids (labels are last column)."""
    labels = data_np[:, -1].astype(np.int64)
    mask = np.isin(labels, np.array(keep_class_ids, dtype=np.int64))
    return data_np[mask]


def _get_benign_split_indices(
    all_data: np.ndarray, benign_class: int = 0, exp_id: int = 0, n_experiences: int = 4
) -> np.ndarray:
    """
    Get the benign sample indices for a specific experience across all datasets.
    This ensures consistent splitting across train/val/test to prevent data leakage.

    Args:
        all_data: Combined data from train/val/test (used to determine total benign samples)
        benign_class: The benign class ID (default: 0)
        exp_id: Current experience ID
        n_experiences: Total number of experiences

    Returns:
        Array of benign sample indices for this experience
    """
    all_labels = all_data[:, -1].astype(np.int64)
    benign_indices = np.where(all_labels == benign_class)[0]

    if len(benign_indices) == 0:
        return np.array([], dtype=int)

    # Split benign samples across experiences
    benign_per_exp = len(benign_indices) // n_experiences
    start_idx = exp_id * benign_per_exp
    end_idx = start_idx + benign_per_exp

    # For the last experience, take remaining samples
    if exp_id == n_experiences - 1:
        end_idx = len(benign_indices)

    return benign_indices[start_idx:end_idx]


def _subset_by_classes_with_scenario(
    data_np: np.ndarray,
    keep_class_ids: Sequence[int],
    scenario: int = 2,
    benign_class: int = 0,
    exp_id: int = 0,
    n_experiences: int = 4,
    benign_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Filter rows by labels ∈ keep_class_ids based on scenario type.

    Args:
        data_np: Input data array
        keep_class_ids: Class IDs to keep for this experience
        scenario: 1 (benign only in first exp) or 2 (benign split across all exps)
        benign_class: The benign class ID (default: 0)
        exp_id: Current experience ID
        n_experiences: Total number of experiences
        benign_indices: Pre-computed benign indices for this experience (for scenario 2)
    """
    labels = data_np[:, -1].astype(np.int64)

    # Handle non-benign classes normally
    non_benign_mask = np.isin(
        labels,
        np.array([c for c in keep_class_ids if c != benign_class], dtype=np.int64),
    )

    # Handle benign class based on scenario type
    benign_mask = np.zeros(len(labels), dtype=bool)

    if benign_class in keep_class_ids:
        if scenario == 2:
            # Scenario 2: Split benign samples across experiences
            if benign_indices is not None:
                # Use pre-computed benign indices
                benign_mask[benign_indices] = True
            else:
                # Fallback to old method (for backward compatibility)
                benign_indices = np.where(labels == benign_class)[0]
                if len(benign_indices) > 0:
                    benign_per_exp = len(benign_indices) // n_experiences
                    start_idx = exp_id * benign_per_exp
                    end_idx = start_idx + benign_per_exp

                    if exp_id == n_experiences - 1:
                        end_idx = len(benign_indices)

                    selected_benign_indices = benign_indices[start_idx:end_idx]
                    benign_mask[selected_benign_indices] = True

        elif scenario == 1:
            # Scenario 1: All benign samples only in first experience
            if exp_id == 0:
                benign_mask = labels == benign_class

    # Combine masks
    final_mask = non_benign_mask | benign_mask
    return data_np[final_mask]


def build_class_incremental_scenario(
    train_np: np.ndarray,
    val_np: np.ndarray,
    test_np: np.ndarray,
    all_classes: Sequence[str],  # global ordered names (len = num_class)
    categorical_indices_file: str,
    n_experiences: int,
    class_order: Optional[
        Sequence[int]
    ] = None,  # order of global class ids; default = [0..C-1]
    scenario: int = 1,  # 1 (benign only in first exp) or 2 (benign split across all exps)
) -> CIScenario:
    """
    Returns a CIScenario with n_experiences based on the specified scenario type.

    Args:
        train_np, val_np, test_np: Data arrays
        all_classes: Global ordered class names
        categorical_indices_file: Path to categorical indices file
        n_experiences: Number of experiences to create
        class_order: Order of global class ids
        scenario: 1 (benign only in first exp) or 2 (benign split across all exps)

    Returns:
        CIScenario object with experiences
    """
    num_class = len(all_classes)
    if class_order is None:
        class_order = list(range(num_class))
    else:
        class_order = list(class_order)
        assert len(set(class_order)) == num_class, (
            "class_order must include each class exactly once"
        )

    class_groups = _split_classes_into_experiences(class_order, n_experiences, scenario)

    # print("This is class groups:",class_groups)
    exps: List[CIExperience] = []

    # Handle scenario 2 with benign splitting across train/val/test
    if scenario == 2:
        for exp_id, cls_ids in enumerate(class_groups):
            cls_names = [all_classes[c] for c in cls_ids]

            # Get benign indices for each dataset separately but consistently
            benign_indices_train = _get_benign_split_indices(
                train_np, benign_class=0, exp_id=exp_id, n_experiences=n_experiences
            )
            benign_indices_val = _get_benign_split_indices(
                val_np, benign_class=0, exp_id=exp_id, n_experiences=n_experiences
            )
            benign_indices_test = _get_benign_split_indices(
                test_np, benign_class=0, exp_id=exp_id, n_experiences=n_experiences
            )

            # slice each split by class set, with benign samples split across experiences
            tr_np = _subset_by_classes_with_scenario(
                train_np,
                cls_ids,
                scenario,
                benign_class=0,
                exp_id=exp_id,
                n_experiences=n_experiences,
                benign_indices=benign_indices_train,
            )
            va_np = _subset_by_classes_with_scenario(
                val_np,
                cls_ids,
                scenario,
                benign_class=0,
                exp_id=exp_id,
                n_experiences=n_experiences,
                benign_indices=benign_indices_val,
            )
            te_np = _subset_by_classes_with_scenario(
                test_np,
                cls_ids,
                scenario,
                benign_class=0,
                exp_id=exp_id,
                n_experiences=n_experiences,
                benign_indices=benign_indices_test,
            )

            # Build train TabularDataset (fit=True → creates scaler), then reuse that scaler
            train_ds = TabularDataset(
                tr_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=True,
                scaler=None,
            )
            val_ds = TabularDataset(
                va_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=False,
                scaler=train_ds.scaler,
            )
            test_ds = TabularDataset(
                te_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=False,
                scaler=train_ds.scaler,
            )

            exps.append(
                CIExperience(
                    exp_id=exp_id,
                    class_ids=list(cls_ids),
                    class_names=cls_names,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                )
            )

    else:  # scenario == 1
        for exp_id, cls_ids in enumerate(class_groups):
            cls_names = [all_classes[c] for c in cls_ids]

            # slice each split by class set, with benign only in first experience
            tr_np = _subset_by_classes_with_scenario(
                train_np, cls_ids, scenario, benign_class=0, exp_id=exp_id
            )
            va_np = _subset_by_classes_with_scenario(
                val_np, cls_ids, scenario, benign_class=0, exp_id=exp_id
            )
            te_np = _subset_by_classes_with_scenario(
                test_np, cls_ids, scenario, benign_class=0, exp_id=exp_id
            )

            # Build train TabularDataset (fit=True → creates scaler), then reuse that scaler
            train_ds = TabularDataset(
                tr_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=True,
                scaler=None,
            )
            val_ds = TabularDataset(
                va_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=False,
                scaler=train_ds.scaler,
            )
            test_ds = TabularDataset(
                te_np,
                categorical_indices_file,
                task_classes=cls_names,
                all_classes=all_classes,
                fit=False,
                scaler=train_ds.scaler,
            )

            exps.append(
                CIExperience(
                    exp_id=exp_id,
                    class_ids=list(cls_ids),
                    class_names=cls_names,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    test_ds=test_ds,
                )
            )

    return CIScenario(exps)


def remap_labels(train, test, val, all_classes, num_class_prev):
    x_tr, label_tr = train[:, :-1], train[:, -1]
    x_ts, label_ts = test[:, :-1], test[:, -1]
    x_va, label_va = val[:, :-1], val[:, -1]
    # print(num_class_prev)

    # label_tr[label_tr > 0] += num_class_prev
    # label_ts[label_ts > 0] += num_class_prev
    # label_va[label_va > 0] += num_class_prev
    label_tr = label_tr + num_class_prev
    label_ts = label_ts + num_class_prev
    label_va = label_va + num_class_prev

    new_label = np.unique(label_tr).astype(int).tolist()

    class_group = [list(range(0, num_class_prev)), new_label]

    # print(class_group)

    train = np.c_[x_tr, label_tr]
    test = np.c_[x_ts, label_ts]
    val = np.c_[x_va, label_va]

    # sys.exit()
    return train, test, val, class_group


def build_dataset_incremental_scenario(
    data: np.ndarray,
    all_classes: Sequence[str],
    num_class_1,
    num_class_2,
    class_order: Optional[
        Sequence[int]
    ] = None,  # order of global class ids; default = [0..C-1]
) -> CIScenario:
    """
    Returns a CIScenario with experience = number of dataset. Each experience bundles train/val/test
    subset for its class group and shares a train-fitted scaler with its val/test.
    """
    num_class_total = len(all_classes)
    if class_order is None:
        class_order = list(range(num_class_total))
    else:
        class_order = list(class_order)
        assert len(set(class_order)) == num_class_total, (
            "class_order must include each class exactly once"
        )

    # class_groups = _split_classes_into_experiences(class_order, n_experiences)

    CIC_full, UNSW_full = data

    CIC_cat_idx_file, CIC_train_np, CIC_val_np, CIC_test_np = CIC_full
    UNSW_cat_idx_file, UNSW_train_np, UNSW_val_np, UNSW_test_np = UNSW_full

    UNSW_train_np, UNSW_val_np, UNSW_test_np, class_groups = remap_labels(
        UNSW_train_np, UNSW_val_np, UNSW_test_np, all_classes, num_class_1
    )

    # print("This is class groups:",class_groups)
    exps: List[CIExperience] = []

    exp = [
        # exp 0 (CIC)
        (0, CIC_cat_idx_file, CIC_train_np, CIC_val_np, CIC_test_np, class_groups[0]),
        # exp 1 (UNSW)
        (
            1,
            UNSW_cat_idx_file,
            UNSW_train_np,
            UNSW_val_np,
            UNSW_test_np,
            class_groups[1],
        ),
    ]

    # print(UNSW_train_np[:,0])

    # sys.exit()
    for exp_id, categorical_indices_file, tr_np, va_np, te_np, cls_ids in exp:
        cls_names = [all_classes[c] for c in cls_ids]

        # Build train TabularDataset (fit=True → creates scaler), then reuse that scaler
        # print(f"this is {exp_id} start")
        train_ds = TabularDataset(
            tr_np,
            categorical_indices_file,
            task_classes=cls_names,
            all_classes=all_classes,
            fit=True,
            scaler=None,
        )
        val_ds = TabularDataset(
            va_np,
            categorical_indices_file,
            task_classes=cls_names,
            all_classes=all_classes,
            fit=False,
            scaler=train_ds.scaler,
        )
        test_ds = TabularDataset(
            te_np,
            categorical_indices_file,
            task_classes=cls_names,
            all_classes=all_classes,
            fit=False,
            scaler=train_ds.scaler,
        )
        # print(f"this is {exp_id} end")

        exps.append(
            CIExperience(
                exp_id=exp_id,
                class_ids=list(cls_ids),
                class_names=cls_names,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
            )
        )

    return CIScenario(exps)
