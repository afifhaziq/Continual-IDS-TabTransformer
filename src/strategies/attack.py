import random
import torch


def label_flip(buffer, seed, poison_rate):
    """
    Deterministically flip labels for a percentage of samples in the buffer.

    Parameters
    ----------
    buffer : list[tuple]
        List of samples, each as (features, label_tensor).
    seed : int
        Seed for reproducibility of which items are flipped and target labels.
    poison_rate : int or float
        Percentage of buffer to flip (0-100).

    Returns
    -------
    list[tuple]
        The same buffer list with a subset of labels flipped to an incorrect class.
    """
    if not buffer or poison_rate <= 0:
        return buffer

    rng = random.Random(seed)

    labels_present = sorted({int(sample[1].item()) for sample in buffer})
    total_samples = len(buffer)
    num_to_flip = int(total_samples * (poison_rate / 100.0))
    if num_to_flip <= 0 or len(labels_present) <= 1:
        return buffer

    flip_indices = rng.sample(range(total_samples), num_to_flip)

    for idx in flip_indices:
        features, label_tensor = buffer[idx]
        original_label = int(label_tensor.item())
        # pick a different label deterministically
        candidate_labels = [lbl for lbl in labels_present if lbl != original_label]
        if not candidate_labels:
            continue
        new_label = rng.choice(candidate_labels)
        new_label_tensor = torch.tensor(
            new_label, dtype=label_tensor.dtype, device=label_tensor.device
        )
        buffer[idx] = (features, new_label_tensor)

    return buffer


def model_poisoning(
    buffer,
    seed,
    poison_rate,
    vmin: float = 14.0,
    vmax: float = 16.0,
    iat_indices: list | None = None,
    cat_indices: list | None = None,
):
    """
    Poison a percentage of benign (label==0) samples by setting IAT feature
    columns to random values in [vmin, vmax]. Percentage is computed on the
    benign subset only.

    Parameters
    ----------
    buffer : list[tuple]
        List of samples, each as ((x_categ_tensor, x_cont_tensor), label_tensor).
    seed : int
        Seed for deterministic selection and value generation.
    poison_rate : int or float
        Percentage (0-100) of the BENIGN subset to poison.
    iat_index_file : str
        Numpy .npy file with global column indices of IAT features.
    cat_index_file : str
        Numpy .npy file with global column indices of categorical features.
    vmin, vmax : float
        Min/max of uniform range for poisoned IAT values.

    Returns
    -------
    list[tuple]
        The buffer with a subset of benign samples modified in-place.
    """
    if not buffer or poison_rate <= 0:
        return buffer

    rng = random.Random(seed)

    # Consider only benign samples (label==0)
    benign_indices = [
        i for i, sample in enumerate(buffer) if int(sample[1].item()) == 0
    ]
    if not benign_indices:
        return buffer

    # Compute target based on benign subset size
    target_poison = int(len(benign_indices) * (poison_rate / 100.0))
    if target_poison <= 0:
        return buffer

    chosen = rng.sample(benign_indices, target_poison)

    # Use provided indices (required)
    if iat_indices is None or cat_indices is None:
        raise ValueError(
            "model_poisoning requires preloaded iat_indices and cat_indices"
        )
    iat_global = list(iat_indices)
    cat_global = set(cat_indices)

    # Infer mapping: global feature index -> position within x_cont
    # Assumes buffer samples come from TabularDataset returning ((x_categ, x_cont), y)
    x_categ0, x_cont0 = buffer[0][0]
    num_categ = int(x_categ0.shape[0])
    num_cont = int(x_cont0.shape[0])
    total_features = num_categ + num_cont
    cont_indices = [i for i in range(total_features) if i not in cat_global]
    cont_pos = {g: p for p, g in enumerate(cont_indices)}
    # All IAT are guaranteed continuous; direct mapping
    iat_pos = [cont_pos[g] for g in iat_global]

    for idx in chosen:
        (x_categ, x_cont), y = buffer[idx]
        device, dtype = x_cont.device, x_cont.dtype
        pos_tensor = torch.tensor(iat_pos, device=device, dtype=torch.long)
        # Per-feature random values in [vmin, vmax]
        values = (
            torch.rand(len(iat_pos), device=device, dtype=dtype) * (vmax - vmin)
        ) + vmin
        # In-place write
        x_cont.index_copy_(0, pos_tensor, values)
        buffer[idx] = ((x_categ, x_cont), y)

    return buffer


def choose_backdoor_target_from_buffer(buffer, seed, benign_label: int = 0):
    """
    Choose a random non-benign class label present in the buffer (seeded).

    Parameters
    ----------
    buffer : list[tuple]
        List of samples, each as ((features), label_tensor).
    seed : int
        Seed for deterministic selection.
    benign_label : int
        Label id considered benign (default: 0).

    Returns
    -------
    int | None
        Chosen non-benign class label, or None if none available.
    """
    non_benign = sorted(
        {int(lbl.item()) for (_, lbl) in buffer if int(lbl.item()) != benign_label}
    )
    if not non_benign:
        return None
    rng = random.Random(seed)
    return rng.choice(non_benign)
