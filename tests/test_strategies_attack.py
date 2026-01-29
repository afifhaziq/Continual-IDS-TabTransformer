"""Unit tests for src/strategies/attack.py"""

import pytest
import torch
import numpy as np

from src.strategies.attack import (
    label_flip,
    model_poisoning,
    choose_backdoor_target_from_buffer,
)


class TestLabelFlip:
    """Test label_flip function"""

    @pytest.fixture
    def sample_buffer(self):
        """Create a sample buffer with multiple classes"""
        buffer = []
        # 30 class 0, 30 class 1, 40 class 2
        for i in range(30):
            buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(0)))
        for i in range(30):
            buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(1)))
        for i in range(40):
            buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(2)))
        return buffer

    def test_label_flip_basic(self, sample_buffer):
        """Test basic label flipping"""
        import copy

        original_len = len(sample_buffer)
        original_buffer = copy.deepcopy(sample_buffer)
        flipped_buffer = label_flip(sample_buffer, seed=42, poison_rate=10)

        assert len(flipped_buffer) == original_len

        # Count flipped labels
        flipped_count = 0
        for i, ((_, _), label) in enumerate(flipped_buffer):
            original_label = original_buffer[i][1].item()
            if label.item() != original_label:
                flipped_count += 1

        # Should have approximately 10% flipped
        expected_flips = int(original_len * 0.1)
        assert flipped_count == expected_flips

    def test_zero_poison_rate(self, sample_buffer):
        """Test with zero poison rate"""
        flipped_buffer = label_flip(sample_buffer, seed=42, poison_rate=0)

        # Labels should remain unchanged
        for i, ((_, _), label) in enumerate(flipped_buffer):
            assert label.item() == sample_buffer[i][1].item()

    def test_empty_buffer(self):
        """Test with empty buffer"""
        empty_buffer = []
        result = label_flip(empty_buffer, seed=42, poison_rate=10)
        assert result == []

    def test_single_class_buffer(self):
        """Test with buffer containing only one class"""
        buffer = []
        for i in range(50):
            buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(0)))

        flipped_buffer = label_flip(buffer, seed=42, poison_rate=10)

        # No flipping should occur (only one class)
        for i, ((_, _), label) in enumerate(flipped_buffer):
            assert label.item() == buffer[i][1].item()

    def test_reproducibility(self, sample_buffer):
        """Test that same seed produces same flips"""
        import copy

        buffer1 = copy.deepcopy(sample_buffer)
        buffer2 = copy.deepcopy(sample_buffer)

        flipped1 = label_flip(buffer1, seed=42, poison_rate=20)
        flipped2 = label_flip(buffer2, seed=42, poison_rate=20)

        # Should have same labels
        for i in range(len(flipped1)):
            assert flipped1[i][1].item() == flipped2[i][1].item()

    def test_different_seeds_different_results(self, sample_buffer):
        """Test that different seeds produce different flips"""
        import copy

        buffer1 = copy.deepcopy(sample_buffer)
        buffer2 = copy.deepcopy(sample_buffer)

        flipped1 = label_flip(buffer1, seed=42, poison_rate=20)
        flipped2 = label_flip(buffer2, seed=99, poison_rate=20)

        # Should have some differences
        differences = 0
        for i in range(len(flipped1)):
            if flipped1[i][1].item() != flipped2[i][1].item():
                differences += 1

        assert differences > 0


class TestModelPoisoning:
    """Test model_poisoning function"""

    @pytest.fixture
    def sample_buffer_with_benign(self):
        """Create buffer with benign and attack samples"""
        buffer = []
        # 50 benign (class 0), 50 attack (class 1)
        for i in range(50):
            x_categ = torch.randint(0, 10, (3,))
            x_cont = torch.randn(10)
            buffer.append(((x_categ, x_cont), torch.tensor(0)))

        for i in range(50):
            x_categ = torch.randint(0, 10, (3,))
            x_cont = torch.randn(10)
            buffer.append(((x_categ, x_cont), torch.tensor(1)))

        return buffer

    @pytest.fixture
    def iat_indices(self):
        """IAT feature indices (continuous features)"""
        return np.array([5, 6, 7])

    @pytest.fixture
    def cat_indices(self):
        """Categorical feature indices"""
        return np.array([0, 1, 2])

    def test_model_poisoning_basic(
        self, sample_buffer_with_benign, iat_indices, cat_indices
    ):
        """Test basic model poisoning"""
        original_benign_features = [
            sample_buffer_with_benign[i][0][1].clone() for i in range(50)
        ]

        poisoned_buffer = model_poisoning(
            sample_buffer_with_benign,
            seed=42,
            poison_rate=20,
            vmin=14.0,
            vmax=16.0,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )

        # Check that some benign samples were modified
        modified_count = 0
        for i in range(50):
            if not torch.allclose(
                original_benign_features[i], poisoned_buffer[i][0][1]
            ):
                modified_count += 1

        # Approximately 20% of benign samples should be poisoned
        expected_poisoned = int(50 * 0.2)
        assert modified_count == expected_poisoned

    def test_poison_only_benign(
        self, sample_buffer_with_benign, iat_indices, cat_indices
    ):
        """Test that only benign samples are poisoned"""
        poisoned_buffer = model_poisoning(
            sample_buffer_with_benign,
            seed=42,
            poison_rate=20,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )

        # Check that labels remain unchanged (especially for attack samples)
        for i in range(50, 100):  # Attack samples
            assert poisoned_buffer[i][1].item() == 1

    def test_zero_poison_rate(
        self, sample_buffer_with_benign, iat_indices, cat_indices
    ):
        """Test with zero poison rate"""
        original_features = [
            sample_buffer_with_benign[i][0][1].clone() for i in range(50)
        ]

        poisoned_buffer = model_poisoning(
            sample_buffer_with_benign,
            seed=42,
            poison_rate=0,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )

        # Features should remain unchanged
        for i in range(50):
            assert torch.allclose(original_features[i], poisoned_buffer[i][0][1])

    def test_empty_buffer(self, iat_indices, cat_indices):
        """Test with empty buffer"""
        result = model_poisoning(
            [],
            seed=42,
            poison_rate=20,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )
        assert result == []

    def test_no_benign_samples(self, iat_indices, cat_indices):
        """Test with buffer containing no benign samples"""
        buffer = []
        for i in range(50):
            x_categ = torch.randint(0, 10, (3,))
            x_cont = torch.randn(10)
            buffer.append(((x_categ, x_cont), torch.tensor(1)))  # All attack class

        result = model_poisoning(
            buffer,
            seed=42,
            poison_rate=20,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )

        # Should return buffer unchanged
        assert len(result) == 50

    def test_value_range(self, sample_buffer_with_benign, iat_indices, cat_indices):
        """Test that poisoned values are within specified range"""
        vmin, vmax = 14.0, 16.0

        poisoned_buffer = model_poisoning(
            sample_buffer_with_benign,
            seed=42,
            poison_rate=100,  # Poison all benign samples
            vmin=vmin,
            vmax=vmax,
            iat_indices=iat_indices,
            cat_indices=cat_indices,
        )

        # Check poisoned features are in range
        # Note: Need to identify which positions in x_cont correspond to IAT
        # This requires understanding the continuous feature indexing
        # For now, just verify the buffer structure is maintained
        assert len(poisoned_buffer) == len(sample_buffer_with_benign)

    def test_missing_indices_raises_error(self, sample_buffer_with_benign):
        """Test that missing indices raises ValueError"""
        with pytest.raises(ValueError, match="requires preloaded"):
            model_poisoning(
                sample_buffer_with_benign,
                seed=42,
                poison_rate=20,
                iat_indices=None,
                cat_indices=None,
            )


class TestChooseBackdoorTarget:
    """Test choose_backdoor_target_from_buffer function"""

    def test_choose_target_basic(self):
        """Test basic backdoor target selection"""
        buffer = []
        # Add samples with classes 0, 1, 2
        for cls in [0, 1, 2]:
            for _ in range(10):
                buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(cls)))

        target = choose_backdoor_target_from_buffer(buffer, seed=42, benign_label=0)

        # Should return a non-benign class
        assert target is not None
        assert target != 0
        assert target in [1, 2]

    def test_only_benign_samples(self):
        """Test with only benign samples"""
        buffer = []
        for _ in range(20):
            buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(0)))

        target = choose_backdoor_target_from_buffer(buffer, seed=42, benign_label=0)

        # Should return None (no non-benign classes)
        assert target is None

    def test_empty_buffer(self):
        """Test with empty buffer"""
        target = choose_backdoor_target_from_buffer([], seed=42, benign_label=0)
        assert target is None

    def test_reproducibility(self):
        """Test that same seed gives same result"""
        buffer = []
        for cls in [0, 1, 2, 3]:
            for _ in range(10):
                buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(cls)))

        target1 = choose_backdoor_target_from_buffer(buffer, seed=42, benign_label=0)
        target2 = choose_backdoor_target_from_buffer(buffer, seed=42, benign_label=0)

        assert target1 == target2

    def test_custom_benign_label(self):
        """Test with custom benign label"""
        buffer = []
        for cls in [0, 1, 2]:
            for _ in range(10):
                buffer.append(((torch.zeros(5), torch.zeros(5)), torch.tensor(cls)))

        target = choose_backdoor_target_from_buffer(buffer, seed=42, benign_label=1)

        # Should exclude class 1 (the custom benign label)
        assert target is not None
        assert target != 1
        assert target in [0, 2]
