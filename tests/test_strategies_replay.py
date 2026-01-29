"""Unit tests for src/strategies/replay.py"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from src.strategies.replay import (
    ReplayDataset,
    build_replay_buffer,
    print_buffer_distribution,
)


class TestReplayDataset:
    """Test ReplayDataset class"""

    @pytest.fixture
    def sample_buffer(self):
        """Create a sample replay buffer"""
        buffer = []
        for i in range(50):
            x_categ = torch.randint(0, 10, (3,))
            x_cont = torch.randn(7)
            label = torch.tensor(i % 3)
            buffer.append(((x_categ, x_cont), label))
        return buffer

    def test_initialization(self, sample_buffer):
        """Test ReplayDataset initialization"""
        dataset = ReplayDataset(sample_buffer)
        assert len(dataset) == 50
        assert dataset.buffer is sample_buffer

    def test_getitem(self, sample_buffer):
        """Test __getitem__ method"""
        dataset = ReplayDataset(sample_buffer)
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 2
        (x_categ, x_cont), label = item
        assert isinstance(x_categ, torch.Tensor)
        assert isinstance(x_cont, torch.Tensor)
        assert isinstance(label, torch.Tensor)


class TestBuildReplayBuffer:
    """Test build_replay_buffer function"""

    @pytest.fixture
    def mock_experience(self):
        """Create a mock experience with dataset"""
        exp = Mock()
        exp.class_ids = [0, 1]

        # Create mock dataset
        mock_ds = Mock()
        mock_ds.labels = np.array([0] * 50 + [1] * 50)

        # Mock __getitem__ to return sample (needs self parameter)
        def getitem(self, idx):
            label_val = mock_ds.labels[idx]
            x_categ = torch.randint(0, 10, (3,))
            x_cont = torch.randn(7)
            label = torch.tensor(label_val)
            return ((x_categ, x_cont), label)

        mock_ds.__getitem__ = getitem
        exp.train_ds = mock_ds
        exp.val_ds = mock_ds

        return exp

    def test_basic_buffer_building(self, mock_experience):
        """Test basic replay buffer building"""
        buffer = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        assert len(buffer) > 0
        assert all(isinstance(item, tuple) for item in buffer)

    def test_balanced_sampling(self, mock_experience):
        """Test balanced sampling mode"""
        buffer = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            balanced=True,
        )

        # Count samples per class
        class_counts = {}
        for (_, _), label in buffer:
            class_id = int(label.item())
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # In balanced mode, classes should have similar counts
        counts = list(class_counts.values())
        if len(counts) > 1:
            assert max(counts) - min(counts) <= 1  # At most 1 sample difference

    def test_percentage_sampling(self, mock_experience):
        """Test percentage-based sampling"""
        buffer_10 = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        buffer_20 = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=20,
            seed=42,
            balanced=False,
        )

        # 20% should have more samples than 10%
        assert len(buffer_20) >= len(buffer_10)

    def test_validation_set_sampling(self, mock_experience):
        """Test sampling from validation set"""
        buffer = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            use_validation_set=True,
            balanced=False,
        )

        assert len(buffer) > 0

    def test_min_samples_per_class(self, mock_experience):
        """Test minimum samples per class constraint"""
        buffer = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=1,  # Very low percentage
            seed=42,
            min_samples_per_class=5,
            balanced=False,
        )

        # Count samples per class
        class_counts = {}
        for (_, _), label in buffer:
            class_id = int(label.item())
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # Each class should have at least min_samples_per_class
        for count in class_counts.values():
            assert count >= 5

    def test_empty_experience_handling(self):
        """Test handling of experiences with no samples for seen classes"""
        exp = Mock()
        exp.class_ids = [5, 6]  # Different classes than seen_class_ids
        mock_ds = Mock()
        mock_ds.labels = np.array([5] * 50)

        def getitem(idx):
            return ((torch.zeros(3), torch.zeros(7)), torch.tensor(5))

        mock_ds.__getitem__ = getitem
        exp.train_ds = mock_ds
        exp.val_ds = mock_ds

        buffer = build_replay_buffer(
            experiences=[exp],
            seen_class_ids={0, 1},  # Different from exp classes
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        # Should return empty or minimal buffer
        assert isinstance(buffer, list)

    def test_reproducibility_with_seed(self, mock_experience):
        """Test that same seed produces same buffer"""
        buffer1 = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        buffer2 = build_replay_buffer(
            experiences=[mock_experience],
            seen_class_ids={0, 1},
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        assert len(buffer1) == len(buffer2)

        # Compare labels
        labels1 = [int(item[1].item()) for item in buffer1]
        labels2 = [int(item[1].item()) for item in buffer2]
        assert labels1 == labels2


class TestPrintBufferDistribution:
    """Test print_buffer_distribution function"""

    @pytest.fixture
    def sample_buffer(self):
        """Create a sample buffer with known distribution"""
        buffer = []
        # 30 samples of class 0, 20 of class 1
        for i in range(30):
            buffer.append(((torch.zeros(3), torch.zeros(7)), torch.tensor(0)))
        for i in range(20):
            buffer.append(((torch.zeros(3), torch.zeros(7)), torch.tensor(1)))
        return buffer

    def test_print_buffer_distribution(self, sample_buffer, capsys):
        """Test that function prints correct distribution"""
        print_buffer_distribution(sample_buffer, "Test")

        captured = capsys.readouterr()
        assert "Test replay buffer size: 50" in captured.out
        assert "Class 0:" in captured.out
        assert "Class 1:" in captured.out

    def test_empty_buffer(self, capsys):
        """Test with empty buffer"""
        print_buffer_distribution([], "Empty")

        captured = capsys.readouterr()
        assert "Empty replay buffer size: 0" in captured.out
