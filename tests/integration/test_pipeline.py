"""Integration tests for the full training pipeline"""

import pytest
import numpy as np


class TestPipelineIntegration:
    """Integration tests for the complete pipeline"""

    @pytest.fixture
    def synthetic_dataset(self, tmp_path):
        """Create a minimal synthetic dataset for testing"""
        num_samples = 200
        num_features = 10
        num_classes = 4

        # Generate synthetic data
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(0, num_classes, size=(num_samples, 1))
        data = np.hstack([X, y])

        # Save data files
        dataset_dir = tmp_path / "CICIDS2017"
        dataset_dir.mkdir()

        np.save(dataset_dir / "train.npy", data)
        np.save(dataset_dir / "val.npy", data[:50])
        np.save(dataset_dir / "test.npy", data[:50])

        # Save categorical indices
        cat_indices = np.array([0, 1])
        np.save(dataset_dir / "catfeaturelist.npy", cat_indices)

        return dataset_dir

    def test_data_loading(self, synthetic_dataset):
        """Test that data can be loaded successfully"""
        train_data = np.load(synthetic_dataset / "train.npy")
        val_data = np.load(synthetic_dataset / "val.npy")
        test_data = np.load(synthetic_dataset / "test.npy")
        cat_indices = np.load(synthetic_dataset / "catfeaturelist.npy")

        assert train_data.shape[0] == 200
        assert val_data.shape[0] == 50
        assert test_data.shape[0] == 50
        assert len(cat_indices) == 2

    def test_scenario_creation(self, synthetic_dataset):
        """Test creating a continual learning scenario"""
        from src.data.ci_builder import build_class_incremental_scenario

        train_data = np.load(synthetic_dataset / "train.npy")
        val_data = np.load(synthetic_dataset / "val.npy")
        test_data = np.load(synthetic_dataset / "test.npy")

        all_classes = ["Class0", "Class1", "Class2", "Class3"]
        cat_file = str(synthetic_dataset / "catfeaturelist.npy")

        scenario = build_class_incremental_scenario(
            train_np=train_data,
            val_np=val_data,
            test_np=test_data,
            all_classes=all_classes,
            categorical_indices_file=cat_file,
            n_experiences=2,
            scenario=1,
        )

        assert len(scenario) == 2
        assert scenario[0].exp_id == 0
        assert scenario[1].exp_id == 1

    def test_experience_replay_workflow(self, synthetic_dataset):
        """Test the experience replay workflow"""
        from src.data.ci_builder import build_class_incremental_scenario
        from src.strategies.replay import build_replay_buffer

        train_data = np.load(synthetic_dataset / "train.npy")
        val_data = np.load(synthetic_dataset / "val.npy")
        test_data = np.load(synthetic_dataset / "test.npy")

        all_classes = ["Class0", "Class1", "Class2", "Class3"]
        cat_file = str(synthetic_dataset / "catfeaturelist.npy")

        scenario = build_class_incremental_scenario(
            train_np=train_data,
            val_np=val_data,
            test_np=test_data,
            all_classes=all_classes,
            categorical_indices_file=cat_file,
            n_experiences=2,
            scenario=1,
        )

        # Build replay buffer from first experience
        exp0 = scenario[0]
        seen_classes = set(exp0.class_ids)

        buffer = build_replay_buffer(
            experiences=[exp0],
            seen_class_ids=seen_classes,
            memory_percentage=10,
            seed=42,
            balanced=False,
        )

        assert len(buffer) > 0
        assert all(isinstance(item, tuple) for item in buffer)
