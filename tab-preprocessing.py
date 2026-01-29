import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import time


# Configuration
CONFIG_PATH = "configs/config.yaml"
DATASET_NAME = "CICIDS2017"
FOLDER_PATH = "preprocess_csv/CICIDS2017/"
INPUT_FILE = "CICIDS2017_standardised.csv"
TEST_SIZE = 0.4
VAL_TEST_SPLIT = 0.5
RANDOM_STATE = 42


def load_config(config_path, dataset_name):
    """Load configuration from YAML file and extract class names."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    class_names = config["datasets"][dataset_name]["classes"]
    print(f"Loaded {len(class_names)} class names from config: {dataset_name}")
    return class_names


def load_data(file_path):
    """Load CSV file and return DataFrame."""
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {df.shape[0]} rows from {INPUT_FILE}.")
    return df


def clean_data(df):
    """Remove infinity and NaN values from DataFrame."""
    # Check for infinity values
    inf_counts = df.isin([np.inf, -np.inf]).sum()
    problematic_cols = inf_counts[inf_counts > 0]

    if not problematic_cols.empty:
        print("\nFound columns with infinity values:")
        print(problematic_cols)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Replaced all infinity values with NaN.")
    else:
        print("No infinity values found in the dataset.")

    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    print("Replaced all NaN values with 0.")

    return df


def split_data(X, y, test_size, val_test_split, random_state):
    """Split data into train, validation, and test sets."""
    # Split into train and temp (which will be split into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(
        f"Data split into training and temp sets ({int((1 - test_size) * 100)}/{int(test_size * 100)} split)."
    )

    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_test_split,
        random_state=random_state,
        stratify=y_temp,
    )
    print(
        f"Temp data split into validation and test sets ({int((1 - val_test_split) * 100)}/{int(val_test_split * 100)} split)."
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def print_class_distribution(y_partition, partition_name, class_names):
    """Print class distribution for a given partition."""
    _, num_instances = np.unique(y_partition, return_counts=True)
    for class_name, count in zip(class_names, num_instances):
        print(f"{partition_name}: Class {class_name} : {count} instances")


def save_processed_data(
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, folder_path
):
    """Save processed data to .npy files."""
    # Reshape labels
    y_train = y_train.values.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    # Concatenate features and labels
    train = np.concatenate((X_train, y_train), axis=1)
    val = np.concatenate((X_val, y_val), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    # Save to files
    np.save(f"{folder_path}train.npy", train)
    np.save(f"{folder_path}val.npy", val)
    np.save(f"{folder_path}test.npy", test)
    np.save(f"{folder_path}class_names.npy", class_names)

    # Print shapes
    print("\nFinal dataset shapes:")
    for name, array in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name}: {array.shape}")

    return train, val, test


def main():
    """Main preprocessing pipeline."""
    print("--- Starting Data Preprocessing ---")
    start_time = time.time()

    # Step 1: Load configuration
    class_names = load_config(CONFIG_PATH, DATASET_NAME)

    # Step 2: Load data
    file_path = f"{FOLDER_PATH}{INPUT_FILE}"
    df = load_data(file_path)

    # Step 3: Clean data
    df = clean_data(df)

    # Step 4: Separate features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(f"Separated features ({X.shape[1]} columns) and labels.")

    # Step 5: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, TEST_SIZE, VAL_TEST_SPLIT, RANDOM_STATE
    )

    # Step 6: Print class distributions
    print("\nClass distributions:")
    print_class_distribution(y_train, "train", class_names)
    print_class_distribution(y_val, "val", class_names)
    print_class_distribution(y_test, "test", class_names)

    # Step 7: Save processed data
    print("\nSaving processed data...")
    save_processed_data(
        X_train, X_val, X_test, y_train, y_val, y_test, class_names, FOLDER_PATH
    )

    # Completion
    end_time = time.time()
    print("\n--- Preprocessing Complete ---")
    print("Saved 4 files (train.npy, val.npy, test.npy, class_names.npy)")
    print(f"Total preprocessing time: {(end_time - start_time):.2f} seconds.")


if __name__ == "__main__":
    main()
