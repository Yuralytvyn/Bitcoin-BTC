"""
Dataset splitting utility for cryptocurrency time‑series.

This script provides a simple interface for splitting a feature‑engineered
dataset into chronologically ordered training (and optionally validation
and test) sets.  It is intended to be used after applying feature
engineering to raw OHLCV data.  Key parameters such as the input
filename and the default proportion of data reserved for testing are
defined as top‑level constants for easy configuration.  You can still
override ``test_size`` and specify a validation split when calling
``split_dataset``.

Usage
-----
You can run this module directly to create the splits using the default
configuration:

    python split_train_test.py

Alternatively, import the ``split_dataset`` function in your own code to
programmatically generate splits with custom arguments.
"""

import os
from typing import Optional, Tuple

import pandas as pd

try:
    # Attempt to import DATA_DIR from an external init module; fall back to CWD
    from init import DATA_DIR  # type: ignore
except Exception:
    DATA_DIR = os.getcwd()

# =============================================================================
# Configuration variables
#
# Adjust these values to change the behaviour of the splitting routine.  When
# tweaking splits, consider that the data will be divided chronologically: the
# earliest samples form the training set and the most recent samples form the
# test (and optional validation) sets.  This avoids look‑ahead bias.
# =============================================================================

# Name of the feature‑engineered CSV file to split (relative to DATA_DIR)
FILENAME: str = "btc_hourly_5y_feature.csv"

# Fraction of the total dataset to reserve for the test set (between 0 and 1)
TEST_SIZE: float = 0.2



def train_test_split(
    filename: str = FILENAME,
    test_size: float = TEST_SIZE,
    create_validation: bool = False,
    validation_size: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Split a feature dataset into training, optional validation and test sets.

    The split is performed in chronological order to preserve the temporal
    structure of time‑series data.  The resulting DataFrames are returned
    directly and also saved to disk with appropriate suffixes.

    Parameters
    ----------
    filename : str
        Name of the CSV file containing the feature matrix and target
        variable.  The file must reside in ``DATA_DIR``.
    test_size : float
        Proportion of the dataset to allocate to the test set.  Must be in
        the interval ``(0, 1)``.
    create_validation : bool
        Whether to carve out a validation set from the end of the training
        portion.  Useful when tuning hyperparameters without touching the test
        set.
    validation_size : float
        Proportion of the training data (after the test split) to allocate
        to the validation set.  Ignored if ``create_validation`` is ``False``.

    Returns
    -------
    tuple
        A tuple containing the training DataFrame, test DataFrame and, if
        requested, the validation DataFrame.  If validation is not created,
        the third element of the tuple will be ``None``.
    """
    # Validate parameters
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive)")
    if create_validation and not (0.0 < validation_size < 1.0):
        raise ValueError("validation_size must be between 0 and 1 when creating a validation set")
    # Load the dataset
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    n_samples = len(df)
    # Compute indices for splitting
    test_split_idx = int(n_samples * (1.0 - test_size))
    train_df = df.iloc[:test_split_idx].copy()
    test_df = df.iloc[test_split_idx:].copy()
    val_df: Optional[pd.DataFrame] = None
    if create_validation:
        val_length = int(len(train_df) * validation_size)
        # Split tail of training set into validation
        val_df = train_df.iloc[-val_length:].copy()
        train_df = train_df.iloc[:-val_length].copy()
    # Determine base name for output files
    base_name = filename.rsplit(".csv", 1)[0]
    # Save train/test/val to disk
    train_path = os.path.join(DATA_DIR, f"{base_name}_train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"Saved: {train_path}")
    test_path = os.path.join(DATA_DIR, f"{base_name}_test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {test_path}")
    if val_df is not None:
        val_path = os.path.join(DATA_DIR, f"{base_name}_val.csv")
        val_df.to_csv(val_path, index=False)
        print(f"Saved: {val_path}")
    return train_df, test_df, val_df