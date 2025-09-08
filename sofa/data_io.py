"""
Data input/output operations and file management.

This module contains functions for loading and saving data in various formats,
including CSV, JSON, HDF5, and joblib pickle files, with backup functionality.
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import h5py
import polars as pl
from scipy.io import loadmat
from joblib import dump, load


def load_16gbaud_db(path: Path) -> pl.DataFrame:
    dfs = []
    for directory in path.iterdir():
        if directory.is_file():
            continue
        name = directory.name
        GHz_index = name.find("GHz")
        spacing = name[:GHz_index]
        if GHz_index == -1:
            spacing = "50"

        for subdir in directory.iterdir():
            if subdir.is_dir():
                continue
            name = subdir.name
            consY_index = name.find("consY") + len("consY")
            dB_index = name.find("dB")
            osnr = name[consY_index:dB_index]

            # Load CSV into DF and add spacing and OSNR columns
            read_df = pl.read_csv(
                subdir, schema={"I": pl.Float64, "Q": pl.Float64})
            read_df = read_df.with_columns(
                [
                    pl.lit(float(spacing)).alias("Spacing"),
                    pl.lit(float(osnr)).alias("OSNR"),
                ]
            )
            dfs.append(read_df)
        df = pl.concat(dfs, rechunk=True)
    return df


def load_32gbaud_db(
    path: Path, full: bool = False, subfolder: str = "0km_0dBm"
) -> pl.DataFrame:
    # Check subfolder parameter
    if subfolder not in ["0km_0dBm", "270km_0dBm", "270km_9dBm"]:
        raise ValueError("Invalid subfolder name.")

    METADATA_PATTERN = re.compile(
        r"Song\d+_[XY]_"  # Match the prefix (e.g., Song1_X or Song1_Y)
        # Match OSNR (integer or decimal with 'p')
        r"(?P<OSNR>[\d]+(?:p[\d]+)?)dB_"
        # Match Spacing (integer or decimal with 'p')
        r"(?P<Spacing>[\d]+(?:p[\d]+)?)GHz_"
        # Match Distance (integer or decimal with 'p')
        r"(?P<Distance>[\d]+(?:p[\d]+)?)km_"
        # Match Power (integer or decimal with 'p')
        r"(?P<Power>[\d]+(?:p[\d]+)?)dBm"
    )
    dfs = []
    for directory in path.iterdir():
        if not directory.is_dir():
            continue
        # Check directory name when not loading whole DB
        if not full and directory.name != subfolder:
            continue
        for file_path in directory.rglob("*.mat"):
            match = METADATA_PATTERN.search(file_path.name)
            if not match:
                raise ValueError(
                    f"File name {
                        file_path.name} does not match the expected pattern."
                )

            metadata = {
                key: float(value.replace("p", "."))
                for key, value in match.groupdict().items()
            }

            mat = loadmat(file_path)
            mat = mat["rconst"][0]

            I = mat.real
            Q = mat.imag

            df = pl.DataFrame(
                {
                    "I": I,
                    "Q": Q,
                    "Distance": metadata["Distance"],
                    "Power": metadata["Power"],
                    "OSNR": metadata["OSNR"],
                    "Spacing": metadata["Spacing"],
                }
            )
            dfs.append(df)

    # Combinar todos los dataframes en uno solo
    df_32gbd = pl.concat(dfs, rechunk=True)
    return df_32gbd


def __should_skip_save(filename: str, n_backups: int = -1) -> bool:
    """
    Checks if saving a file should be skipped based on n_backups and file existence.

    Args:
        filename (str): The name of the file to check.
        n_backups (int, optional): The number of backups to keep (affects skipping). Defaults to -1.

    Returns:
        bool: True if saving should be skipped, False otherwise.
    """
    if n_backups == -1 and os.path.exists(filename):
        logging.getLogger(__name__).warning(
            f"Skipping saving {filename} (n_backups=-1 and file exists)."
        )
        return True
    else:
        return False


def __do_backup(filename: str, n_backups: int = 0) -> None:
    """
    Perform backup rotation for a file, keeping a specified number of backups.

    Parameters:
        filename (str): The name of the file to create or overwrite.
        n_backups (int, optional): The number of backup files to keep. Defaults to 0.
            - **-1**: Skips backup entirely (no changes made).
            - **0**: Overwrites the existing file (no backups kept).
            - **Positive value**:
                - Rotates existing backups to keep no more than `n_backups` versions.
                - Creates a new backup of the original file (if it exists) with the highest index.

    Returns:
        None
    """

    # Function to get backup filenames
    def backup_filename(index):
        return f"{filename}.bak{index}"

    # Check for n_backups sentinel value (-1) to skip backup logic
    if n_backups == -1:
        logging.getLogger(__name__).warning(
            f"Skipping backup for {filename} (n_backups=-1)."
        )
        return

    # Backup logic for positive n_backups
    for i in range(n_backups, 0, -1):
        src = backup_filename(i - 1) if i - 1 > 0 else filename
        dst = backup_filename(i)
        os.rename(src, dst) if os.path.exists(src) else None


def save_json(data: dict, filename: str, n_backups: int = 3) -> None:
    """
    Save data to a JSON file with backup rotation.

    Parameters:
        data (dict): A dictionary containing datasets to be saved.
        filename (str): The name of the JSON file to create or overwrite.
        n_backups (int, optional): The number of backup files to keep. Defaults to 3.
            - **-1**: Skips saving entirely if the file already exists.
            - **0**: Overwrites the existing file (no backups kept).
            - **Positive value**:
                - Performs backup rotation using `__do_backup`.
                - Overwrites the existing file with the new data.

    Returns:
        None
    """
    # Check for n_backups and handle skipping if necessary
    if __should_skip_save(filename, n_backups):
        return

    # Perform backup rotation (unless skipped)
    __do_backup(filename, n_backups)

    try:
        # Save data to the main file
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def load_json(filename: str) -> dict:
    """
    Load data from a JSON file.

    Parameters:
        filename (str): The name of the JSON file to load data from.

    Returns:
        defaultdict: A nested defaultdict containing the loaded data.
    """

    def dict_factory():
        return defaultdict(dict_factory)

    loaded_data = defaultdict(dict_factory)

    try:
        # Load data from the file
        with open(filename, "r") as json_file:
            loaded_data = json.load(json_file)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

    return loaded_data


def save_hdf5(data: dict, filename: str, n_backups: int = 3) -> None:
    """
    DEPRECATED: old structure was too complex
    Save data to an HDF5 file with backup rotation.

    Parameters:
        data (dict): A dictionary containing datasets to be saved.
        filename (str): The name of the HDF5 file to create or overwrite.
        n_backups (int, optional): The number of backup files to keep. Defaults to 3.
            - -1: Skips saving entirely if the file already exists.
            - 0: Overwrites the existing file (no backups kept).
            - Positive value:
                - Performs backup rotation using `__do_backup`.
                - Overwrites the existing file with the new data.

    Returns:
        None
    """
    # Check for n_backups and handle skipping if necessary
    if __should_skip_save(filename, n_backups):
        return

    # Perform backup rotation (unless skipped)
    __do_backup(filename, n_backups)

    try:
        # Save data to the main file
        with h5py.File(filename, "w") as f:

            def store_dict(group, data_dict):
                for key, value in data_dict.items():
                    if isinstance(value, (dict, defaultdict)):
                        subgroup = group.create_group(key)
                        store_dict(subgroup, value)
                    elif key == "model":
                        # Save model as JSON
                        group.create_dataset(key, data=json.dumps(value))
                    elif key in {"loss", "train", "test", "prod"}:
                        # Save k-fold scores in separate groups
                        scores_group = group.create_group(key)
                        for i, vector in enumerate(value, start=1):
                            scores_group.create_dataset(str(i), data=vector)
                    else:
                        # Save other keys as NumPy arrays
                        group.create_dataset(key, data=value)

            store_dict(f, data)

    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def load_hdf5(filename: str):
    """
    DEPRECATED: old structure was too complex
    Load data from an HDF5 file.
    This function recursively loads data from an HDF5 file.

    Parameters:
        filename (str): The name of the HDF5 file to load data from.

    Returns:
        defaultdict: A nested defaultdict containing the loaded data.
    """

    def dict_factory():
        return defaultdict(dict_factory)

    loaded_data = defaultdict(dict_factory)

    with h5py.File(filename, "r") as f:

        def load_dict(group):
            data_dict = defaultdict(dict_factory)
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    data_dict[key] = load_dict(group[key])
                elif isinstance(group[key], h5py.Dataset):
                    if key == "model":
                        data_dict[key] = json.loads(
                            group[key][()].decode("utf-8"))
                    else:
                        data_dict[key] = group[key][()]
            return data_dict

        loaded_data = load_dict(f)

    return loaded_data


def joblib_load(file):
    try:
        return load(file)
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"File {file} not found")
        return None


def joblib_save(var, file):
    dump(var, file)
