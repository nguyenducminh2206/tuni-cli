import pandas as pd
import h5py
import numpy as np
import os
import re


def read_file(folder_path):
    """
    Find all .h5 files 
    """
    h5_files = []
    files = os.listdir(folder_path)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files


def build_df(data_path, n_samples_per_file: int = 1):
    """
    Build a dataframe from a directory of HDF5 files.

    For each .h5 file, take the first `n_samples_per_file` samples, and extract:
    - time traces (per cell)
    - distance to target (per cell)
    - selected feature vectors (per feature, per sample, per cell), filling missing with NaN.
    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        # Extract metadata from filename
        noise_level = extract_noise(os.path.basename(file_path))
        with h5py.File(file_path, 'r') as f:
            sample_keys = sorted(f['timeTraces'].keys(), key=lambda x: int(x))[:max(1, int(n_samples_per_file))]
            feature_names = list(f['features'].keys())

            for sample_idx, sample_key in enumerate(sample_keys):
                time_traces = np.array(f['timeTraces'][sample_key]).T  # (n_cells, n_timepoints)
                distance_to_target = np.array(f['tissue']['distanceToTarget'][()])
                n_cells = time_traces.shape[0]

                # For each feature, get vector for this sample_key (handle missing and shape)
                feature_vectors = {}
                for feature in feature_names:
                    if sample_key in f['features'][feature]:
                        vector = np.array(f['features'][feature][sample_key])  # shape (1, 25) or (0, 25)
                        if vector.shape == (1, n_cells):
                            feature_vectors[feature] = vector[0, :]  # shape (25,)
                        else:
                            # Missing, empty, or wrong shape: fill with NaN
                            feature_vectors[feature] = np.full(n_cells, np.nan)
                    else:
                        feature_vectors[feature] = np.full(n_cells, np.nan)

                for cell_id in range(n_cells):
                    row = {
                        #'simulation_id': simulation_ids[sample_idx],
                        #'sample_key': sample_key,
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'simulation_file': os.path.basename(file_path),
                        'noise': noise_level,
                        'cMax': feature_vectors['cMax'][cell_id],
                        'cVar': feature_vectors['cVariance'][cell_id]
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def extract_noise(filename):
    match = re.search(r'noise[_\-]?([0-9.]+)', filename)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def balance_data(df: pd.DataFrame, y_col: str, *, random_state: int = 42) -> pd.DataFrame:
    """Undersample each class to the minimum class count based on y_col.

    General and simple: finds min count across classes in df[y_col] and samples
    that many rows from each class with a fixed random_state. If y_col is missing
    or there are no valid labels, returns df unchanged.
    """
    if y_col not in df.columns:
        return df
    counts = df[y_col].dropna().value_counts()
    if counts.empty:
        return df
    n_samples = int(counts.min())
    if n_samples <= 0:
        return df
    return (
        df.dropna(subset=[y_col])
          .groupby(y_col, group_keys=False)
          .apply(lambda x: x.sample(n=n_samples, random_state=42))
          .reset_index(drop=True)
    )

