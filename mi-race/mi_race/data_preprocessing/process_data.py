import pandas as pd
import h5py
import numpy as np
import os
import re


#----------- Build data frame from H5DF files (specified for tuni-data) ---------------
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


def extract_noise(filename):
    match = re.search(r'noise[_\-]?([0-9.]+)', filename)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except Exception:
        return np.nan


def build_df(data_path, n_samples_per_file: int = 25) -> pd.DataFrame:
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
        fname = os.path.basename(file_path)
        noise_level = extract_noise(fname)
        kcross_level = extract_kcross(fname)
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
                        'cell_id': cell_id,
                        'time_trace': time_traces[cell_id],
                        'dis_to_target': distance_to_target[cell_id],
                        'noise': noise_level,
                        'kcross': kcross_level,
                        'cMax': feature_vectors['cMax'][cell_id],
                        'cVar': feature_vectors['cVariance'][cell_id]
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df

def extract_kcross(filename):
    """Extract kcross value from a filename like '..._kcross_0.0050_...'.

    Returns float value if found, else NaN.
    """
    match = re.search(r'kcross[_\-]?([0-9.]+)', filename)
    if not match:
        return np.nan
    try:
        return float(match.group(1))
    except Exception:
        return np.nan