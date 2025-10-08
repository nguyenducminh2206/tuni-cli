import pandas as pd
import h5py
import numpy as np
import os
import re
from pathlib import Path


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


def build_df(data_path):
    """
    For each .h5 file, for the first 100 samples, extract time traces, distance to target,
    and all feature values (per feature, per sample, per cell), filling missing with NaN.
    """
    rows = []
    file_paths = read_file(data_path)

    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f:
            sample_keys = sorted(f['timeTraces'].keys(), key=lambda x: int(x))[:1]  # first 100 samples
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
                        #'simulation_file': os.path.basename(file_path),
                        'cMax': feature_vectors['cMax'][cell_id],
                        'cVar': feature_vectors['cVariance'][cell_id]
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def extract_noise(filename):
    match = re.search(r'noise[_\-]?([0-9.]+)', filename)
    return float(match.group(1) if match else None)


def main():
    # Example: expand and export time_trace into 1000 columns CSV under processed_data
    out_path = export_time_trace_to_csv("data_7x7", n_cols=1000)
    print(f"[mi-race] Wrote: {out_path}")

def _safe_array(x) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    return np.array([], dtype=float)


def expand_time_trace_columns(series: pd.Series, prefix: str = "time_trace", n_cols: int = 1000) -> pd.DataFrame:
    """
    Expand a column of sequences (list/array) into fixed number of columns.
    - Outputs columns: prefix_1, ..., prefix_{n_cols}
    - If a sequence is shorter than n_cols -> pad with NaN
    - If longer -> truncate
    """
    arrays = series.apply(_safe_array)
    n_rows = len(arrays)
    out = np.full((n_rows, n_cols), np.nan, dtype=float)
    for i, arr in enumerate(arrays):
        m = min(arr.size, n_cols)
        if m:
            out[i, :m] = arr[:m]
    cols = [f"{prefix}_{i}" for i in range(1, n_cols + 1)]
    return pd.DataFrame(out, columns=cols, index=series.index)


def export_time_trace_to_csv(
    data_path: str | os.PathLike,
    out_dir: str | os.PathLike = "processed_data",
    filename: str | None = None,
    n_cols: int = 1000,
) -> str:
    """
    Build a dataframe from HDF5 dataset id or directory, expand `time_trace` into
    `n_cols` columns (time_trace_1..time_trace_{n_cols}), and save CSV under processed_data.

    Returns absolute path to the written CSV file.
    """
    # Resolve project root (mi-race/)
    project_root = Path(__file__).resolve().parents[2]

    # Resolve input location relative to project root if not absolute
    data_path = Path(data_path)
    if not data_path.is_absolute():
        candidate = project_root / data_path
        data_path = candidate if candidate.exists() else data_path

    # Build dataframe from HDF5 directory or dataset id string
    df = build_df(str(data_path))

    if "time_trace" not in df.columns:
        raise SystemExit("[mi-race] 'time_trace' column not found in built dataframe.")

    tt_df = expand_time_trace_columns(df["time_trace"], prefix="time_trace", n_cols=n_cols)

    # Combine with non-time_trace columns
    base_df = df.drop(columns=["time_trace"]).reset_index(drop=True)
    out_df = pd.concat([base_df, tt_df.reset_index(drop=True)], axis=1)

    # Prepare output path
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = "sample_df.csv"
    out_csv = out_dir / filename

    out_df.to_csv(out_csv, index=False)
    return str(out_csv.resolve())


if __name__ == "__main__":
    main()