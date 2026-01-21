from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv_columns(path: Path) -> list[str]:
    # Read header only (fast)
    cols = list(pd.read_csv(path, nrows=0).columns)
    if len(cols) != len(set(cols)):
        dupes = sorted({c for c in cols if cols.count(c) > 1})
        raise SystemExit(f"[mi-race] Duplicate column names in CSV: {path} -> {dupes}")
    return cols


def _validate_same_schema(csv_paths: list[Path]) -> list[str]:
    if not csv_paths:
        raise SystemExit("[mi-race] No CSV files to validate")

    ref_cols = _read_csv_columns(csv_paths[0])
    ref_set = set(ref_cols)

    for p in csv_paths[1:]:
        cols = _read_csv_columns(p)
        cols_set = set(cols)

        if len(cols) != len(ref_cols) or cols_set != ref_set:
            missing = sorted(ref_set - cols_set)
            extra = sorted(cols_set - ref_set)
            raise SystemExit(
                "[mi-race] CSV schema mismatch; all files must have the same columns.\n"
                f"Reference: {csv_paths[0]}\n"
                f"Mismatch : {p}\n"
                f"Ref cols ({len(ref_cols)}): {ref_cols}\n"
                f"This cols({len(cols)}): {cols}\n"
                f"Missing: {missing}\n"
                f"Extra  : {extra}"
            )

    return ref_cols


def _csv_paths(folder: Path, *, pattern: str, recursive: bool) -> list[Path]:
    paths = sorted(folder.rglob(pattern) if recursive else folder.glob(pattern))
    return [p for p in paths if p.is_file()]


def _concat_csvs_in_folder(
    folder: Path,
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    auto_recursive: bool = True,
) -> pd.DataFrame:
    """Concatenate CSVs in a folder.

    Behavior:
      - If `recursive=True`, uses rglob.
      - If `recursive=False` and no CSVs found at top-level:
          * when `auto_recursive=True`, automatically falls back to recursive search.
    """
    paths = _csv_paths(folder, pattern=pattern, recursive=recursive)
    if not paths and (not recursive) and auto_recursive:
        # Folder may contain only subfolders (common case: OU/s_0.1/*.csv)
        paths = _csv_paths(folder, pattern=pattern, recursive=True)
        if paths:
            print(f"[mi-race] No top-level CSVs in '{folder}'. Found {len(paths)} CSVs in subfolders; concatenating recursively.")

    if not paths:
        raise SystemExit(f"[mi-race] No CSV files found in folder: {folder} (pattern='{pattern}', recursive={recursive})")

    # Validate that all CSV files have the same schema (same columns)
    ref_cols = _validate_same_schema(paths)

    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p)
        # Allow different column order as long as the set matches
        if list(df.columns) != ref_cols:
            df = df[ref_cols]
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True, sort=False)


def concat_csv_files_from_config(
    config_path: Path,
    *,
    pattern: str = "*.csv",
    recursive: bool = False,
    out_path: Path | None = None,
) -> Path:
    if not config_path.exists():
        raise SystemExit(f"[mi-race] Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)

    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    out_cfg = cfg.get("output", {}) if isinstance(cfg, dict) else {}

    path_raw = data_cfg.get("path")
    if not path_raw:
        raise SystemExit("[mi-race] concat csv-files requires data.path in config")

    p = Path(str(path_raw))
    if not p.exists():
        raise SystemExit(f"[mi-race] data.path does not exist: {p}")

    if p.is_dir():
        df = _concat_csvs_in_folder(p, pattern=pattern, recursive=recursive)
        default_name = f"{p.name}_concat.csv"
    else:
        if p.suffix.lower() != ".csv":
            raise SystemExit(f"[mi-race] data.path must be a .csv file or a folder of csvs: {p}")
        df = pd.read_csv(p)
        default_name = f"{p.stem}_concat.csv"

    if out_path is None:
        # Default: write next to the provided data.path.
        # - folder -> <folder>/<folder>_concat.csv
        # - file   -> <file_dir>/<file_stem>_concat.csv
        out_dir = p if p.is_dir() else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / default_name
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"[mi-race] Wrote: {out_path}")
    print(f"[mi-race] Rows: {len(df):,}  Columns: {len(df.columns):,}")
    return out_path


def run_concat_csv_files(args) -> None:
    cfg_path = Path(getattr(args, "config", "config.json"))
    out = getattr(args, "out", None)
    out_path = Path(out) if out else None
    pattern = getattr(args, "pattern", "*.csv")
    recursive = bool(getattr(args, "recursive", False))
    concat_csv_files_from_config(cfg_path, pattern=pattern, recursive=recursive, out_path=out_path)
