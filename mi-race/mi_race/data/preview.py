from __future__ import annotations
from pathlib import Path
import pandas as pd
from tabulate import tabulate

def count_rows(csv_path: Path) -> int:
    with csv_path.open('r', encoding='utf-8', errors='ignore') as f:
        return sum(1 for _ in f) - 1
    

def count_class(csv_path: Path, label: str) -> int:
    uniq = set()
    for chunk in pd.read_csv(csv_path, usecols=[label], chunksize=10000):
        uniq.update(chunk[label].dropna().unique())
    return len(uniq)


def preview_csv(csv_path: str, label: str | None = None, nrows=5) -> None:
    p = Path(csv_path)
    if not p.exists():
        raise SystemExit(f'[mi-race] file not found: {p.resolve()}')
    
    head = pd.read_csv(p, nrows=nrows)

    print(" === Preview (first rows) ===\n")
    print(tabulate(head, headers='keys', tablefmt='psql', showindex=False))
    print()

    total = count_rows(p)

    if label: 
        try:
            classes = count_class(p, label)
            print(f"Total rows: {total:,} • Classes ({label}): {classes}")
        except ValueError:
            print(f"Total rows: {total:,} • (label '{label}' not found)")
    else:
        print(f"Total rows: {total:,}")