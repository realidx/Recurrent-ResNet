#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine many small CSV result files into one CSV, optionally aggregating mean±std."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing CSV files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern within input-dir (default: *.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="config",
        help="Column to sort by when not aggregating (default: config).",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate rows by group columns and write mean±std for numeric columns.",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default="config",
        help="Comma-separated group columns for --aggregate (default: config).",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def to_float(x: str | None) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s in {"N/A"} or s.lower() == "nan":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"--input-dir must be an existing directory: {input_dir}")

    pattern = str(input_dir / args.pattern)
    paths = [Path(p) for p in glob.glob(pattern)]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"No CSV files found for pattern: {pattern}")
    paths = sorted(paths, key=lambda p: p.name)

    all_fieldnames: list[str] = []
    all_rows: list[dict[str, str]] = []

    for path in paths:
        fieldnames, rows = read_csv_rows(path)
        if not all_fieldnames:
            all_fieldnames = fieldnames
        else:
            for f in fieldnames:
                if f not in all_fieldnames:
                    all_fieldnames.append(f)
        all_rows.extend(rows)

    # Normalize to union header.
    normalized = [{k: r.get(k, "") for k in all_fieldnames} for r in all_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not args.aggregate:
        sort_by = (args.sort_by or "").strip()
        if sort_by:
            if sort_by not in all_fieldnames:
                raise SystemExit(f"--sort-by column not found: {sort_by}")
            normalized.sort(key=lambda r: (r.get(sort_by, ""), r.get("seed", "")))

        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        with tmp_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
        os.replace(tmp_path, args.output)
        print(f"Wrote {len(normalized)} rows to {args.output}")
        return

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    for c in group_cols:
        if c not in all_fieldnames:
            raise SystemExit(f"--group-cols column not found: {c}")

    # Determine numeric columns by attempting to parse at least one value.
    numeric_cols: list[str] = []
    for col in all_fieldnames:
        if col in group_cols:
            continue
        if any(to_float(r.get(col)) is not None for r in normalized):
            numeric_cols.append(col)

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in normalized:
        key = tuple(r.get(c, "") for c in group_cols)
        groups[key].append(r)

    out_fieldnames = group_cols + ["n_rows"]
    for col in numeric_cols:
        out_fieldnames.append(f"{col}_mean")
        out_fieldnames.append(f"{col}_std")

    out_rows: list[dict[str, str]] = []
    for key in sorted(groups.keys()):
        members = groups[key]
        out: dict[str, str] = {c: v for c, v in zip(group_cols, key)}
        out["n_rows"] = str(len(members))
        for col in numeric_cols:
            vals = [to_float(m.get(col)) for m in members]
            vals = [v for v in vals if v is not None]
            if not vals:
                out[f"{col}_mean"] = ""
                out[f"{col}_std"] = ""
            elif len(vals) == 1:
                out[f"{col}_mean"] = f"{vals[0]:.6g}"
                out[f"{col}_std"] = "0"
            else:
                out[f"{col}_mean"] = f"{statistics.mean(vals):.6g}"
                out[f"{col}_std"] = f"{statistics.stdev(vals):.6g}"
        out_rows.append(out)

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp_path, args.output)
    print(f"Wrote {len(out_rows)} aggregated rows to {args.output}")


if __name__ == "__main__":
    main()

