#!/usr/bin/env python3
"""Aggregate final MVDet/PUMA metrics across seeds from log directories."""

import argparse
import ast
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"moda:\s*([-+\d.]+)%,\s*modp:\s*([-+\d.]+)%,\s*"
    r"precision:\s*([-+\d.]+)%,\s*recall:\s*([-+\d.]+)%",
    re.IGNORECASE,
)
GROUP_FIELDS = (
    "dataset", "drop_ratio", "train_annotation_dir", "loss",
    "train_end_ratio", "eval_start_ratio", "eval_end_ratio",
    "pu_propensity_mode", "variant", "nms_radius_m",
    "puma_feature_channels", "puma_fused_channels", "puma_query_channels",
    "camera_drop_prob", "consistency_loss_weight",
)
METRIC_FIELDS = ("moda", "modp", "precision", "recall")


def parse_log(path):
    settings = None
    metrics = None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "Settings:" and index + 1 < len(lines):
            try:
                settings = ast.literal_eval(lines[index + 1].strip())
            except (SyntaxError, ValueError):
                pass
        match = METRIC_PATTERN.search(line)
        if match:
            metrics = dict(zip(METRIC_FIELDS, map(float, match.groups())))
    if not isinstance(settings, dict) or metrics is None:
        return None
    return {**settings, **metrics, "log_path": str(path)}


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        key = tuple(record.get(field) for field in GROUP_FIELDS)
        groups[key].append(record)
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        row = dict(zip(GROUP_FIELDS, key))
        row["runs"] = len(group)
        row["seeds"] = ",".join(str(item.get("seed", "")) for item in group)
        for metric in METRIC_FIELDS:
            values = [item[metric] for item in group]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path("logs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    logs = sorted(args.root.rglob("log.txt"))
    records = [record for path in logs if (record := parse_log(path)) is not None]
    rows = aggregate(records)
    fieldnames = list(GROUP_FIELDS) + ["runs", "seeds"] + [
        f"{metric}_{suffix}" for metric in METRIC_FIELDS for suffix in ("mean", "std")
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stream = args.output.open("w", newline="", encoding="utf-8")
    else:
        import sys
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    print(
        f"Parsed {len(records)}/{len(logs)} completed logs into {len(rows)} groups",
        file=__import__("sys").stderr,
    )


if __name__ == "__main__":
    main()
