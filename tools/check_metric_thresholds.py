#!/usr/bin/env python3
"""Aggregate MVDet result JSON files and enforce the requested metric gates."""

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    'moda_percent',
    'modp_percent',
    'detection_precision_percent',
    'detection_recall_percent',
)

THRESHOLDS = {
    ('wildtrack', 'drop'): (79.9, 73.7, 93.9, 85.5),
    ('wildtrack', 'perfect'): (88.0, 74.7, 93.2, 95.0),
    ('multiviewx', 'drop'): (66.5, 78.2, 98.5, 67.5),
    ('multiviewx', 'perfect'): (82.4, 79.0, 97.7, 84.4),
}


def expand_result_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise ValueError(f'no result files match: {pattern}')
        paths.extend(Path(match) for match in matches)
    return paths


def load_result(path):
    path = Path(path)
    with path.open('r', encoding='utf-8') as result_file:
        payload = json.load(result_file)
    config = payload.get('config', {})
    metrics = payload.get('metrics', {})
    dataset = str(config.get('dataset', '')).lower()
    if dataset not in {'wildtrack', 'multiviewx'}:
        raise ValueError(f'{path}: missing or unsupported config.dataset')
    try:
        partial_percent = int(config['pa'])
        seed = int(config['seed'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'{path}: config.pa and config.seed must be integers') from exc
    if not 0 <= partial_percent < 100:
        raise ValueError(f'{path}: config.pa must be in [0, 100)')

    parsed_metrics = {}
    for name in METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{path}: missing numeric metrics.{name}') from exc
        if not math.isfinite(value):
            raise ValueError(f'{path}: metrics.{name} must be finite')
        parsed_metrics[name] = value
    return {
        'path': str(path.resolve()),
        'dataset': dataset,
        'pa': partial_percent,
        'seed': seed,
        'metrics': parsed_metrics,
    }


def evaluate_results(results, min_seeds=3):
    if min_seeds < 1:
        raise ValueError('min_seeds must be positive')
    groups = defaultdict(list)
    for result in results:
        groups[(result['dataset'], result['pa'])].append(result)

    summaries = []
    for (dataset, partial_percent), entries in sorted(groups.items()):
        seeds = [entry['seed'] for entry in entries]
        duplicate_seeds = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        mode = 'perfect' if partial_percent == 0 else 'drop'
        thresholds = dict(zip(METRICS, THRESHOLDS[(dataset, mode)]))
        means = {
            name: statistics.fmean(entry['metrics'][name] for entry in entries)
            for name in METRICS
        }
        stddev = {
            name: statistics.pstdev(entry['metrics'][name] for entry in entries)
            for name in METRICS
        }
        metric_pass = {name: means[name] > thresholds[name] for name in METRICS}
        enough_seeds = len(set(seeds)) >= min_seeds and not duplicate_seeds
        summaries.append({
            'dataset': dataset,
            'pa': partial_percent,
            'mode': mode,
            'seeds': sorted(seeds),
            'duplicate_seeds': duplicate_seeds,
            'minimum_distinct_seeds': min_seeds,
            'enough_seeds': enough_seeds,
            'mean': means,
            'population_stddev': stddev,
            'thresholds_strictly_greater_than': thresholds,
            'metric_pass': metric_pass,
            'passed': enough_seeds and all(metric_pass.values()),
        })
    return summaries


def print_summary(summaries):
    print('| dataset | pa | seeds | MODA | MODP | precision | recall | gate |')
    print('|---|---:|---|---:|---:|---:|---:|:---:|')
    for item in summaries:
        mean = item['mean']
        marker = 'PASS' if item['passed'] else 'FAIL'
        print(
            f"| {item['dataset']} | {item['pa']} | {item['seeds']} | "
            f"{mean['moda_percent']:.2f} | {mean['modp_percent']:.2f} | "
            f"{mean['detection_precision_percent']:.2f} | "
            f"{mean['detection_recall_percent']:.2f} | {marker} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description='Check strict Wildtrack/MultiviewX metric gates over distinct seeds.'
    )
    parser.add_argument('results', nargs='+', help='final_metrics.json files or glob patterns')
    parser.add_argument('--min-seeds', type=int, default=3)
    parser.add_argument('--output', type=Path, default=None, help='optional aggregate JSON path')
    args = parser.parse_args()

    try:
        paths = expand_result_paths(args.results)
        summaries = evaluate_results([load_result(path) for path in paths], args.min_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if not summaries:
        raise SystemExit('no result groups were provided')
    print_summary(summaries)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8') as output_file:
            json.dump(summaries, output_file, indent=2, sort_keys=True)
    raise SystemExit(0 if all(item['passed'] for item in summaries) else 1)


if __name__ == '__main__':
    main()
