import json
import tempfile
import unittest
from pathlib import Path

from tools.check_metric_thresholds import METRICS, evaluate_results, expand_result_paths, load_result


def result(dataset, pa, seed, values):
    return {
        'dataset': dataset,
        'pa': pa,
        'seed': seed,
        'metrics': dict(zip(METRICS, values)),
        'path': f'{dataset}-{pa}-{seed}.json',
    }


class MetricThresholdTest(unittest.TestCase):
    def test_three_distinct_seeds_above_every_threshold_pass(self):
        rows = [result('wildtrack', 45, seed, (80.0, 74.0, 94.0, 86.0)) for seed in (1, 2, 3)]
        summary = evaluate_results(rows)[0]
        self.assertTrue(summary['enough_seeds'])
        self.assertTrue(summary['passed'])

    def test_threshold_is_strict_and_equal_value_fails(self):
        rows = [result('multiviewx', 0, seed, (82.4, 80.0, 98.0, 85.0)) for seed in (1, 2, 3)]
        summary = evaluate_results(rows)[0]
        self.assertFalse(summary['metric_pass']['moda_percent'])
        self.assertFalse(summary['passed'])

    def test_duplicate_or_too_few_seeds_fail(self):
        rows = [
            result('wildtrack', 20, 1, (90.0, 80.0, 99.0, 95.0)),
            result('wildtrack', 20, 1, (90.0, 80.0, 99.0, 95.0)),
            result('wildtrack', 20, 2, (90.0, 80.0, 99.0, 95.0)),
        ]
        summary = evaluate_results(rows)[0]
        self.assertEqual(summary['duplicate_seeds'], [1])
        self.assertFalse(summary['enough_seeds'])
        self.assertFalse(summary['passed'])

    def test_loads_main_result_schema(self):
        payload = {
            'config': {'dataset': 'wildtrack', 'pa': 60, 'seed': 3},
            'metrics': dict(zip(METRICS, (81, 75, 95, 87))),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final_metrics.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            loaded = load_result(path)
        self.assertEqual((loaded['dataset'], loaded['pa'], loaded['seed']), ('wildtrack', 60, 3))

    def test_expands_recursive_glob_for_windows_shells(self):
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / 'seed1'
            nested.mkdir()
            expected = nested / 'final_metrics.json'
            expected.write_text('{}', encoding='utf-8')
            paths = expand_result_paths([str(Path(directory) / '**' / 'final_metrics.json')])
        self.assertEqual(paths, [expected])


if __name__ == '__main__':
    unittest.main()
