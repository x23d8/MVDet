from pathlib import Path
import runpy


MODULE = runpy.run_path(str(
    Path(__file__).parents[1] / "tools" / "summarize_puma_runs.py"
))


def _write_log(path, seed, moda):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "Settings:\n"
        + repr({
            "dataset": "wildtrack",
            "drop_ratio": 60,
            "loss": "pu",
            "pu_propensity_mode": "scar",
            "variant": "puma_hybrid",
            "seed": seed,
        })
        + f"\nmoda: {moda}%, modp: 80.0%, precision: 90.0%, recall: 85.0%\n",
        encoding="utf-8",
    )


def test_log_parser_and_seed_aggregation(tmp_path):
    first = tmp_path / "one" / "log.txt"
    second = tmp_path / "two" / "log.txt"
    _write_log(first, 1, 70.0)
    _write_log(second, 2, 74.0)
    records = [MODULE["parse_log"](first), MODULE["parse_log"](second)]
    rows = MODULE["aggregate"](records)
    assert len(rows) == 1
    assert rows[0]["runs"] == 2
    assert rows[0]["moda_mean"] == 72.0
    assert rows[0]["moda_std"] > 0
