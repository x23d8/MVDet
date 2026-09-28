# MVDet temporal pseudo-label grid

This private Kaggle script runs the original `PerspTransDetector` with the
`pseudo_only` BRL objective. It generates YOLO26x-pose BEV labels once, using
camera-aware clustering and temporal singleton recovery, then trains the low
pseudo-loss grid `0.005, 0.01, 0.025` for seven epochs per setting.

Python and MVDet logging are unbuffered so the kernel output contains live
`Train Epoch`, `Testing`, and `moda/modp` lines. Logs, weights, curves, the
pseudo-generation summary, and `run_manifest.json` are copied into Kaggle
outputs and bundled as `mvdet_temporal_low_lambda_results.zip`.
