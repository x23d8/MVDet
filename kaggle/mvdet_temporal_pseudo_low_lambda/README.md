# MVDet temporal pseudo-label grid

This private Kaggle notebook runs the original `PerspTransDetector` with the
`pseudo_only` BRL objective. It generates YOLO26x detection-only BEV labels
once, using each person box's bottom-center point as the ground footprint,
camera-aware clustering, and temporal singleton recovery. It then trains the
low pseudo-loss grid `0.005, 0.01, 0.025` for seven epochs per setting.

Python and MVDet logging are unbuffered so the kernel output contains live
`Train Epoch`, `Testing`, and `moda/modp` lines. Logs, weights, curves, the
pseudo-generation summary, and `run_manifest.json` are copied into Kaggle
outputs and bundled as `mvdet_temporal_low_lambda_results.zip`.

Required inputs:

- `/kaggle/input/thesis-dataset-new/Wildtrack/Wildtrack`

The notebook automatically downloads the detection model `yolo26x.pt` with
Ultralytics into `/kaggle/working`. It does not load a pose model or use ankle
keypoints. Inference is restricted to class 0 (`person`).

Source code is cloned at runtime from branch `pseudolabelloss` of
`https://github.com/x23d8/MVDet.git`; Kaggle Internet must be enabled.
