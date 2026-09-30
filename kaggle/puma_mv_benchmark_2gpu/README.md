# PUMA-MV Kaggle benchmark

This notebook runs the PUMA-MV validation protocol on Wildtrack with two
GPUs. Its default `validation_scar` mode trains on 0--80%, evaluates 80--90%, saves
raw score maps, and selects the classification threshold and metric NMS radius
offline. It does not touch the 90--100% test split.

Before pushing the notebook, commit and push the PUMA implementation to the
configured GitHub branch. Then edit `REPO_URL` and `BRANCH` in the first code
cell if necessary. The repository default is the `puma` branch.

`DATASET_ROOT` defaults to
`/kaggle/input/datasets/lee735/thesis-dataset-new/Wildtrack/Wildtrack`.
Set it to an empty string to auto-discover Wildtrack below `/kaggle/input`.

Modes:

- `smoke`: one real seven-camera AMP forward/backward step;
- `validation_scar`: validation run with uniformly missing instances;
- `validation_sar`: validation run with visibility/scale-biased missingness;
- `final_scar`: final PUMA run after validation hyperparameters are frozen;
- `final_sar`: final learned-propensity SAR run;
- `baselines`: the complete matched-protocol baseline matrix.

The notebook validates that all seven camera folders and annotations are
complete. It creates partial annotations and evaluator GT under
`/kaggle/working`, because Kaggle input datasets are read-only. If a session
stops, attach the previous result directory and set `RESUME_SOURCE`; the full
optimizer, scheduler, AMP scaler, history, and RNG state are restored.

Select two T4 GPUs in Kaggle. `BATCH_SIZE=2` is a global DataParallel batch,
therefore each GPU receives one synchronized sample.

`PARALLEL_VIEW_ENCODING=False` is the safe default. Each GPU still trains one
sample, but its seven cameras pass through the shared encoder sequentially.
This avoids the large cuDNN `FIND` workspace that can produce an illegal CUDA
access on T4 with the 32/64-channel model. The preflight smoke test uses the
same channel widths and full-view/camera-drop two-pass pattern as training.
