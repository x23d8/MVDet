# Original MVDet training with YOLO-pose BEV pseudo labels

The training entrypoint is `MVDetBRL/main.py`. The model is the original
`PerspTransDetector`. Each grid script automatically prepares the offline YOLO
pseudo-label set before starting training.

## Quick run

```bash
cd MVDetBRL
bash train_grid_pseudo_only.sh
# or
bash train_grid_pseudo_confuse.sh
```

On startup, the script validates paths and hyperparameters. If fewer than 360
pseudo JSON files exist, it reuses a local `yolo26m-pose.pt` when available or
lets Ultralytics download the model, then generates the missing pseudo labels.
YOLO is not run again during MVDet epochs or for every lambda value.

Set `AUTO_PREPARE_PSEUDO=0` to require an existing pseudo set, or
`FORCE_REGENERATE_PSEUDO=1` to rebuild it.

For a short pipeline smoke test:

```bash
LAMBDAS="0.1" SEEDS="1" EPOCHS=1 MIN_PSEUDO_FILES=5 \
bash train_grid_pseudo_only.sh
```

The script validates dataset folders, dropped-annotation folders, integer
settings, probability thresholds, optimizer values, and requires every pseudo
lambda to satisfy `0 < lambda < 1` before starting YOLO or MVDet.

## Manual pseudo-label generation

Generate the full training split from the repository's `SHOTBRL` utilities:

```bash
cd SHOTBRL
python tools/generate_yolo_pose_bev_pseudo.py \
  --dataset wildtrack \
  --data-root ../Data/Wildtrack \
  --output ../MVDetBRL/pseudo_labels/wildtrack_yolo26m \
  --model yolo26m-pose.pt \
  --imgsz 1280 \
  --foot-anchor pose_x_bbox_y \
  --min-views 2 \
  --merge-radius-m 0.75 \
  --device 0
```

Then run either grid from `MVDetBRL`:

```bash
cd ../MVDetBRL
bash train_grid_pseudo_only.sh
bash train_grid_pseudo_confuse.sh
```

Defaults are relative to `MVDetBRL`: dataset `../Data/Wildtrack`, pseudo labels
`./pseudo_labels/wildtrack_yolo26m`, drop ratio `45`, seed `1`, ten epochs, and
lambda grid `0.025 0.05 0.1 0.2 0.4`.

Override them without editing the scripts:

```bash
DATA_PATH=../datasets/Wildtrack \
PSEUDO_DIR=./pseudo_labels/wildtrack_yolo26m \
LAMBDAS="0.05 0.1 0.2" \
SEEDS="1 2 3" \
EPOCHS=20 \
bash train_grid_pseudo_only.sh
```

Loss modes:

- `pseudo_only`: GT positive + confidence-weighted pseudo positive + background;
  prediction-derived confuse handling is disabled.
- `pseudo_confuse`: the same external pseudo supervision plus conservative BRL
  self-confuse handling (`--brl_no_mirror`).

The pseudo loss is applied only to the BEV output. Original per-camera MVDet
supervision remains in the total loss. Validation/test uses complete GT with no
pseudo labels and no dropped annotations.
