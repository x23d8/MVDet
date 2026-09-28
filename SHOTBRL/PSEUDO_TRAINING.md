# YOLO-pose pseudo supervision for original MVDet

The YOLO model runs offline. Training reads per-frame JSON files and does not
rerun YOLO every epoch. The grid scripts now automate this preparation: when
pseudo labels are incomplete, they reuse/download `yolo26m-pose.pt` and run the
generator once before training. YOLO utilities live in `SHOTBRL`, while all
training commands run `MVDetBRL/main.py` with the original
`PerspTransDetector` architecture.

## 1. Inspect five C1 frames

```powershell
cd SHOTBRL
python tools\yolo_pose_foot_demo.py `
  --source "..\Data\Wildtrack\Image_subsets\C1" `
  --output "artifacts\yolo26m_pose_c1_demo" `
  --model yolo26m-pose.pt `
  --num-frames 5 `
  --imgsz 1280 `
  --device 0
```

Observed on frames 0--20 with the current defaults:

- Detection precision: 0.7875 at IoU 0.30.
- Detection recall: 0.3938.
- Median matched foot error: 20.0 image pixels.

## 2. Generate multi-view BEV pseudo labels

```powershell
python tools\generate_yolo_pose_bev_pseudo.py `
  --dataset wildtrack `
  --data-root "..\Data\Wildtrack" `
  --output "..\MVDetBRL\pseudo_labels\wildtrack_yolo26m" `
  --model yolo26m-pose.pt `
  --imgsz 1280 `
  --foot-anchor pose_x_bbox_y `
  --min-views 2 `
  --merge-radius-m 0.75 `
  --device 0
```

`pose_x_bbox_y` uses the pose-estimated horizontal foot location and the bbox
bottom as the ground-contact vertical coordinate. This matches MVDet's original
foot-label convention better than projecting the ankle midpoint directly.

Five-frame BEV checks at a 0.5 m matching radius:

| Anchor | Precision | Recall | Median error |
|---|---:|---:|---:|
| ankle/pose point | 0.459 | 0.189 | 0.319 m |
| pose x + bbox bottom y | 0.847 | 0.400 | 0.181 m |
| bbox bottom center | 0.857 | 0.400 | 0.174 m |

Use `--evaluate-gt` only for development checks. It does not alter or filter
pseudo labels, but full GT metrics should not be used to tune the test split.

## 3. Train either loss option

Move to the MVDet project before training:

```powershell
cd ..\MVDetBRL
```

External pseudo only, without self-confuse:

```powershell
python main.py -d wildtrack `
  --variant default `
  --data_path "..\Data\Wildtrack" `
  --loss brl `
  --drop_ratio 45 `
  --pseudo_mode pseudo_only `
  --pseudo_dir "pseudo_labels\wildtrack_yolo26m" `
  --lambda_pseudo 0.1 `
  --brl_no_mirror
```

External pseudo plus BRL self-confuse:

```powershell
python main.py -d wildtrack `
  --variant default `
  --data_path "..\Data\Wildtrack" `
  --loss brl `
  --drop_ratio 45 `
  --pseudo_mode pseudo_confuse `
  --pseudo_dir "pseudo_labels\wildtrack_yolo26m" `
  --lambda_pseudo 0.1 `
  --brl_confuse_thr 0.3 `
  --brl_beta 0.1 `
  --brl_no_mirror
```

`--brl_no_mirror` is recommended for `pseudo_confuse`: externally confirmed
pseudo points are positive, while unsupported high MVDet predictions merely get
a reduced negative penalty.

## 4. Lambda grid search

Two independent Bash entrypoints are provided so the experiment modes cannot
be mixed accidentally.

Option 1 -- external pseudo supervision, no self-confuse:

```bash
bash train_grid_pseudo_only.sh
```

Option 2 -- external pseudo supervision plus conservative self-confuse:

```bash
bash train_grid_pseudo_confuse.sh
```

Both scripts first change to the `MVDetBRL` directory, then use these defaults:
`../Data/Wildtrack` for the dataset and
`./pseudo_labels/wildtrack_yolo26m` for pseudo labels. Override them through
environment variables when a remote or Kaggle dataset is mounted elsewhere:

```bash
LAMBDAS="0.05 0.1 0.2" \
SEEDS="1 2 3" \
DROP_RATIOS="20 45 60" \
EPOCHS=20 \
GPU=0 \
bash train_grid_pseudo_only.sh
```

The grid scripts expect at least 360 pseudo JSON files, matching the default
90% training split for Wildtrack and MultiviewX. If the set is incomplete,
`AUTO_PREPARE_PSEUDO=1` downloads/reuses YOLO and generates the missing files
before training. For a deliberate short smoke test only, override the guard
with `MIN_PSEUDO_FILES=<count>`.

Arguments after the script name are forwarded to `main.py`. Both scripts pass
`--variant default` explicitly; in `MVDetBRL` this is the original
`PerspTransDetector`. By default they tune on drop ratio 45, one seed, and
lambdas `0.025 0.05 0.1 0.2 0.4`.

The two scripts enforce the intended loss logic:

- `pseudo_only`: `GT + weighted pseudo + background`; self-confuse is disabled.
- `pseudo_confuse`: `GT + weighted pseudo + down-weighted self-confuse + easy
  background`; `--brl_no_mirror` prevents unsupported predictions from being
  promoted to positive.

The external pseudo region is removed from both confuse and background masks in
both cases. Test data uses full GT and never loads pseudo labels.

### PowerShell alternative

```powershell
.\run_pseudo_grid.ps1 `
  -DataPath "..\Data\Wildtrack" `
  -PseudoDir ".\pseudo_labels\wildtrack_yolo26m" `
  -DropRatio 45 `
  -Lambdas 0.025,0.05,0.1,0.2,0.4 `
  -Seeds 1
```

After the coarse search, rerun nearby lambda values and seeds 1, 2, and 3.
Select by validation MODA rather than raw training-loss magnitude.
