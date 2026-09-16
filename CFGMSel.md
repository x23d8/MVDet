# cfgmsel: MVDet, SHOTBRL, MVDeTr with partial annotations

## Dataset layout

Run `simulate_dropped_anotations.py` on the complete dataset first. The
generator keeps images and calibration untouched and writes JSON files here:

```text
Wildtrack/                         # or MultiviewX/
  Image_subsets/
  calibrations/
  annotations_positions/           # complete labels, used for validation GT
  drop_annotations/
    drop_20/
      annotations_positions/       # observed training labels
      hidden_annotations_positions/ # never passed to the trainer
    drop_45/
    drop_60/
```

The split stays 90% training frames and 10% validation frames. `--pa 0` uses
complete labels. `--pa 20`, `45`, or `60` reads only the matching observed
training folder. All three models evaluate against complete labels. On Kaggle,
generated `gt.txt` is written under `/kaggle/working/mvdet_cache`; the input
mount is not modified. `MVDET_CACHE_DIR` overrides that directory.

## Commands

The root `main.py` chooses a runner before importing any model package. Use
`--data_path` for the complete dataset root or a Kaggle input wrapper. The
dataset name is explicit with `-d` if a wrapper contains both datasets.

```bash
python simulate_dropped_anotations.py -d Wildtrack -r /path/to/parent -s drop20 drop45 drop60 --seed 1

python main.py --model mvdet -d wildtrack --data_path /path/to/Wildtrack --pa 45 --loss confuse_gaussian
python main.py --model shot -d wildtrack --data_path /path/to/Wildtrack --pa 45 --loss confuse_gaussian
python main.py --model mvdetr -d wildtrack --data_path /path/to/Wildtrack --pa 45 --loss confuse_gaussian --world_feat conv
```

On Kaggle, `/kaggle/input/<slug>/Wildtrack` can be passed as `--data_path`.
When omitted, each runner searches `~/Data/<dataset>` and then `/kaggle/input`.
The MVDeTr `deform_trans` and `aio` feature modes require the native deformable
attention extension; build it from
`MVDeTr/multiview_detector/models/ops/setup.py` in the intended CUDA runtime.
`--world_feat conv` and `trans` can run without that extension. SHOT's default
variant is its depth aware detector; `--depth_scales` controls depth sampling.

### Loss selection

| Runner | `--loss auto`, full labels | `--loss auto`, partial labels | Other choices |
| --- | --- | --- | --- |
| MVDet | GaussianMSE | ConfuseGaussianMSE | `mse`, `confuse_gaussian` |
| SHOTBRL | GaussianMSE | ConfuseGaussianMSE | `mse`, `confuse_gaussian`, legacy `brl` |
| MVDeTr | original focal plus regression | ConfuseGaussianMSE plus regression on observed objects | `focal`, `mse`, `confuse_gaussian` |

The shared ConfuseGaussianMSE controls are `--confuse_pred_thr` (default
`0.3`), `--confuse_beta` (default `0.1`) and `--confuse_no_mirror`.
MVDeTr accepts its older `--brl_*` spellings as aliases. Explicit `--loss`
overrides `auto`. MVDeTr's plain `mse` choice remains heatmap only; its
`confuse_gaussian` choice retains offset and size regression on visible labels.

## ConfuseGaussianMSE

| Part | Purpose | Behavior |
| --- | --- | --- |
| Gaussian target | Smooth observed labels | MVDet/SHOT convolve sparse points with a Gaussian kernel; MVDeTr receives a heatmap already drawn by its dataset. |
| `confuse_pred_thr` | Find possible omitted people | Pixels with detached prediction at or above the threshold enter the confused region. |
| `beta` | Control confused region strength | Multiplies the background part of confused pixels. Zero removes its gradient far from known labels. |
| `mirror=True` | Preserve strong uncertain predictions | The confused background term pulls predictions toward `1`, with strength `beta`. |
| `mirror=False` | Reduce negative supervision | The confused background term remains squared error to the observed target, scaled by `beta`. |
| Known person protection | Keep observed positives supervised | The confused term blends continuously with ordinary MSE according to the soft target. |
| Reduction | Produce one scalar | Mean over all pixels. Assignment to the confused region has no gradient. |

For a pixel with prediction `x`, soft observed target `g`, and
`m = 1[x >= confuse_pred_thr]`, the loss is

`(1-m)(x-g)^2 + m[(1-g) beta (x-t)^2 + g(x-g)^2]`,

where `t=1` with mirror mode and `t=g` otherwise. This is a heuristic for
missing labels: a confident false positive can also receive reduced negative
supervision. Compare it with the full-label and ordinary MSE/focal baselines.

## Verification

`python -m unittest tests.test_dataset_detection tests.test_confuse_gaussian_mse`
checks the embedded drop layout and numerical loss behavior. Model training
requires the actual image datasets and a CUDA environment; no checkpoint or
benchmark result is bundled with this branch.
