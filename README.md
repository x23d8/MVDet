# Multiview Detection with Feature Perspective Transformation [[Website](https://hou-yz.github.io/publication/2020-eccv2020-mvdet)] [[arXiv](https://arxiv.org/abs/2007.07247)]

```
@inproceedings{hou2020multiview,
  title={Multiview Detection with Feature Perspective Transformation},
  author={Hou, Yunzhong and Zheng, Liang and Gould, Stephen},
  booktitle={ECCV},
  year={2020}
}
```

Please visit [link](https://github.com/hou-yz/MVDeTr) for our new work MVDeTr, a transformer-powered multiview detector that achieves new state-of-the-art!

## Overview
We release the PyTorch code for **MVDet**, a state-of-the-art multiview pedestrian detector; and **MultiviewX** dataset, a novel synthetic multiview pedestrian detection datatset.

Wildtrack             |  MultiviewX
:-------------------------:|:-------------------------:
![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_wildtrack_demo.gif "Detection results on Wildtrack dataset")  |  ![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_demo.gif "Detection results on MultiviewX dataset")

 
## Content
- [MultiviewX dataset](#multiviewx-dataset)
    * [Download MultiviewX](#download-multiviewx)
    * [Build your own version](#build-your-own-version)
- [MVDet Code](#mvdet-code)
    * [Dependencies](#dependencies)
    * [Data Preparation](#data-preparation)
    * [Training](#training)



## MultiviewX dataset
Using pedestrian models from [PersonX](https://github.com/sxzrt/Dissecting-Person-Re-ID-from-the-Viewpoint-of-Viewpoint), in Unity, we build a novel synthetic dataset **MultiviewX**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_multiviewx_dataset.jpg "Visualization of MultiviewX dataset")

MultiviewX dataset covers a square of 16 meters by 25 meters. We quantize the ground plane into a 640x1000 grid. There are 6 cameras with overlapping field-of-view in MultiviewX dataset, each of which outputs a 1080x1920 resolution image. We also generate annotations for 400 frames in MultiviewX at 2 fps (same as Wildtrack). On average, 4.41 cameras are covering the same location. 

### Download MultiviewX
Please refer to this [link](https://1drv.ms/u/s!AtzsQybTubHfhYZ9Ghhahbp20OX9kA?e=Hm9Xdg) for download.

### Build your own version
Please refer to this [repo](https://github.com/hou-yz/MultiviewX) for a detailed guide & toolkits you might need.




## MVDet Code
This repo is dedicated to the code for **MVDet**. 

![alt text](https://hou-yz.github.io/images/eccv2020_mvdet_architecture.png "Architecture for MVDet")

### Dependencies
This code uses the following libraries
- python 3.7+
- pytorch 1.4+ & tochvision
- numpy
- matplotlib
- pillow
- opencv-python
- kornia
- wandb & python-dotenv
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)

### Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](#multiviewx-dataset) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

On Kaggle, the dataset classes also automatically discover datasets mounted anywhere under
`/kaggle/input` or `/kaggle/working`, so the paths in the training scripts do not need to be changed.
The local `~/Data/` path is still preferred when it exists.
Generated ground-truth files for read-only datasets under `/kaggle/input` are written to
`/kaggle/working/mvdet_cache`. Set `MVDET_CACHE_DIR` to use a different cache directory.

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

### Training
The training entrypoint logs config, train/validation metrics, timings, and GPU usage to
the `GFA26AI02/baseline-expriments` Weights & Biases project. Put either
`WANDB_API_KEY=...` (preferred) or `WANDB_API=...` in the repository `.env` file.

Run one command per dataset; each command creates a separate, comparable W&B run:
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
CUDA_VISIBLE_DEVICES=0,1 python main.py -d multiviewx
``` 

For a custom dataset location, pass the full dataset root (or a parent that
contains it). The dataset type is detected automatically when the path contains
only one supported dataset:

```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path /path/to/MultiviewX
CUDA_VISIBLE_DEVICES=0,1 python main.py --data_path /path/to/Wildtrack
```

If a parent contains both datasets, select one explicitly with `-d wildtrack`
or `-d multiviewx`. Kaggle inputs remain read-only; generated `gt.txt` is stored
under `/kaggle/working/mvdet_cache` as before.

When the complete and dropped datasets are mounted as separate Kaggle inputs,
use `--data_path` for the complete dataset containing images and calibration,
and `--dropped_path` for the separate partial-annotation input. The dropped
path may be a parent, a `<Dataset>_dropped` directory, or a selected `drop<pa>`
directory; repeated Kaggle wrapper directories are handled automatically:

```shell script
python main.py -d multiviewx \
  --dropped_path /kaggle/input/thesis-dataset/MultiviewX_dropped \
  --pa 45
```

With `-d multiviewx`, the complete dataset is located automatically in the
default location or Kaggle inputs. Use `--data_path` as an optional override
when more than one compatible complete dataset is mounted.

To train with dropped annotations, keep each generated dropped directory next
to its complete dataset and select the percentage with `--pa`:

```text
Wildtrack_dataset/
|-- Wildtrack/
`-- Wildtrack_dropped/
    |-- drop20/
    |-- drop45/
    `-- drop60/
```

```shell script
python main.py -d wildtrack --data_path /path/to/Wildtrack_dataset --pa 0
python main.py -d wildtrack --data_path /path/to/Wildtrack_dataset --pa 20
python main.py -d wildtrack --data_path /path/to/Wildtrack_dataset --pa 45
python main.py -d wildtrack --data_path /path/to/Wildtrack_dataset --pa 60
```

`--pa 0` preserves the original full-annotation training behavior. For nonzero
settings, only training targets come from
`<Dataset>_dropped/drop<pa>/annotations_positions`; images, calibrations,
validation labels, test labels, and evaluation ground truth remain from the
complete dataset. Files in `hidden_annotations_positions` are deliberately
excluded from the baseline loss. The supported settings are `0`, `20`, `45`,
and `60`; `tools/simulate_dropped_anotations.py` generates the corresponding
`drop20`, `drop45`, and `drop60` directories. Local outputs are separated under
`logs/<dataset>_frame/<variant>/pa<pa>/` so concurrently launched settings do
not share a checkpoint directory.

### Partial-annotation-aware loss

`--loss auto` is the default. It keeps the max-Gaussian MSE baseline for
`--pa 0` and selects BEV-BRL plus partial-aware head/foot supervision whenever
`--pa` is nonzero. Explicitly selecting `--loss gaussian_mse` with partial
annotations is rejected because dense MSE would train every dropped pedestrian
as background.

```shell script
python main.py -d wildtrack --data_path /path/to/Wildtrack_dataset --pa 45 --loss auto
```

For a partial run, the total objective is the BEV-BRL loss plus `--alpha`
times the camera-view loss. Observed head/foot points use bounded max-Gaussian
targets. Unlabelled head pixels are ignored by default. Unlabelled foot points
become pseudo positives only after warm-up and only when fused BEV confidence
and leave-one-view-out camera consensus agree. Reliable foot negatives also
require valid multi-camera ground-plane coverage. Confidence-derived masks are
detached from autograd.

After warm-up, low external consensus selects the complete reliable-negative
set; current prediction confidence only separates its hard subset. This avoids
the previous gap where cells above the easy-negative cutoff but below the
mirror threshold received no negative gradient despite crossing the inference
threshold. BEV hard negatives default to confidence `0.40`, while mirror
positives retain the stricter `0.60` threshold. The analogous foot thresholds
are also separate. The default schedule uses one warm-up epoch and a two-epoch
ramp for a ten-epoch run. The partial-label head weight defaults to `0.05` and
its unlabelled negative weight remains zero.

The default and `no_joint_conv` variants continue to project their C-channel
backbone features unchanged. In `res_proj`, foot logits are converted to
probabilities before projection. `img_proj` has no learned camera head, so its
view loss and view consensus are disabled.

Useful controls include `--brl_warmup_epochs`, `--brl_ramp_epochs`,
`--brl_hard_negative_threshold`, `--brl_mirror_threshold`,
`--brl_hard_negative_weight`, `--view_hard_negative_weight`,
`--brl_min_views`, `--brl_consensus_topk`, `--view_head_weight`, and
`--view_foot_weight`. Keep `--view_head_negative_weight 0` unless an independent
teacher or calibrated head-height model is added.

### Two-epoch local diagnostic

The `local_2ep` profile provides a bounded single-GPU check of the partial loss:

```powershell
python main.py --run_profile local_2ep -d wildtrack `
  --data_path D:\path\to\Wildtrack `
  --dropped_path D:\MVDet\Wildtrack_dropped `
  --pa 45 --loss auto
```

The profile uses two epochs, 64 evenly spaced training frames, 20 evenly spaced
validation frames, one worker, input size `360x640`, image reduction `8`, and
ground-grid reduction `8`. It uses a learning rate of `0.005` with gradient
norm clipping at `1.0`, because the logit-space positive objective has a much
stronger low-confidence gradient than probability MSE. Hard evidence-negative
peaks receive additional BEV and foot-view weights of `2.0` and `1.0`,
respectively, so they are not diluted by easy negatives. W&B logging is
disabled. It skips the redundant
evaluation before training and after the last epoch. Validation still runs
after each epoch against complete annotations for exactly the selected
validation frames. A single validation forward pass is evaluated at confidence
thresholds from `0.30` to `0.90`, with denser `0.02` steps around `0.40–0.50`;
this adds only post-processing and does not
change the model or loss. Override the diagnostic sweep with, for example,
`--eval_thresholds 0.4 0.6 0.8`. Outputs are written below
`logs/wildtrack_frame/default/pa45/local_2ep/`.
After epoch 2, `local_2ep_diagnostic.json` records the complete validation
history, the fixed-threshold result at `--cls_thres`, and every swept result.
It prints `PASS` only when the best validation MODA across the predeclared
thresholds is greater than zero and both the BEV and projected foot-view
evidence-negative paths were active. The report also marks cases where MODA is
positive only after threshold calibration while the fixed `0.4` result remains
zero.

The local profile does not import or require the `wandb` package. Metrics are
written to `log.txt` and the diagnostic JSON instead.

This profile is a diagnostic and its metrics are not comparable with the full
benchmark configuration. A useful run should show nonzero
`bev/evidence_negative_cells` and `bev/hard_negative_cells`, fewer uncertain
cells, and improving detection precision. Use the full profile for reported
results.

This should automatically return evaluation results similar to the reported 88.2\% MODA on Wildtrack dataset. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRE9Iy8IjsGMXB17A?e=CCqhIQ).
