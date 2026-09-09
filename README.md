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
excluded from the baseline loss. Any integer drop percentage in `[0, 100)` is
accepted when a matching `drop<percentage>` directory exists;
`tools/simulate_dropped_anotations.py` generates the standard
`drop20`, `drop45`, and `drop60` directories. Local outputs are separated under
`logs/<dataset>_frame/<variant>/pa<pa>/` so concurrently launched settings do
not share a checkpoint directory.

For nonzero `--pa`, `--loss auto` selects the propensity-constrained adaptive
BRL loss. It treats zero target cells as unlabeled, caps pseudo positives from
the declared retained-label probability, and uses geometrically aligned
agreement from at least two cameras. The fully annotated path remains the
original Gaussian MSE by default.

```shell script
python main.py -d wildtrack --data_path /path/to/Wildtrack \
  --dropped_path /path/to/Wildtrack_dropped --pa 45 --loss adaptive_brl
```

See [`docs/partial_annotation_loss_research.md`](docs/partial_annotation_loss_research.md)
for the research synthesis, equation, limitations, and benchmark protocol.

With the complete dataset and the published configuration, the MVDet paper
reports 88.2% MODA on Wildtrack; reproduce and report local results rather than
treating that number as guaranteed.

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRE9Iy8IjsGMXB17A?e=CCqhIQ).
