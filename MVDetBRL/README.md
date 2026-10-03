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
- matlab & matlabengine (required for evaluation) (see this [link](/multiview_detector/evaluation/README.md) for detailed guide)

### Data Preparation
By default, all datasets are in `~/Data/`. We use [MultiviewX](#multiviewx-dataset) and [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) in this project. 

Your `~/Data/` folder should look like this
```
Data
├── MultiviewX/
│   └── ...
└── Wildtrack/ 
    └── ...
```

### Training
In order to train classifiers, please run the following,
```shell script
CUDA_VISIBLE_DEVICES=0,1 python main.py -d wildtrack
``` 
This should automatically return evaluation results similar to the reported 88.2\% MODA on Wildtrack dataset. 

### Pre-trained models
You can download the checkpoints at this [link](https://1drv.ms/u/s!AtzsQybTubHfhNRE9Iy8IjsGMXB17A?e=CCqhIQ).

## MVDetBRL (this fork)
Heatmap adaptation of **Background Recalibration Loss** (BRL / Selective-IoU idea) for missing-annotation training.

Default loss is `--loss brl`. Use dropped datasets under `../Data/*_dropped/{drop20,drop45,drop60}`.

```bash
conda activate thesis_env
cd MVDetBRL

# single run
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop60 --loss brl

# ablation: original MSE
CUDA_VISIBLE_DEVICES=0 python main.py -d wildtrack --drop_ratio drop60 --loss mse

# BRL knobs
#   --brl_pos_thr       soft-GT threshold for positives (default 0.1)
#   --brl_confuse_thr   pred threshold on background → confuse (default 0.3)
#   --brl_beta          confuse term weight (default 0.1)
#   --brl_no_mirror     down-weight bg MSE instead of mirroring toward 1

bash run_dropped_annotation.sh
```

### Offline detector pseudo-labels
Install the added `ultralytics` dependency, then create a cache from the original camera images. Detection boxes are stored in original image coordinates.

```bash
python generate_pseudo_cache.py --dataset wildtrack --data_root ~/Data/Wildtrack \
  --output pseudo_cache/wildtrack_yolo26s.json

python evaluate_pseudo_cache.py --dataset wildtrack --data_root ~/Data/Wildtrack \
  --cache pseudo_cache/wildtrack_yolo26s.json

python main.py -d wildtrack --data_path ~/Data/Wildtrack --drop_ratio 60 --loss brl --use_pseudo_labels \
  --pseudo_cache pseudo_cache/wildtrack_yolo26s.json
```

The cache generator defaults to `yolo26s.pt`, COCO person class, and confidence 0.20. The evaluator excludes projections outside the BEV, reports per-camera metrics, and estimates cross-view fusion with a 0.5 m NMS radius. It compares points with full annotations at 0.5 m, 1 m, and 2 m matching radii. Training defaults are pseudo loss weight 0.01, BEV Gaussian sigma 0.5 m, and GT suppression radius 1 m. Training logs report base and weighted pseudo loss separately. The cache is dataset-specific; generate a separate one for MultiviewX. Cache output should be kept outside version control when it contains all detections.

### Gaussian YOLO pseudo supervision

The notebook's Gaussian driver is now available in the source. Select it with
`--pseudo_method gaussian` alongside `--use_pseudo_labels --pseudo_cache ...`.
It uses the maximum Gaussian heatmap from projected YOLO foot points, suppresses
evidence near real annotations, and divides weighted MSE by the sum of pseudo
weights. `disk` remains the default for existing runs. The notebook
`mvdet-yolo26x-pseudo.ipynb` passes these options to `main.py` directly.

```bash
python main.py -d wildtrack --data_path ~/Data/Wildtrack --drop_ratio 60 \
  --loss brl --use_pseudo_labels --pseudo_cache pseudo_cache/wildtrack_yolo26x.json \
  --pseudo_method gaussian --pseudo_suppress_radius_m 0.5
```

### Frozen VGGT encoder

Install the [official VGGT repository](https://github.com/facebookresearch/vggt)
and `huggingface_hub`. `--arch vggt` downloads the official
[`facebook/VGGT-1B` checkpoint](https://huggingface.co/facebook/VGGT-1B) on
first use. Use `--vggt_weights /path/to/model.pt` for a local copy of the same
checkpoint. The VGGT aggregator processes all cameras together; its weights are
frozen, and only the new feature adapter, per-camera head and BEV head train.
Images keep their full field of view and aspect ratio. `--vggt_input_width`
controls the encoder resolution (default 518, multiple of 14). The VGGT path
requires `--variant default` and a CUDA GPU with substantial memory.

```bash
pip install huggingface_hub einops safetensors
pip install --no-deps git+https://github.com/facebookresearch/vggt.git
```

```bash
python main.py -d wildtrack --data_path ~/Data/Wildtrack --arch vggt \
  --variant default --loss brl --drop_ratio 60
```

The encoder choice and pseudo method are independent; add the Gaussian pseudo
flags above to the VGGT command if both are wanted. VGGT uses the original
dataset calibration for MVDet projection. It does not replace calibration with
VGGT's camera predictions. The checkpoint's [license](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)
applies to its weights.

For BRL + VGGT without pseudo labels, use the VGGT command above without
`--use_pseudo_labels`, `--pseudo_cache`, or `--pseudo_method`. In the notebook,
set `CONFIG["ARCH"] = "vggt"` and `CONFIG["USE_PSEUDO_LABELS"] = False`.
This skips YOLO installation, cache generation, and pseudo evaluation.

### Detic detector checkpoint as a frozen encoder

`--arch detic` loads the official [Detic R50 COCO + ImageNet-21K
checkpoint](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth)
(`Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth`). Its FPN
backbone is frozen; a 1×1 adapter, MVDet's per-view heads, and the BEV head
are trained. This uses the detector-trained image features, **not** Detic's
ROI boxes or classification predictions. The COCO-inclusive checkpoint was
chosen because its training includes the person class.

Install [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html)
for your PyTorch/CUDA build, then clone [Detic](https://github.com/facebookresearch/Detic)
with its CenterNet2 submodule and install its requirements. Pin `timm==0.5.4`
after the Detic requirements: Detic's backbone code expects that release's
dictionary model configs and `build_model_with_cfg(default_cfg=...)` API.
The checkpoint is downloaded by Detectron2 on first use, or supply `--detic_weights` with a
local file. `--detic_input_width` defaults to 640; `--detic_feature` defaults
to FPN level `p3` (the earliest FPN level in this checkpoint's configuration).

```bash
git clone --recurse-submodules https://github.com/facebookresearch/Detic.git
pip install -r Detic/requirements.txt
pip install --no-deps timm==0.5.4
python main.py -d wildtrack --data_path ~/Data/Wildtrack --arch detic \
  --detic_root /path/to/Detic --loss brl --drop_ratio 60
```

### MV2GF style DA3 feature fusion and pointmap aggregation

`--arch mv2gf` uses frozen [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
features and predicted depth, plus trainable ResNet18 task features. The
`generate_da3_cache.py` script supplies the original calibrated camera poses
to DA3, saves four transformer layers, and backprojects its depth to 3D world
pointmaps in metres. The model fuses these features and max-pools them into
four 0.5 m height slices before the BEV head. Missing or malformed cache files
raise an error. For Wildtrack, calibration units are converted from cm to m.

This adapts MV2GF's TGF and FPA to the existing MVDet/BRL dense heatmap
pipeline. It is **not** a reproduction of the paper's focal loss and offset
head. The [official MV2GF code](https://github.com/Yamameeee/MV2GF) currently
provides a one-sequence GMVD example and precomputes DA3 outputs; its reported
Wildtrack scores are not directly comparable to this training recipe.

```bash
pip install -e /path/to/Depth-Anything-3
python generate_da3_cache.py --dataset wildtrack --data_path ~/Data/Wildtrack \
  --output_dir /path/to/da3_cache
python main.py -d wildtrack --data_path ~/Data/Wildtrack --arch mv2gf \
  --da3_cache /path/to/da3_cache --loss brl --drop_ratio 60
```

Both branches require `--variant default`. The notebook
`mvdet-yolo26x-pseudo.ipynb` exposes `CONFIG["ARCH"] = "detic"` or `"mv2gf"`
and optionally combines either with the existing YOLO pseudo-label loss.
Large official weights and generated caches are not stored in this repository.
