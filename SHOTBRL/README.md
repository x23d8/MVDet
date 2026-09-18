# SHOT

Our code is based on [MVDet](https://github.com/hou-yz/MVDet). Please see `README-mvdet.md` for installation and dataset preparation.

# Our code

- `multiview_detector/models/dpersp_trans_detector.py` is the implementation of our method.
- `multiview_detector/models/dpersp_trans_detector_visualize.py` is for visualizing the feature maps in our method.

## Run BRL on Kaggle

Attach the `lee735/thesis-dataset-new` dataset and enable a GPU, then run from
the repository root:

```bash
bash SHOTBRL/run_brl.sh
```

The script reads Wildtrack and MultiviewX from their Kaggle input mounts and
writes logs, checkpoints, and generated evaluation ground truth to
`/kaggle/working/shotbrl-output`. By default it runs drop ratios 20, 45, and 60
for both datasets. A subset can be selected with environment variables:

```bash
DATASETS="wildtrack" DROP_RATIOS="20" bash SHOTBRL/run_brl.sh
```

The input, output, Python executable, and GPU can also be overridden with
`WILDTRACK_ROOT`, `MULTIVIEWX_ROOT`, `OUTPUT_ROOT`, `PYTHON_BIN`, and `GPU_ID`.

