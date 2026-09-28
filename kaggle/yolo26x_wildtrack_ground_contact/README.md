# YOLO26x Wildtrack ground-contact fine-tuning

This Kaggle notebook converts Wildtrack `positionID` annotations into one
calibration-derived ground-contact keypoint per visible camera box and
fine-tunes `yolo26x-pose.pt`.

- Dataset: `lee735/thesis-dataset-new`
- Train/validation: first/last contiguous 80/20 timestamp blocks
- Default image size: 960
- Default epochs: 30
- GPU: uses all GPUs exposed by Kaggle (`0,1` on T4 x2)
- Stable output checkpoint: `yolo26x_wildtrack_ground_contact_best.pt`
- Notebook: `yolo26x_wildtrack_ground_contact.ipynb`
- Required Wildtrack path:
  `/kaggle/input/thesis-dataset-new/Wildtrack/Wildtrack`

The held-out temporal block is suitable for detector evaluation. Do not use a
model trained on a frame to claim missing-label recovery performance on that
same frame; use out-of-fold training for the final pseudo-label experiment.
