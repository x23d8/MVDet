"""Precompute calibrated DA3 geometry/features for the MV2GF experiment."""

import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from multiview_detector.datasets.Wildtrack import Wildtrack
from multiview_detector.datasets.MultiviewX import MultiviewX


LAYERS = (19, 27, 33, 39)


def backproject(depth, intrinsics, extrinsics):
    """OpenCV camera depth and world-to-camera poses -> world points in metres."""
    views, height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    points = []
    for cam in range(views):
        k, pose, d = intrinsics[cam], extrinsics[cam], depth[cam]
        rays = np.stack([(xx - k[0, 2]) / k[0, 0] * d,
                         (yy - k[1, 2]) / k[1, 1] * d, d], axis=-1)
        world = (rays.reshape(-1, 3) - pose[:3, 3]) @ pose[:3, :3]
        points.append(world.reshape(height, width, 3))
    return np.stack(points).astype(np.float32)


def main(args):
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        if exc.name == 'depth_anything_3':
            raise ImportError('Install the official Depth-Anything-3 package before generating the cache') from exc
        raise ImportError(f'Depth-Anything-3 is missing dependency {exc.name!r}; install it before generating the cache') from exc
    base = Wildtrack(args.data_path) if args.dataset == 'wildtrack' else MultiviewX(args.data_path)
    source = Path(args.data_path) / 'annotations_positions'
    frames = sorted(int(path.stem) for path in source.glob('*.json'))
    if not frames:
        raise FileNotFoundError(f'No annotated frames found in {source}')
    paths = base.get_image_fpaths(frames)
    model = DepthAnything3.from_pretrained(args.da3_weights).to('cuda:0').eval()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit = 0.01 if args.dataset == 'wildtrack' else 1.0
    extrinsics = np.stack([np.vstack([np.column_stack([matrix[:, :3], matrix[:, 3] * unit]),
                                       [0, 0, 0, 1]]) for matrix in base.extrinsic_matrices]).astype(np.float32)
    intrinsics = np.stack(base.intrinsic_matrices).astype(np.float32)
    for frame in tqdm(frames, desc='DA3 cache'):
        target = output / f'{frame:04d}.npz'
        if target.exists() and not args.overwrite:
            continue
        images = [paths[cam][frame] for cam in range(base.num_cam)]
        prediction = model.inference(images, extrinsics=extrinsics, intrinsics=intrinsics,
                                     align_to_input_ext_scale=True, process_res=args.process_res,
                                     export_feat_layers=LAYERS)
        if prediction.extrinsics is None or prediction.intrinsics is None:
            raise RuntimeError('DA3 returned no calibrated camera parameters')
        record = {'point': backproject(prediction.depth, prediction.intrinsics,
                                       prediction.extrinsics).astype(np.float16)}
        for layer in LAYERS:
            key = f'feat_layer_{layer}'
            if key not in prediction.aux:
                raise RuntimeError(f'DA3 output lacks {key}; use the official DA3 giant model')
            record[key] = np.asarray(prediction.aux[key], dtype=np.float16)
        temporary = target.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **record)
        os.replace(temporary, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wildtrack', 'multiviewx'], default='wildtrack')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--da3_weights', default='depth-anything/DA3NESTED-GIANT-LARGE',
                        help='Hugging Face model ID or local model directory')
    parser.add_argument('--process_res', type=int, default=504)
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
