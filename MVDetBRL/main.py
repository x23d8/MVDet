import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
import datetime
import random
from pathlib import Path
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE
from multiview_detector.loss.pu_gaussian_mse import PUGaussianMSE
from multiview_detector.loss.pu_query_loss import PUQuerySetLoss
from multiview_detector.loss.camera_drop_consistency import CameraDropConsistencyLoss
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.models.puma_dense_detector import PUMADenseDetector
from multiview_detector.models.puma_hybrid_detector import PUMAHybridDetector
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.utils.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)
from multiview_detector.trainer import PerspectiveTrainer


def snapshot_training_sources(project_root, destination):
    """Copy reproducibility-critical text sources without bundled toolkits.

    The legacy recursive copy included the complete MOTChallenge MATLAB
    devkit. Besides being unrelated to training, its deep paths exceed the
    default Windows path limit once nested below a descriptive log directory.
    """
    project_root = Path(project_root)
    destination = Path(destination)
    candidates = list((project_root / 'multiview_detector').rglob('*.py'))
    candidates.extend(
        path for path in project_root.iterdir()
        if path.is_file() and path.suffix.lower() in {'.py', '.sh', '.txt', '.md'}
    )
    for source in candidates:
        relative = source.relative_to(project_root)
        if 'motchallenge-devkit' in relative.parts or '__pycache__' in relative.parts:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_criterion(args, device):
    if args.loss == 'brl':
        return BRLGaussianMSE(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
        ).to(device)
    if args.loss == 'pu':
        propensity = args.pu_annotation_propensity
        if propensity is None:
            propensity = 1.0 - args.drop_ratio / 100.0
        return PUGaussianMSE(
            annotation_propensity=propensity,
            pos_thr=args.pu_pos_thr,
            class_prior=args.pu_class_prior,
            beta=args.pu_beta,
            gamma=args.pu_gamma,
        ).to(device)
    return GaussianMSE().to(device)


def parse_cuda_devices(spec):
    devices = [int(value.strip()) for value in spec.split(',') if value.strip()]
    if not devices:
        raise ValueError('--devices must contain at least one CUDA device index')
    if len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError('--devices must contain unique non-negative CUDA indices')
    if not torch.cuda.is_available():
        raise RuntimeError('This training entry point requires CUDA')
    unavailable = [device for device in devices if device >= torch.cuda.device_count()]
    if unavailable:
        raise ValueError(f'CUDA devices are unavailable: {unavailable}')
    return devices


def main(args):
    device_ids = parse_cuda_devices(args.devices)
    primary_device = torch.device(f'cuda:{device_ids[0]}')
    torch.cuda.set_device(primary_device)
    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    # cuDNN benchmark uses a FIND pass that can request a large temporary
    # workspace for the flattened seven-camera batch. On Kaggle T4 this can
    # fail with "FIND was unable to find an engine" before the first step.
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if args.image_height <= 0 or args.image_width <= 0:
        raise ValueError('--image_height and --image_width must be positive')
    train_trans = T.Compose([
        T.Resize([args.image_height, args.image_width]),
        T.ToTensor(),
        normalize,
    ])
    
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '/kaggle/working/Data_temp/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '/kaggle/working/Data_temp/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    if args.nms_radius_m is None:
        args.nms_radius_m = 0.3 if args.variant in ('puma_dense', 'puma_hybrid') else 0.5
    if args.nms_radius_m <= 0:
        raise ValueError('--nms_radius_m must be positive')
    native_to_m = 0.01 if base.__name__.lower() == 'wildtrack' else 1.0
    origin = np.asarray(base.get_worldcoord_from_worldgrid(np.array([0, 0])), dtype=float)
    one_cell = np.asarray(base.get_worldcoord_from_worldgrid(np.array([1, 0])), dtype=float)
    grid_cell_m = np.linalg.norm(one_cell - origin) * native_to_m
    if grid_cell_m <= 0:
        raise ValueError('dataset world grid has invalid metric cell size')
    nms_radius_grid = args.nms_radius_m / grid_cell_m
    if not (0 < args.train_end_ratio <= args.eval_start_ratio < args.eval_end_ratio <= 1.0):
        raise ValueError(
            'require 0 < train_end_ratio <= eval_start_ratio < eval_end_ratio <= 1'
        )
    train_frame_range = (0, int(base.num_frame * args.train_end_ratio))
    eval_frame_range = (
        int(base.num_frame * args.eval_start_ratio),
        int(base.num_frame * args.eval_end_ratio),
    )

    if args.use_pseudo_labels and not args.pseudo_cache:
        raise ValueError('--use_pseudo_labels requires --pseudo_cache')
    if args.gt_path:
        gt_fpath = os.path.abspath(os.path.expanduser(args.gt_path))
    else:
        dataset_gt = os.path.join(data_path, 'gt.txt')
        gt_fpath = dataset_gt if os.path.isfile(dataset_gt) else os.path.abspath(
            os.path.join('generated_gt', f'{args.dataset}_gt.txt')
        )
    train_set = frameDataset(
        base, train=True, transform=train_trans, grid_reduce=4, drop_ratio=args.drop_ratio,
        annotation_dir=args.train_annotation_dir,
        frame_range=train_frame_range,
        gt_fpath=gt_fpath, force_download=args.regenerate_gt,
        pseudo_cache=args.pseudo_cache if args.use_pseudo_labels else None,
        pseudo_conf_threshold=args.pseudo_conf_threshold,
        pseudo_sigma_m=args.pseudo_sigma_m,
        pseudo_suppress_radius_m=args.pseudo_suppress_radius_m)
    # Evaluation must always use exhaustive annotations, independently of the
    # partial-label mechanism used for training.
    test_set = frameDataset(
        base, train=False, transform=train_trans, grid_reduce=4, drop_ratio=0,
        frame_range=eval_frame_range, gt_fpath=gt_fpath, force_download=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    elif args.variant == 'puma_dense':
        if args.arch != 'resnet18':
            raise ValueError('puma_dense currently supports --arch resnet18 only')
        model = PUMADenseDetector(
            train_set,
            feature_channels=args.puma_feature_channels,
            fused_channels=args.puma_fused_channels,
            pretrained=args.puma_pretrained,
            parallel_view_encoding=args.puma_parallel_view_encoding,
        )
    elif args.variant == 'puma_hybrid':
        if args.arch != 'resnet18':
            raise ValueError('puma_hybrid currently supports --arch resnet18 only')
        model = PUMAHybridDetector(
            train_set,
            feature_channels=args.puma_feature_channels,
            fused_channels=args.puma_fused_channels,
            num_queries=args.puma_num_queries,
            query_hidden_channels=args.puma_query_channels,
            query_layers=args.puma_query_layers,
            pretrained=args.puma_pretrained,
            parallel_view_encoding=args.puma_parallel_view_encoding,
        )
    else:
        raise Exception('no support for this variant')
    if args.variant not in ('puma_dense', 'puma_hybrid') and device_ids != [0]:
        raise ValueError('legacy variants contain hard-coded cuda:0 placement; use --devices 0 only')
    model = model.to(primary_device)
    if len(device_ids) > 1:
        if args.batch_size < len(device_ids):
            print(f'Warning: batch_size={args.batch_size} cannot keep all {len(device_ids)} GPUs busy')
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])

    optimizer_name = args.optimizer
    if optimizer_name == 'auto':
        optimizer_name = 'adamw' if args.variant in ('puma_dense', 'puma_hybrid') else 'sgd'
    if args.lr is None:
        args.lr = 2e-4 if optimizer_name == 'adamw' else 0.1
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = build_criterion(args, primary_device)
    query_criterion = None
    if args.variant == 'puma_hybrid' and args.puma_query_loss_weight > 0:
        propensity = args.pu_annotation_propensity
        if propensity is None:
            propensity = 1.0 - args.drop_ratio / 100.0
        query_criterion = PUQuerySetLoss(
            annotation_propensity=propensity,
            coordinate_weight=args.puma_query_coordinate_weight,
            beta=args.pu_beta,
            gamma=args.pu_gamma,
            separation_weight=args.puma_separation_loss_weight,
            minimum_separation_m=args.puma_minimum_separation_m,
            propensity_mode=args.pu_propensity_mode,
            propensity_regularization=args.pu_propensity_regularization,
        ).to(primary_device)
    consistency_criterion = None
    if (args.variant in ('puma_dense', 'puma_hybrid')
            and args.consistency_loss_weight > 0
            and args.camera_drop_prob > 0):
        consistency_criterion = CameraDropConsistencyLoss(
            confidence_threshold=args.consistency_threshold
        ).to(primary_device)

    # logging
    drop_tag = f'drop_{args.drop_ratio}' if args.drop_ratio > 0 else 'full'
    if args.loss == 'brl':
        loss_tag = f'brl_b{args.brl_beta}_c{args.brl_confuse_thr}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
    elif args.loss == 'pu':
        propensity = args.pu_annotation_propensity
        if propensity is None:
            propensity = 1.0 - args.drop_ratio / 100.0
        prior_tag = 'dynamic' if args.pu_class_prior is None else f'{args.pu_class_prior:g}'
        loss_tag = (
            f'pu_{args.pu_propensity_mode}_e{propensity:g}'
            f'_p{prior_tag}_t{args.pu_pos_thr:g}'
        )
    else:
        loss_tag = 'mse'
    if args.use_pseudo_labels:
        loss_tag += (f'_pseudo_w{args.pseudo_loss_weight:g}_c{args.pseudo_conf_threshold:g}'
                     f'_s{args.pseudo_sigma_m:g}_r{args.pseudo_suppress_radius_m:g}')
    split_tag = (
        f'train_{args.train_end_ratio:g}_eval_'
        f'{args.eval_start_ratio:g}-{args.eval_end_ratio:g}'
    )
    log_parent = os.path.join(
        'logs', f'{args.dataset}_frame', split_tag, drop_tag, loss_tag, args.variant
    )
    if not args.resume:
        logdir = os.path.join(
            log_parent, datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        )
    else:
        resume_path = os.path.abspath(os.path.expanduser(args.resume))
        logdir = resume_path if os.path.isabs(args.resume) else os.path.join(
            log_parent, args.resume
        )
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        snapshot_training_sources('.', os.path.join(logdir, 'scripts'))
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'))
    else:
        if not os.path.isdir(logdir):
            raise FileNotFoundError(f'Resume directory not found: {logdir}')
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), mode='a')
    print('Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha,
                                 args.pseudo_loss_weight if args.use_pseudo_labels else 0.0,
                                 amp=args.amp,
                                 query_criterion=query_criterion,
                                 query_loss_weight=args.puma_query_loss_weight,
                                 query_warmup_epochs=args.puma_query_warmup_epochs,
                                 query_ramp_epochs=args.puma_query_ramp_epochs,
                                 consistency_criterion=consistency_criterion,
                                 consistency_loss_weight=args.consistency_loss_weight,
                                 camera_drop_prob=args.camera_drop_prob,
                                 consistency_ramp_epochs=args.consistency_ramp_epochs,
                                 nms_radius_grid=nms_radius_grid)

    # Restore a complete training state when available. Older weight-only
    # checkpoints remain evaluation-compatible but cannot resume optimization.
    start_epoch = 1
    model_without_wrapper = model.module if isinstance(model, torch.nn.DataParallel) else model
    if args.resume is not None:
        training_state_path = os.path.join(logdir, 'training_state.pth')
        if os.path.isfile(training_state_path):
            state = load_checkpoint(training_state_path, map_location=primary_device)
            saved_args = state.get('args', {})
            resume_keys = (
                'dataset', 'data_path', 'variant', 'arch', 'loss', 'drop_ratio',
                'train_annotation_dir', 'train_end_ratio', 'batch_size',
                'image_height', 'image_width',
                'optimizer', 'lr', 'puma_feature_channels',
                'puma_fused_channels', 'puma_query_channels',
                'puma_query_layers', 'puma_query_warmup_epochs',
                'puma_query_ramp_epochs', 'pu_propensity_mode',
            )
            mismatches = [
                key for key in resume_keys
                if key in saved_args and saved_args[key] != getattr(args, key)
            ]
            if mismatches:
                details = ', '.join(
                    f'{key}: saved={saved_args[key]!r}, current={getattr(args, key)!r}'
                    for key in mismatches
                )
                raise ValueError(f'resume configuration mismatch: {details}')
            configured_epochs = state.get('configured_epochs')
            if configured_epochs is not None and int(configured_epochs) != args.epochs:
                raise ValueError(
                    f'checkpoint was configured for {configured_epochs} epochs; '
                    f'resume requires --epochs {configured_epochs}'
                )
            model_without_wrapper.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            trainer.scaler.load_state_dict(state.get('scaler', {}))
            restore_rng_state(state['rng_state'])
            start_epoch = int(state['epoch']) + 1
            x_epoch = list(state.get('x_epoch', []))
            train_loss_s = list(state.get('train_loss_s', []))
            train_prec_s = list(state.get('train_prec_s', []))
            test_loss_s = list(state.get('test_loss_s', []))
            test_prec_s = list(state.get('test_prec_s', []))
            test_moda_s = list(state.get('test_moda_s', []))
            print(f'Resuming training at epoch {start_epoch}')
        else:
            resume_fname = os.path.join(logdir, 'MultiviewDetector.pth')
            model_without_wrapper.load_state_dict(
                torch.load(resume_fname, map_location=primary_device, weights_only=True)
            )
            start_epoch = args.epochs + 1
            print('Legacy weight-only checkpoint loaded for evaluation')

    if start_epoch == 1 and not args.skip_initial_test:
        print('Initial evaluation...')
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)

    for epoch in tqdm.tqdm(range(start_epoch, args.epochs + 1)):
        print('Training...')
        train_loss, train_prec = trainer.train(
            epoch, train_loader, optimizer, args.log_interval, scheduler)
        x_epoch.append(epoch)
        train_loss_s.append(train_loss)
        train_prec_s.append(train_prec)
        should_evaluate = args.eval_every > 0 and (
            epoch % args.eval_every == 0 and epoch != args.epochs
        )
        if should_evaluate:
            print('Testing...')
            test_loss, test_prec, moda = trainer.test(
                test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True
            )
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            test_moda_s.append(moda)
            if args.eval_every == 1:
                draw_curve(
                    os.path.join(logdir, 'learning_curve.jpg'),
                    x_epoch, train_loss_s, train_prec_s,
                    test_loss_s, test_prec_s, test_moda_s,
                )
        torch.save(
            model_without_wrapper.state_dict(),
            os.path.join(logdir, 'MultiviewDetector.pth'),
        )
        save_checkpoint_atomic(os.path.join(logdir, 'training_state.pth'), {
            'epoch': epoch,
            'configured_epochs': args.epochs,
            'model': model_without_wrapper.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': trainer.scaler.state_dict(),
            'rng_state': capture_rng_state(),
            'x_epoch': x_epoch,
            'train_loss_s': train_loss_s,
            'train_prec_s': train_prec_s,
            'test_loss_s': test_loss_s,
            'test_prec_s': test_prec_s,
            'test_moda_s': test_moda_s,
            'args': vars(args),
        })
    print('Testing...')
    trainer.test(
        test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True,
        score_cache_path=(os.path.join(logdir, 'score_cache.npz')
                          if args.save_score_cache else None),
    )


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector + BRL heatmap loss')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--nms_radius_m', type=float, default=None,
                        help='validation-selected point-NMS radius in metres; defaults to 0.3 for PUMA')
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv',
                                 'puma_dense', 'puma_hybrid'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('--puma_feature_channels', type=int, default=16,
                        help='per-view channels retained before multi-height sampling')
    parser.add_argument('--puma_fused_channels', type=int, default=48,
                        help='channels in the uncertainty-gated BEV representation')
    parser.add_argument('--puma_pretrained', action=argparse.BooleanOptionalAction, default=True,
                        help='initialize the PUMA ResNet-18 encoder from ImageNet weights')
    parser.add_argument('--puma_parallel_view_encoding',
                        action=argparse.BooleanOptionalAction, default=False,
                        help='encode all cameras together (faster, but uses more peak memory)')
    parser.add_argument('--puma_num_queries', type=int, default=64,
                        help='number of dense peaks refined by the sparse cylindrical decoder')
    parser.add_argument('--puma_query_channels', type=int, default=48,
                        help='hidden width of the sparse cylindrical query decoder')
    parser.add_argument('--puma_query_layers', type=int, default=2,
                        help='number of iterative sparse query refinement layers')
    parser.add_argument('--puma_query_loss_weight', type=float, default=0.02,
                        help='weight of the partial-label query set objective')
    parser.add_argument('--puma_query_warmup_epochs', type=int, default=3,
                        help='dense-only epochs before enabling direct query supervision')
    parser.add_argument('--puma_query_ramp_epochs', type=int, default=5,
                        help='epochs used to ramp query loss to its configured weight')
    parser.add_argument('--puma_query_coordinate_weight', type=float, default=1.0,
                        help='coordinate term inside the query set objective')
    parser.add_argument('--puma_separation_loss_weight', type=float, default=0.1,
                        help='confidence-weighted local duplicate-query repulsion')
    parser.add_argument('--puma_minimum_separation_m', type=float, default=0.25,
                        help='minimum physical distance used by query separation')
    parser.add_argument('--camera_drop_prob', type=float, default=0.25,
                        help='probability of removing each camera in the consistency branch')
    parser.add_argument('--consistency_loss_weight', type=float, default=0.1,
                        help='weight of positive-only camera-drop consistency')
    parser.add_argument('--consistency_threshold', type=float, default=0.3,
                        help='minimum full-view soft target used for consistency')
    parser.add_argument('--consistency_ramp_epochs', type=int, default=5,
                        help='epochs used to linearly warm up consistency weight')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--data_path', type=str, default=None, help='Dataset root; defaults to ../Data_temp/<dataset>')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='writable full-GT cache path; useful when data_path is read-only')
    parser.add_argument('--regenerate_gt', action='store_true',
                        help='rebuild the full evaluator GT cache from annotations_positions')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=str, default='0',
                        help='comma-separated CUDA device indices, e.g. 0 or 0,1')
    parser.add_argument('--amp', action='store_true',
                        help='use CUDA mixed precision and dynamic gradient scaling')
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction,
                        default=False,
                        help='enable cuDNN autotuning; off by default to avoid large FIND workspaces')
    parser.add_argument('--image_height', type=int, default=720,
                        help='network input height; calibration remains in original-image coordinates')
    parser.add_argument('--image_width', type=int, default=1280,
                        help='network input width; preserve the source aspect ratio')
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', choices=['auto', 'sgd', 'adamw'], default='auto',
                        help='auto selects AdamW for PUMA and SGD for legacy variants')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='default: 2e-4 for AdamW, 0.1 for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='evaluate every N epochs; 0 evaluates only once after training')
    parser.add_argument('--skip_initial_test', action='store_true',
                        help='skip the untrained-model evaluation')
    parser.add_argument('--save_score_cache', action='store_true',
                        help='save final raw BEV maps for validation-only postprocess tuning')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    
    # Dropped annotations
    parser.add_argument('--drop_ratio', type=int, default=0,
                        choices=[0, 20, 45, 60],
                        help='0 = full labels; 20/45/60 = drop_annotations/drop_XX')
    parser.add_argument('--train_annotation_dir', type=str, default=None,
                        help='explicit partial training annotation directory, absolute or relative to dataset root')
    parser.add_argument('--train_end_ratio', type=float, default=0.9,
                        help='exclusive end of the chronological training split')
    parser.add_argument('--eval_start_ratio', type=float, default=0.9,
                        help='inclusive start of validation/test split')
    parser.add_argument('--eval_end_ratio', type=float, default=1.0,
                        help='exclusive end of validation/test split')
    # BRL heatmap loss
    parser.add_argument('--loss', type=str, default='brl', choices=['brl', 'mse', 'pu'],
                        help='brl = background recalibration; pu = positive-unlabeled risk; mse = original loss')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='soft-GT threshold for positive pixels')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pred threshold on background to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='if set, down-weight bg MSE on confuse instead of mirroring toward 1')
    parser.add_argument('--pu_annotation_propensity', type=float, default=None,
                        help='P(label observed | true positive); defaults to 1-drop_ratio/100')
    parser.add_argument('--pu_class_prior', type=float, default=None,
                        help='positive pixel prior; default estimates it from each target and propensity')
    parser.add_argument('--pu_pos_thr', type=float, default=0.1,
                        help='Gaussian heatmap threshold defining observed positive pixels')
    parser.add_argument('--pu_beta', type=float, default=0.0,
                        help='nnPU negative-risk correction margin')
    parser.add_argument('--pu_gamma', type=float, default=1.0,
                        help='nnPU correction strength when empirical negative risk is below -beta')
    parser.add_argument('--pu_propensity_mode', choices=['scar', 'sar'], default='scar',
                        help='constant SCAR-nnPU or learned query-level SAR selection model')
    parser.add_argument('--pu_propensity_regularization', type=float, default=0.1,
                        help='SAR penalty anchoring predicted propensity to the known annotation rate')
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='add offline 2D detector evidence to the BEV training loss')
    parser.add_argument('--pseudo_cache', type=str, default=None,
                        help='JSON cache generated by generate_pseudo_cache.py')
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.01)
    parser.add_argument('--pseudo_conf_threshold', type=float, default=0.2)
    parser.add_argument('--pseudo_sigma_m', type=float, default=0.5)
    parser.add_argument('--pseudo_suppress_radius_m', type=float, default=1.0)
    args = parser.parse_args()

    main(args)
