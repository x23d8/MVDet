import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import math
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
from pathlib import Path
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from dotenv import load_dotenv
from multiview_detector.datasets import *
from multiview_detector.loss import BEVBRLLoss, GaussianMSE, PartialViewBRLLoss
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


WANDB_ENTITY = 'GFA26AI02'
WANDB_PROJECT = 'baseline-expriments'


def configure_classifier_for_logits(classifier, foreground_prior=0.01):
    """Add and initialize a foreground-prior bias on a classifier's last conv."""
    if not 0 < foreground_prior < 1:
        raise ValueError('foreground_prior must be between 0 and 1')
    if not isinstance(classifier, nn.Sequential) or not isinstance(classifier[-1], nn.Conv2d):
        raise TypeError('Expected a sequential classifier ending in nn.Conv2d')

    old_conv = classifier[-1]
    if old_conv.bias is None:
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=True,
            padding_mode=old_conv.padding_mode,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight)
        classifier[-1] = new_conv
        old_conv = new_conv

    initial_bias = math.log(foreground_prior / (1.0 - foreground_prior))
    with torch.no_grad():
        old_conv.bias.fill_(initial_bias)


def init_wandb(args, model, train_set, test_set, logdir):
    """Initialize one W&B run for either Wildtrack or MultiviewX."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'Cannot import Weights & Biases. Reinstall the W&B dependencies from requirements.txt.'
        ) from exc

    load_dotenv(Path(__file__).resolve().with_name('.env'))

    # W&B expects WANDB_API_KEY. Keep compatibility with the WANDB_API name
    # already used by this project without ever adding the secret to config.
    if not os.getenv('WANDB_API_KEY') and os.getenv('WANDB_API'):
        os.environ['WANDB_API_KEY'] = os.environ['WANDB_API']

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_type = 'inference' if args.resume else 'train'
    run_name = args.wandb_run_name or (
        f'{args.dataset}-pa{args.pa}-{args.variant}-{args.arch}-{job_type}-{timestamp}'
    )
    config = dict(vars(args))
    config.update({
        'model_name': type(model).__name__,
        'model_variant': args.variant,
        'architecture': args.arch,
        'dataset_name': args.dataset,
        'optimizer': 'SGD',
        'scheduler': 'OneCycleLR',
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'classification_threshold': args.cls_thres,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'grid_reduce': train_set.grid_reduce,
        'image_reduce': train_set.img_reduce,
        'train_ratio': 0.9,
        'input_size': [720, 1280],
        'num_cameras': train_set.num_cam,
        'train_samples': len(train_set),
        'test_samples': len(test_set),
        'partial_annotation_percent': args.pa,
        'annotation_drop_rate': args.pa / 100,
        'train_annotation_dir': train_set.annotation_dir,
        'resume_checkpoint': args.resume,
    })
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        job_type=job_type,
        config=config,
        dir=logdir,
        mode=args.wandb_mode,
        tags=[args.dataset, f'pa{args.pa}', args.variant, args.arch],
    )
    run.define_metric('epoch')
    for namespace in ('train/*', 'validation/*', 'final_test/*', 'gpu/*'):
        run.define_metric(namespace, step_metric='epoch')
    return run


def gpu_metrics():
    """Return explicit per-GPU memory/utilization metrics for W&B and stdout."""
    if not torch.cuda.is_available():
        return {'gpu/available': 0}

    metrics = {'gpu/available': 1, 'gpu/device_count': torch.cuda.device_count()}
    for device_idx in range(torch.cuda.device_count()):
        prefix = f'gpu/{device_idx}'
        props = torch.cuda.get_device_properties(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        max_allocated = torch.cuda.max_memory_allocated(device_idx)
        metrics[f'{prefix}/memory_allocated_mb'] = allocated / (1024 ** 2)
        metrics[f'{prefix}/memory_reserved_mb'] = reserved / (1024 ** 2)
        metrics[f'{prefix}/max_memory_allocated_mb'] = max_allocated / (1024 ** 2)
        metrics[f'{prefix}/memory_allocated_percent'] = allocated / props.total_memory * 100
        try:
            metrics[f'{prefix}/utilization_percent'] = torch.cuda.utilization(device_idx)
        except Exception:
            # W&B's system monitor still records utilization when NVML is available.
            pass
    return metrics


def reset_peak_gpu_memory():
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_idx)


def log_phase(run, phase, epoch, metrics, optimizer=None):
    payload = {'epoch': epoch}
    payload.update({f'{phase}/{key}': value for key, value in metrics.items()})
    payload.update(gpu_metrics())
    if optimizer is not None:
        payload['train/learning_rate'] = optimizer.param_groups[0]['lr']
    run.log(payload)

    gpu_text = ', '.join(
        f'GPU {idx}: {payload.get(f"gpu/{idx}/utilization_percent", "n/a")}% util, '
        f'{payload[f"gpu/{idx}/memory_allocated_mb"]:.0f} MB allocated'
        for idx in range(torch.cuda.device_count())
    ) if torch.cuda.is_available() else 'GPU: unavailable'
    metric_text = ', '.join(f'{key}: {value:.4f}' for key, value in metrics.items())
    print(f'[{phase}] Epoch: {epoch}, {metric_text}, {gpu_text}')


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    if args.data_path:
        requested_data_path = args.data_path
        args.dataset, data_path = detect_dataset_root(
            requested_data_path,
            dataset_name=args.dataset,
        )
        args.dataset_root = data_path
        print(f'Detected {args.dataset} dataset at {data_path}')
    elif args.dataset in (None, 'wildtrack'):
        args.dataset = 'wildtrack'
        data_path = os.path.expanduser('~/Data/Wildtrack')
        requested_data_path = data_path
        args.dataset_root = data_path
    elif args.dataset == 'multiviewx':
        data_path = os.path.expanduser('~/Data/MultiviewX')
        requested_data_path = data_path
        args.dataset_root = data_path
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    if args.dataset == 'wildtrack':
        base = Wildtrack(data_path)
    else:
        base = MultiviewX(data_path)
    args.dataset_root = base.root
    if not args.data_path:
        print(f'Located {args.dataset} dataset at {base.root}')

    train_annotation_dir, hidden_annotation_dir = resolve_annotation_dirs(
        base.root,
        args.dataset,
        args.pa,
        search_root=requested_data_path,
        dropped_path=args.dropped_path,
    )
    if args.pa:
        print(f'Using {args.pa}% dropped training annotations from {train_annotation_dir}')
        print(f'Hidden annotations (excluded from training): {hidden_annotation_dir}')
    else:
        print(f'Using full training annotations from {train_annotation_dir}')

    train_set = frameDataset(
        base,
        train=True,
        transform=train_trans,
        grid_reduce=4,
        annotation_dir=train_annotation_dir,
    )
    # Validation/test labels and evaluation GT must remain complete for every
    # partial-annotation setting, so no alternate annotation directory is used.
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

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
    else:
        raise Exception('no support for this variant')

    # loss
    resolved_loss = args.loss
    if resolved_loss == 'auto':
        resolved_loss = 'bev_brl' if args.pa > 0 else 'gaussian_mse'
    if args.pa > 0 and resolved_loss == 'gaussian_mse':
        raise ValueError(
            'GaussianMSE supervises every zero target as background and is unsafe for --pa > 0. '
            'Use --loss bev_brl (or leave --loss auto).'
        )
    args.resolved_loss = resolved_loss

    if resolved_loss == 'bev_brl':
        configure_classifier_for_logits(model.map_classifier, args.brl_occupancy_prior)
        use_consensus = not args.brl_no_consensus and args.variant != 'img_proj'
        if args.variant == 'img_proj':
            print('ImageProjVariant has no learned head/foot branch; disabling view loss and consensus.')
            view_criterion = None
        else:
            configure_classifier_for_logits(model.img_classifier, args.view_occupancy_prior)
            view_criterion = PartialViewBRLLoss(
                positive_threshold=args.view_positive_threshold,
                ignore_threshold=args.view_ignore_threshold,
                negative_threshold=args.view_negative_threshold,
                hard_negative_threshold=args.view_hard_negative_threshold,
                pseudo_threshold=args.view_pseudo_threshold,
                support_negative_threshold=args.view_support_negative_threshold,
                support_positive_threshold=args.view_support_positive_threshold,
                min_views=args.brl_min_views,
                consensus_topk=args.brl_consensus_topk,
                local_max_kernel=args.brl_local_max_kernel,
                positive_weight=args.view_positive_weight,
                negative_weight=args.view_negative_weight,
                pseudo_weight=args.view_pseudo_weight,
                head_weight=args.view_head_weight,
                foot_weight=args.view_foot_weight,
                head_negative_weight=args.view_head_negative_weight,
                warmup_epochs=args.brl_warmup_epochs,
                ramp_epochs=args.brl_ramp_epochs,
                coverage_threshold=args.brl_coverage_threshold,
                max_pseudo_per_observed=args.view_max_pseudo_per_observed,
            ).cuda()
            if args.variant == 'res_proj':
                model.view_outputs_logits = True

        criterion = BEVBRLLoss(
            positive_threshold=args.brl_positive_threshold,
            ignore_threshold=args.brl_ignore_threshold,
            negative_threshold=args.brl_negative_threshold,
            view_negative_threshold=args.brl_view_negative_threshold,
            hard_negative_threshold=args.brl_hard_negative_threshold,
            bev_threshold=args.brl_bev_threshold,
            view_threshold=args.brl_view_threshold,
            min_views=args.brl_min_views,
            consensus_topk=args.brl_consensus_topk,
            local_max_kernel=args.brl_local_max_kernel,
            positive_weight=args.brl_positive_weight,
            negative_weight=args.brl_negative_weight,
            brl_weight=args.brl_weight,
            warmup_epochs=args.brl_warmup_epochs,
            ramp_epochs=args.brl_ramp_epochs,
            negative_warmup_factor=args.brl_negative_warmup_factor,
            coverage_threshold=args.brl_coverage_threshold,
            max_mirror_per_observed=args.brl_max_mirror_per_observed,
            use_consensus=use_consensus,
            view_outputs_logits=True,
        ).cuda()
    else:
        criterion = GaussianMSE().cuda()
        view_criterion = criterion if args.variant != 'img_proj' else None

    # Classifier layers may have been replaced to add logit biases.  Create the
    # optimizer afterwards so every new parameter is trainable.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # local and W&B logging
    variant_logdir = os.path.join('logs', f'{args.dataset}_frame', args.variant)
    pa_logdir = os.path.join(variant_logdir, f'pa{args.pa}')
    if args.resume is None:
        logdir = os.path.join(pa_logdir, datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logdir = os.path.join(pa_logdir, args.resume)
        legacy_logdir = os.path.join(variant_logdir, args.resume)
        if not os.path.isdir(logdir) and os.path.isdir(legacy_logdir):
            logdir = legacy_logdir
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    wandb_run = init_wandb(args, model, train_set, test_set, logdir)

    if args.resume is None:
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = PerspectiveTrainer(
        model,
        criterion,
        logdir,
        denormalize,
        args.cls_thres,
        args.alpha,
        view_criterion=view_criterion,
    )

    # learn
    try:
        if args.resume is None:
            print('Testing before training...')
            reset_peak_gpu_memory()
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
            log_phase(wandb_run, 'validation', 0, trainer.last_test_metrics)

            for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
                print('Training...')
                reset_peak_gpu_memory()
                train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
                log_phase(wandb_run, 'train', epoch, trainer.last_train_metrics, optimizer)

                print('Testing...')
                reset_peak_gpu_memory()
                test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                          train_set.gt_fpath, True)
                log_phase(wandb_run, 'validation', epoch, trainer.last_test_metrics)

                x_epoch.append(epoch)
                train_loss_s.append(train_loss)
                train_prec_s.append(train_prec)
                test_loss_s.append(test_loss)
                test_prec_s.append(test_prec)
                test_moda_s.append(moda)
                draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                           test_loss_s, test_prec_s, test_moda_s)
                # save
                torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
        else:
            resume_fname = os.path.join(logdir, 'MultiviewDetector.pth')
            state_dict = torch.load(resume_fname)
            incompatible = model.load_state_dict(state_dict, strict=resolved_loss != 'bev_brl')
            if resolved_loss == 'bev_brl' and (incompatible.missing_keys or incompatible.unexpected_keys):
                print('Loaded checkpoint with non-strict BRL compatibility:', incompatible)
            model.eval()

        print('Test loaded model...')
        reset_peak_gpu_memory()
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
        final_epoch = args.epochs if args.resume is None else 0
        log_phase(wandb_run, 'final_test', final_epoch, trainer.last_test_metrics)
    finally:
        wandb_run.finish()


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--loss', type=str, default='auto',
                        choices=['auto', 'gaussian_mse', 'bev_brl'],
                        help='auto selects BEV-BRL for partial labels and GaussianMSE for full labels')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default=None, choices=['wildtrack', 'multiviewx'],
                        help='dataset type; searches the default path and Kaggle inputs when --data_path is omitted')
    parser.add_argument('--data_path', type=str, default=None,
                        help='optional complete dataset root containing images and calibration')
    parser.add_argument('--dropped_path', type=str, default=None,
                        help='separate root containing partial annotations; may point to the dropped dataset or drop setting')
    parser.add_argument('--pa', type=int, default=0, choices=[0, 20, 45, 60],
                        help='percentage of training annotations dropped (default: 0/full annotations)')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='optional custom W&B run name')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='W&B sync mode (default: online)')
    # BEV BRL assignment and weighting.
    parser.add_argument('--brl_positive_threshold', type=float, default=0.10)
    parser.add_argument('--brl_ignore_threshold', type=float, default=0.01)
    parser.add_argument('--brl_negative_threshold', type=float, default=0.15)
    parser.add_argument('--brl_view_negative_threshold', type=float, default=0.30,
                        help='maximum cross-view consensus for reliable BEV negatives')
    parser.add_argument('--brl_hard_negative_threshold', type=float, default=0.40,
                        help='BEV confidence that marks a reliable negative as hard')
    parser.add_argument('--brl_bev_threshold', '--brl_mirror_threshold',
                        dest='brl_bev_threshold', type=float, default=0.60,
                        help='BEV confidence required for mirror-positive candidates')
    parser.add_argument('--brl_view_threshold', type=float, default=0.65)
    parser.add_argument('--brl_min_views', type=int, default=2)
    parser.add_argument('--brl_consensus_topk', type=int, default=2)
    parser.add_argument('--brl_local_max_kernel', type=int, default=3)
    parser.add_argument('--brl_positive_weight', type=float, default=1.0)
    parser.add_argument('--brl_negative_weight', type=float, default=0.50)
    parser.add_argument('--brl_weight', type=float, default=0.10)
    parser.add_argument('--brl_warmup_epochs', type=int, default=1)
    parser.add_argument('--brl_ramp_epochs', type=int, default=2)
    parser.add_argument('--brl_negative_warmup_factor', type=float, default=0.25)
    parser.add_argument('--brl_coverage_threshold', type=float, default=0.50)
    parser.add_argument('--brl_max_mirror_per_observed', type=float, default=1.00)
    parser.add_argument('--brl_occupancy_prior', type=float, default=0.01)
    parser.add_argument('--brl_no_consensus', action='store_true')

    # Partial-aware camera head/foot supervision.
    parser.add_argument('--view_positive_threshold', type=float, default=0.10)
    parser.add_argument('--view_ignore_threshold', type=float, default=0.01)
    parser.add_argument('--view_negative_threshold', type=float, default=0.15)
    parser.add_argument('--view_hard_negative_threshold', type=float, default=0.40)
    parser.add_argument('--view_pseudo_threshold', type=float, default=0.60)
    parser.add_argument('--view_support_negative_threshold', type=float, default=0.30)
    parser.add_argument('--view_support_positive_threshold', type=float, default=0.65)
    parser.add_argument('--view_positive_weight', type=float, default=1.0)
    parser.add_argument('--view_negative_weight', type=float, default=0.25)
    parser.add_argument('--view_pseudo_weight', type=float, default=0.10)
    parser.add_argument('--view_head_weight', type=float, default=0.05)
    parser.add_argument('--view_foot_weight', type=float, default=1.0)
    parser.add_argument('--view_head_negative_weight', type=float, default=0.0,
                        help='zero keeps all unlabelled head pixels out of negative supervision')
    parser.add_argument('--view_max_pseudo_per_observed', type=float, default=1.00)
    parser.add_argument('--view_occupancy_prior', type=float, default=0.01)
    args = parser.parse_args()

    main(args)
