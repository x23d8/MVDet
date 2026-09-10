import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import subprocess
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
from pathlib import Path
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from dotenv import load_dotenv
from multiview_detector.datasets import *
from multiview_detector.loss import AdaptiveBRLLoss, GaussianMSE, MultiViewAdaptiveBRLLoss
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


class DisabledRun:
    """Small W&B-compatible sink used when experiment logging is disabled."""

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def current_git_commit():
    """Return the source revision recorded with each reproducible run."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_training_checkpoint(path, payload):
    """Atomically replace a checkpoint so an interrupted save keeps the previous one."""
    temporary_path = f'{path}.tmp'
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def init_wandb(args, model, train_set, test_set, logdir):
    """Initialize one W&B run for either Wildtrack or MultiviewX."""
    if args.wandb_mode == 'disabled':
        return DisabledRun()
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
    if not 0 <= args.pa < 100:
        raise ValueError('--pa must be in [0, 100)')
    if args.eval_interval < 1:
        raise ValueError('--eval_interval must be at least 1')
    if args.checkpoint_interval < 0:
        raise ValueError('--checkpoint_interval must be non-negative')
    if args.resume is not None and args.resume_training is not None:
        raise ValueError('--resume and --resume_training are mutually exclusive')
    cache_dir = args.cache_dir or os.path.join(os.getcwd(), '.cache', 'mvdet')
    args.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    os.environ['MVDET_CACHE_DIR'] = args.cache_dir
    args.git_commit = current_git_commit()
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

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed if args.seed is not None else torch.initial_seed())
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=train_generator,
    )
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

    loss_device = getattr(model, 'fusion_device', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # Loss. With dropped annotations, zero means "unlabeled", not background.
    resolved_loss = args.loss
    if resolved_loss == 'auto':
        resolved_loss = 'adaptive_brl' if args.pa > 0 else 'gaussian_mse'
    if resolved_loss == 'gaussian_mse' and args.pa > 0:
        raise ValueError(
            'GaussianMSE treats missing positives as background; use --loss adaptive_brl for --pa > 0'
        )
    args.resolved_loss = resolved_loss
    if resolved_loss == 'adaptive_brl':
        annotation_probability = 1.0 - args.pa / 100.0
        loss_kwargs = dict(
            annotation_probability=annotation_probability,
            gamma=args.brl_gamma,
            background_weight=args.brl_background_weight,
            pseudo_weight=args.brl_pseudo_weight,
            warmup_epochs=args.brl_warmup_epochs,
            ramp_epochs=args.brl_ramp_epochs,
            max_pseudo_ratio=args.brl_max_pseudo_ratio,
        )
        if args.variant == 'img_proj':
            criterion = AdaptiveBRLLoss(**loss_kwargs).to(loss_device)
            view_criterion = None
        else:
            criterion = MultiViewAdaptiveBRLLoss(
                min_views=args.brl_min_views,
                **loss_kwargs,
            ).to(loss_device)
            view_criterion = AdaptiveBRLLoss(**loss_kwargs).to(loss_device)
    else:
        criterion = GaussianMSE().to(loss_device)
        view_criterion = criterion if args.variant != 'img_proj' else None

    # local and W&B logging
    variant_logdir = os.path.join('logs', f'{args.dataset}_frame', args.variant)
    pa_logdir = os.path.join(variant_logdir, f'pa{args.pa}')
    resume_run = args.resume_training or args.resume
    if resume_run is None:
        logdir = os.path.join(pa_logdir, datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        logdir = os.path.join(pa_logdir, resume_run)
        legacy_logdir = os.path.join(variant_logdir, resume_run)
        if not os.path.isdir(logdir) and os.path.isdir(legacy_logdir):
            logdir = legacy_logdir
    if resume_run is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    wandb_run = init_wandb(args, model, train_set, test_set, logdir)

    if args.resume is None:
        sys.stdout = Logger(
            os.path.join(logdir, 'log.txt'),
            append=args.resume_training is not None,
        )
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
        amp=args.amp,
    )

    checkpoint_path = os.path.join(logdir, 'training_checkpoint.pth')

    def persist_training_state(epoch, next_batch, epoch_generator_state):
        save_training_checkpoint(
            checkpoint_path,
            {
                'epoch': int(epoch),
                'next_batch_in_epoch': int(next_batch),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'grad_scaler_state_dict': trainer.grad_scaler.state_dict(),
                # This is the state immediately before RandomSampler created
                # the epoch permutation. Restoring it recreates that order.
                'epoch_generator_state': epoch_generator_state.cpu(),
                'config': vars(args),
            },
        )

    # learn
    try:
        start_epoch = 1
        start_batch = 0
        resumed_epoch_generator_state = None
        if args.resume_training is not None:
            map_location = None if torch.cuda.device_count() > 1 else loss_device
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            planned_epochs = checkpoint.get('config', {}).get('epochs')
            if planned_epochs is not None and int(planned_epochs) != args.epochs:
                raise ValueError(
                    '--epochs must match the original run when resuming OneCycleLR '
                    f'(expected {planned_epochs}, got {args.epochs})'
                )
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if checkpoint.get('grad_scaler_state_dict'):
                trainer.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

            checkpoint_epoch = int(checkpoint['epoch'])
            # Old epoch-only checkpoints have no next_batch_in_epoch and keep
            # their historical behavior: resume at the following epoch.
            if 'next_batch_in_epoch' not in checkpoint:
                start_epoch = checkpoint_epoch + 1
            else:
                start_batch = int(checkpoint['next_batch_in_epoch'])
                if start_batch >= len(train_loader):
                    start_epoch = checkpoint_epoch + 1
                    start_batch = 0
                else:
                    start_epoch = checkpoint_epoch
                    resumed_epoch_generator_state = checkpoint.get('epoch_generator_state')
                    if resumed_epoch_generator_state is None:
                        raise ValueError('mid-epoch checkpoint is missing epoch_generator_state')
                    train_generator.set_state(resumed_epoch_generator_state.cpu())
            print(
                f'Resuming training at epoch {start_epoch}, batch {start_batch} '
                f'from {checkpoint_path}'
            )

        if args.resume is None and args.resume_training is None and not args.skip_initial_test:
            print('Testing before training...')
            reset_peak_gpu_memory()
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), test_set.gt_fpath, True)
            log_phase(wandb_run, 'validation', 0, trainer.last_test_metrics)

        evaluated_final_epoch = False
        if args.resume is None:
            for epoch in tqdm.tqdm(range(start_epoch, args.epochs + 1)):
                epoch_start_batch = start_batch if epoch == start_epoch else 0
                if epoch_start_batch:
                    epoch_generator_state = resumed_epoch_generator_state.cpu().clone()
                else:
                    epoch_generator_state = train_generator.get_state().clone()

                def checkpoint_callback(current_epoch, next_batch):
                    if (
                            args.checkpoint_interval > 0
                            and next_batch % args.checkpoint_interval == 0
                    ):
                        persist_training_state(
                            current_epoch, next_batch, epoch_generator_state
                        )
                        print(
                            f'Checkpoint saved at epoch {current_epoch}, '
                            f'next batch {next_batch}',
                            flush=True,
                        )

                print('Training...')
                reset_peak_gpu_memory()
                train_loss, train_prec = trainer.train(
                    epoch,
                    train_loader,
                    optimizer,
                    args.log_interval,
                    scheduler,
                    start_batch=epoch_start_batch,
                    checkpoint_callback=checkpoint_callback,
                )
                log_phase(wandb_run, 'train', epoch, trainer.last_train_metrics, optimizer)

                # An epoch-complete checkpoint is written before validation,
                # so even interruption during evaluation loses no training.
                persist_training_state(epoch, len(train_loader), epoch_generator_state)
                torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))

                should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
                if should_evaluate:
                    print('Testing...')
                    reset_peak_gpu_memory()
                    test_loss, test_prec, moda = trainer.test(
                        test_loader,
                        os.path.join(logdir, 'test.txt'),
                        test_set.gt_fpath,
                        True,
                    )
                    log_phase(wandb_run, 'validation', epoch, trainer.last_test_metrics)

                    x_epoch.append(epoch)
                    train_loss_s.append(train_loss)
                    train_prec_s.append(train_prec)
                    test_loss_s.append(test_loss)
                    test_prec_s.append(test_prec)
                    test_moda_s.append(moda)
                    draw_curve(
                        os.path.join(logdir, 'learning_curve.jpg'),
                        x_epoch,
                        train_loss_s,
                        train_prec_s,
                        test_loss_s,
                        test_prec_s,
                        test_moda_s,
                    )
                    evaluated_final_epoch = epoch == args.epochs
        else:
            resume_fname = os.path.join(logdir, 'MultiviewDetector.pth')
            # A checkpoint saved by the original two-GPU split can be resumed
            # on one GPU or CPU by remapping all serialized tensors first.
            map_location = None if torch.cuda.device_count() > 1 else loss_device
            model.load_state_dict(torch.load(resume_fname, map_location=map_location))
            model.eval()

        if not evaluated_final_epoch:
            print('Test loaded model...')
            reset_peak_gpu_memory()
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), test_set.gt_fpath, True)
        final_epoch = args.epochs if args.resume is None else 0
        log_phase(wandb_run, 'final_test', final_epoch, trainer.last_test_metrics)
        metrics_path = os.path.join(logdir, 'final_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as metrics_file:
            json.dump(
                {
                    'config': vars(args),
                    'metrics': trainer.last_test_metrics,
                    'result_file': os.path.abspath(os.path.join(logdir, 'test.txt')),
                    'ground_truth_file': os.path.abspath(test_set.gt_fpath),
                },
                metrics_file,
                indent=2,
                sort_keys=True,
            )
        print(f'Final metrics written to {metrics_path}')
    finally:
        wandb_run.finish()


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--loss', choices=['auto', 'gaussian_mse', 'adaptive_brl'], default='auto',
                        help='auto uses adaptive BRL for partial annotations and MSE otherwise')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default=None, choices=['wildtrack', 'multiviewx'],
                        help='dataset type; searches the default path and Kaggle inputs when --data_path is omitted')
    parser.add_argument('--data_path', type=str, default=None,
                        help='optional complete dataset root containing images and calibration')
    parser.add_argument('--dropped_path', type=str, default=None,
                        help='separate root containing partial annotations; may point to the dropped dataset or drop setting')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='writable GT cache directory (default: .cache/mvdet)')
    parser.add_argument('--pa', type=int, default=0,
                        help='percentage of training annotations dropped, in [0, 100)')
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
    parser.add_argument('--resume_training', type=str, default=None,
                        help='continue a run directory from training_checkpoint.pth')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--amp', action='store_true',
                        help='use CUDA automatic mixed precision to reduce memory and runtime')
    parser.add_argument('--skip_initial_test', action='store_true',
                        help='skip the untrained epoch-0 evaluation (useful for profiling)')
    parser.add_argument('--eval_interval', type=int, default=1,
                        help='evaluate every N epochs and always at the final epoch')
    parser.add_argument('--checkpoint_interval', type=int, default=25,
                        help='save resumable state every N batches; 0 means epoch-only')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='optional custom W&B run name')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='W&B sync mode (default: online)')
    parser.add_argument('--brl_gamma', type=float, default=2.0,
                        help='focal exponent for adaptive BRL')
    parser.add_argument('--brl_background_weight', type=float, default=1.0)
    parser.add_argument('--brl_pseudo_weight', type=float, default=0.25)
    parser.add_argument('--brl_warmup_epochs', type=int, default=1)
    parser.add_argument('--brl_ramp_epochs', type=int, default=3)
    parser.add_argument('--brl_min_views', type=int, default=2)
    parser.add_argument('--brl_max_pseudo_ratio', type=float, default=1.0,
                        help='multiplier on the propensity-derived missing-positive budget')
    args = parser.parse_args()

    main(args)
