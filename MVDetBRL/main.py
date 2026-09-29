import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


def build_criterion(args, device):
    if args.loss == 'brl':
        return BRLGaussianMSE(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
            use_confuse=args.pseudo_mode != 'pseudo_only',
            pseudo_thr=args.pseudo_thr,
            lambda_pseudo=args.lambda_pseudo,
            pseudo_aggregation=args.pseudo_aggregation,
        ).to(device)
    return GaussianMSE().to(device)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_model_state(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    unwrap_model(model).load_state_dict(state)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for MVDet training')
    device_ids = [int(value.strip()) for value in args.device_ids.split(',') if value.strip()]
    if not device_ids:
        raise ValueError('--device_ids must contain at least one logical CUDA device id')
    available_devices = torch.cuda.device_count()
    if max(device_ids) >= available_devices:
        raise ValueError(
            f'--device_ids={args.device_ids} requests a logical CUDA device that is unavailable; '
            f'visible CUDA device count is {available_devices}'
        )
    primary_device = torch.device(f'cuda:{device_ids[0]}')
    torch.cuda.set_device(primary_device)

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
    
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '../Data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '../Data/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    gt_fpath = args.gt_fpath
    if gt_fpath is None and args.data_path is not None:
        gt_fpath = os.path.abspath(os.path.join('cache', f'{args.dataset}_gt.txt'))
    if args.pseudo_mode != 'none':
        if args.loss != 'brl':
            raise ValueError('External pseudo labels require --loss brl')
        if not args.pseudo_dir:
            raise ValueError('--pseudo_dir is required when --pseudo_mode is enabled')

    train_set = frameDataset(
        base,
        train=True,
        transform=train_trans,
        grid_reduce=4,
        drop_ratio=args.drop_ratio,
        pseudo_dir=args.pseudo_dir if args.pseudo_mode != 'none' else None,
        gt_fpath=gt_fpath,
    )
    # Validation/test always uses complete annotations and no pseudo supervision.
    test_set = frameDataset(
        base,
        train=False,
        transform=train_trans,
        grid_reduce=4,
        drop_ratio=0,
        force_download=False,
        gt_fpath=gt_fpath,
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    # The legacy evaluator builds one frame id per BEV map and therefore expects
    # batch size 1. Training can still use a larger global batch across GPUs.
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
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
    model = model.to(primary_device)
    if args.load is not None:
        load_model_state(model, args.load, primary_device)
        print(f'{args.load} loaded')
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=device_ids, output_device=device_ids[0]
        )
        print(f'DataParallel enabled on logical CUDA devices {device_ids}')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = build_criterion(args, primary_device)

    # logging
    drop_tag = f'drop_{args.drop_ratio}' if args.drop_ratio > 0 else 'full'
    if args.loss == 'brl':
        loss_tag = f'brl_b{args.brl_pos_thr}_c{args.brl_confuse_thr}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
        if args.pseudo_mode != 'none':
            loss_tag += f'_{args.pseudo_mode}_lp{args.lambda_pseudo}_agg{args.pseudo_aggregation}'
    else:
        loss_tag = 'mse'
    logdir = args.logdir
    if logdir is None:
        stamp = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        base_log = f'logs/{args.dataset}_frame/{drop_tag}/{loss_tag}/{args.variant}'
        if args.loginfo:
            base_log = f'{base_log}/{args.loginfo}'
        logdir = f'{base_log}/{stamp}' if not args.resume else f'{base_log}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
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

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)

    # learn
    if args.resume is None:
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, False)

        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            print('Testing...')
            test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                      train_set.gt_fpath, False)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       test_loss_s, test_prec_s, test_moda_s)
            # save
            torch.save(unwrap_model(model).state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_dir = f'logs/{args.dataset}_frame/{drop_tag}/{loss_tag}/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        load_model_state(model, resume_fname, primary_device)
        model.eval()
    print('Test loaded model...')
    trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, False)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector + BRL heatmap loss')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--data_path', type=str, default=None,
                        help='dataset root override; defaults to ../Data/<dataset>')
    parser.add_argument('--gt_fpath', type=str, default=None,
                        help='optional writable path for the complete evaluation GT cache')
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
    parser.add_argument('--loginfo', type=str, default='')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--device_ids', type=str, default='0',
                        help='logical CUDA ids visible to this process, e.g. 0 or 0,1')
    
    # Dropped annotations
    parser.add_argument('--drop_ratio', type=int, default=0,
                        choices=[0, 20, 45, 60],
                        help='0 = full labels; 20/45/60 = drop_annotations/drop_XX')
    # BRL heatmap loss
    parser.add_argument('--loss', type=str, default='brl', choices=['brl', 'mse'],
                        help='brl = Background Recalibration heatmap loss; mse = original GaussianMSE')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='soft-GT threshold for positive pixels')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pred threshold on background to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='if set, down-weight bg MSE on confuse instead of mirroring toward 1')
    parser.add_argument('--pseudo_mode', type=str, default='none',
                        choices=['none', 'pseudo_only', 'pseudo_confuse'],
                        help='pseudo_only disables self-confuse; pseudo_confuse keeps both')
    parser.add_argument('--pseudo_dir', type=str, default=None,
                        help='directory containing per-frame BEV pseudo-label JSON files')
    parser.add_argument('--pseudo_thr', type=float, default=0.1,
                        help='Gaussian pseudo heatmap threshold used to define pseudo-positive pixels')
    parser.add_argument('--lambda_pseudo', type=float, default=0.1,
                        help='weight of the separately normalized pseudo-label loss')
    parser.add_argument('--pseudo_aggregation', type=str, default='sum', choices=['sum', 'max'],
                        help='combine projected pseudo Gaussian maps by addition or pixel-wise maximum')
    args = parser.parse_args()

    main(args)
