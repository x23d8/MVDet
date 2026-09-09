import time
import torch
import os
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image


_DEFAULT_VIEW_CRITERION = object()


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0,
                 view_criterion=_DEFAULT_VIEW_CRITERION):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.view_criterion = (
            criterion if view_criterion is _DEFAULT_VIEW_CRITERION else view_criterion
        )
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.last_train_metrics = {}
        self.last_test_metrics = {}
        self.last_detection_threshold_metrics = {}

    def _map_probability(self, map_result):
        if getattr(self.criterion, 'outputs_logits', False):
            return torch.sigmoid(map_result)
        return map_result

    def _view_probability(self, view_result):
        if getattr(self.view_criterion, 'outputs_logits', False):
            return torch.sigmoid(view_result)
        return view_result

    def _compute_loss(self, map_result, images_result, map_gt, images_gt, dataset):
        map_target = map_gt.to(map_result.device)
        core_model = getattr(self.model, 'module', self.model)
        projected_scores = None
        projected_visibility = None

        if getattr(self.criterion, 'requires_multiview_context', False):
            projected_scores, projected_visibility = self.criterion.project_view_evidence(
                images_result,
                core_model.proj_mats,
                map_result.shape[-2:],
            )
            map_loss = self.criterion(
                map_result,
                map_target,
                dataset.map_kernel,
                projected_view_scores=projected_scores,
                projected_visibility=projected_visibility,
            )
        else:
            map_loss = self.criterion(map_result, map_target, dataset.map_kernel)

        if self.view_criterion is None:
            view_loss = map_result.new_zeros(())
        elif getattr(self.view_criterion, 'requires_bev_context', False):
            if projected_scores is None or projected_visibility is None:
                raise RuntimeError('Partial view loss requires a multiview BEV criterion')
            view_loss = self.view_criterion(
                images_result,
                images_gt,
                dataset.img_kernel,
                map_logits=map_result,
                projection_matrices=core_model.proj_mats,
                projected_scores=projected_scores,
                projected_visibility=projected_visibility,
            )
        else:
            view_loss = map_result.new_zeros(())
            for image_result, image_gt in zip(images_result, images_gt):
                view_loss = view_loss + self.view_criterion(
                    image_result,
                    image_gt.to(image_result.device),
                    dataset.img_kernel,
                )
            view_loss = view_loss / max(len(images_gt), 1)

        return map_loss + self.alpha * view_loss, map_loss, view_loss

    def _update_criterion_metrics(self, meters):
        criteria = (('bev', self.criterion), ('view', self.view_criterion))
        for prefix, criterion in criteria:
            if criterion is None:
                continue
            for name, value in getattr(criterion, 'last_stats', {}).items():
                meters.setdefault(f'{prefix}/{name}', AverageMeter()).update(value)

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None,
              grad_clip_norm=0.0):
        self.model.train()
        for criterion in (self.criterion, self.view_criterion):
            if hasattr(criterion, 'set_epoch'):
                criterion.set_epoch(epoch)
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        criterion_metrics = {}
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, (data, map_gt, imgs_gt, _) in enumerate(data_loader):
            optimizer.zero_grad()
            map_res, imgs_res = self.model(data)
            t_f = time.time()
            t_forward += t_f - t_b
            loss, _, _ = self._compute_loss(
                map_res, imgs_res, map_gt, imgs_gt, data_loader.dataset
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'Non-finite loss at epoch {epoch}, batch {batch_idx + 1}; '
                    'reduce the learning rate or inspect the loss masks'
                )
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            optimizer.step()
            losses += loss.item()
            self._update_criterion_metrics(criterion_metrics)
            map_probability = self._map_probability(map_res)
            pred = (map_probability > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

            t_b = time.time()
            t_backward += t_b - t_f

            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
                      'prec: {:.1f}%, recall: {:.1f}%, Time: {:.1f} (f{:.3f}+b{:.3f}), maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1),
                    map_probability.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        self.last_train_metrics = {
            'loss': losses / len(data_loader),
            'precision_percent': precision_s.avg * 100,
            'recall_percent': recall_s.avg * 100,
            'duration_seconds': t_epoch,
        }
        self.last_train_metrics.update({name: meter.avg for name, meter in criterion_metrics.items()})

        return losses / len(data_loader), precision_s.avg * 100

    @staticmethod
    def _threshold_result_path(res_fpath, threshold):
        stem, extension = os.path.splitext(res_fpath)
        threshold_label = f'{threshold:.2f}'.replace('.', 'p')
        return f'{stem}_threshold_{threshold_label}{extension}'

    @staticmethod
    def _evaluate_candidates(all_candidates, threshold, res_fpath, gt_fpath, dataset_name):
        selected = all_candidates[all_candidates[:, 3] > threshold]
        res_list = []
        for frame in np.unique(selected[:, 0]):
            res = selected[selected[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, 20, np.inf)
            res_list.append(torch.cat([
                torch.ones([count, 1]) * frame,
                positions[ids[:count], :],
            ], dim=1))
        detections = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(res_fpath, detections, '%d')
        recall, precision, moda, modp = evaluate(
            os.path.abspath(res_fpath),
            os.path.abspath(gt_fpath),
            dataset_name,
        )
        return {
            'threshold': float(threshold),
            'moda_percent': float(moda),
            'modp_percent': float(modp),
            'detection_precision_percent': float(precision),
            'detection_recall_percent': float(recall),
            'num_detections': int(detections.shape[0]),
        }

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False,
             detection_thresholds=None):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        criterion_metrics = {}
        all_res_list = []
        thresholds = [float(self.cls_thres)]
        if detection_thresholds is not None:
            thresholds.extend(float(threshold) for threshold in detection_thresholds)
        thresholds = sorted(set(thresholds))
        if any(not 0 < threshold < 1 for threshold in thresholds):
            raise ValueError('detection thresholds must be between zero and one')
        candidate_threshold = min(thresholds)
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
            with torch.no_grad():
                map_res, imgs_res = self.model(data)
                map_probability = self._map_probability(map_res)
            if res_fpath is not None:
                map_grid_res = map_probability.detach().cpu().squeeze()
                v_s = map_grid_res[map_grid_res > candidate_threshold].unsqueeze(1)
                grid_ij = (map_grid_res > candidate_threshold).nonzero()
                if data_loader.dataset.base.indexing == 'xy':
                    grid_xy = grid_ij[:, [1, 0]]
                else:
                    grid_xy = grid_ij
                all_res_list.append(torch.cat([torch.ones_like(v_s) * frame, grid_xy.float() *
                                               data_loader.dataset.grid_reduce, v_s], dim=1))

            loss, _, _ = self._compute_loss(
                map_res, imgs_res, map_gt, imgs_gt, data_loader.dataset
            )
            losses += loss.item()
            self._update_criterion_metrics(criterion_metrics)
            pred = (map_probability > self.cls_thres).int().to(map_gt.device)
            true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
            false_positive = pred.sum().item() - true_positive
            false_negative = map_gt.sum().item() - true_positive
            precision = true_positive / (true_positive + false_positive + 1e-4)
            recall = true_positive / (true_positive + false_negative + 1e-4)
            precision_s.update(precision)
            recall_s.update(recall)

        t1 = time.time()
        t_epoch = t1 - t0

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(211, title="output")
            subplt1 = fig.add_subplot(212, title="target")
            subplt0.imshow(map_probability.cpu().detach().numpy().squeeze())
            subplt1.imshow(self.criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
                           .cpu().detach().numpy().squeeze())
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            view_probability = self._view_probability(imgs_res[0])
            heatmap0_head = view_probability[0, 0].detach().cpu().numpy().squeeze()
            heatmap0_foot = view_probability[0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        moda = 0
        modp = 0
        detection_precision = 0
        detection_recall = 0
        selected_metrics = None
        self.last_detection_threshold_metrics = {}
        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            for threshold in thresholds:
                threshold_path = (
                    res_fpath if threshold == self.cls_thres
                    else self._threshold_result_path(res_fpath, threshold)
                )
                metrics = self._evaluate_candidates(
                    all_res_list,
                    threshold,
                    threshold_path,
                    gt_fpath,
                    data_loader.dataset.base.__name__,
                )
                self.last_detection_threshold_metrics[f'{threshold:.2f}'] = metrics
                print(
                    'threshold: {:.2f}, moda: {:.1f}%, modp: {:.1f}%, '
                    'precision: {:.1f}%, recall: {:.1f}%, detections: {}'.format(
                        threshold,
                        metrics['moda_percent'],
                        metrics['modp_percent'],
                        metrics['detection_precision_percent'],
                        metrics['detection_recall_percent'],
                        metrics['num_detections'],
                    )
                )

            primary_metrics = self.last_detection_threshold_metrics[f'{self.cls_thres:.2f}']
            moda = primary_metrics['moda_percent']
            modp = primary_metrics['modp_percent']
            detection_precision = primary_metrics['detection_precision_percent']
            detection_recall = primary_metrics['detection_recall_percent']
            selected_metrics = max(
                self.last_detection_threshold_metrics.values(),
                key=lambda metrics: (
                    metrics['moda_percent'],
                    metrics['detection_precision_percent'],
                    -metrics['threshold'],
                ),
            )

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}, \tTime: {:.3f}'.format(
            losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        self.last_test_metrics = {
            'loss': losses / len(data_loader),
            'grid_precision_percent': precision_s.avg * 100,
            'grid_recall_percent': recall_s.avg * 100,
            'moda_percent': moda,
            'modp_percent': modp,
            'detection_precision_percent': detection_precision,
            'detection_recall_percent': detection_recall,
            'selected_cls_threshold': (
                selected_metrics['threshold'] if selected_metrics is not None else self.cls_thres
            ),
            'selected_moda_percent': (
                selected_metrics['moda_percent'] if selected_metrics is not None else moda
            ),
            'selected_modp_percent': (
                selected_metrics['modp_percent'] if selected_metrics is not None else modp
            ),
            'selected_detection_precision_percent': (
                selected_metrics['detection_precision_percent']
                if selected_metrics is not None else detection_precision
            ),
            'selected_detection_recall_percent': (
                selected_metrics['detection_recall_percent']
                if selected_metrics is not None else detection_recall
            ),
            'duration_seconds': t_epoch,
        }
        self.last_test_metrics.update({name: meter.avg for name, meter in criterion_metrics.items()})

        reported_moda = self.last_test_metrics['selected_moda_percent']
        return losses / len(data_loader), precision_s.avg * 100, reported_moda


class BBOXTrainer(BaseTrainer):
    def __init__(self, model, criterion, cls_thres):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (data, target, _) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = self.model(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if cyclic_scheduler is not None:
                if isinstance(cyclic_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(cyclic_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    cyclic_scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader, log_interval=100, res_fpath=None):
        self.model.eval()
        losses = 0
        correct = 0
        miss = 0
        all_res_list = []
        t0 = time.time()
        for batch_idx, (data, target, (frame, pid, grid_x, grid_y)) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(data)
                output = F.softmax(output, dim=1)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.numel() - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()
            if res_fpath is not None:
                indices = output[:, 1] > self.cls_thres
                all_res_list.append(torch.stack([frame[indices].float(), grid_x[indices].float(),
                                                 grid_y[indices].float(), output[indices, 1].cpu()], dim=1))
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Test Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            len(test_loader), losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if res_fpath is not None:
            all_res_list = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.dirname(res_fpath) + '/all_res.txt', all_res_list.numpy(), '%.8f')
            res_list = []
            for frame in np.unique(all_res_list[:, 0]):
                res = all_res_list[all_res_list[:, 0] == frame, :]
                positions, scores = res[:, 1:3], res[:, 3]
                ids, count = nms(positions, scores, )
                res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy()
            np.savetxt(res_fpath, res_list, '%d')

        return losses / len(test_loader), correct / (correct + miss)
