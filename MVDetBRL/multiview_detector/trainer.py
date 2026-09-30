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


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, criterion, logdir, denormalize, cls_thres=0.4, alpha=1.0,
                 pseudo_loss_weight=0.01, amp=False, query_criterion=None,
                 query_loss_weight=0.0, consistency_criterion=None,
                 consistency_loss_weight=0.0, camera_drop_prob=0.0,
                 consistency_ramp_epochs=5, nms_radius_grid=20.0,
                 query_warmup_epochs=3, query_ramp_epochs=5):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = denormalize
        self.alpha = alpha
        self.pseudo_loss_weight = float(pseudo_loss_weight)
        self.amp = bool(amp)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.query_criterion = query_criterion
        self.query_loss_weight = float(query_loss_weight)
        self.query_warmup_epochs = max(int(query_warmup_epochs), 0)
        self.query_ramp_epochs = max(int(query_ramp_epochs), 1)
        self.consistency_criterion = consistency_criterion
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.camera_drop_prob = float(camera_drop_prob)
        self.consistency_ramp_epochs = max(int(consistency_ramp_epochs), 1)
        self.nms_radius_grid = float(nms_radius_grid)
        if not 0.0 <= self.camera_drop_prob < 1.0:
            raise ValueError('camera_drop_prob must be in [0,1)')

    def _device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _sample_camera_mask(batch, num_views, drop_probability, device):
        keep = torch.rand(batch, num_views, device=device) >= drop_probability
        empty = (~keep).all(dim=1).nonzero(as_tuple=False).flatten()
        if empty.numel() > 0:
            replacement = torch.randint(num_views, (empty.numel(),), device=device)
            keep[empty, replacement] = True
        # When consistency is requested, make every sample a genuine camera
        # perturbation rather than occasionally duplicating the full-view pass.
        if num_views > 1 and drop_probability > 0:
            full = keep.all(dim=1).nonzero(as_tuple=False).flatten()
            if full.numel() > 0:
                removed = torch.randint(num_views, (full.numel(),), device=device)
                keep[full, removed] = False
        return keep

    def _loss_components(self, map_res, map_gt, imgs_res, imgs_gt, dataset,
                         pseudo_target=None, pseudo_weight=None):
        base_loss = self.criterion(map_res, map_gt.to(map_res.device), dataset.map_kernel)
        view_loss = map_res.new_zeros(())
        for img_res, img_gt in zip(imgs_res, imgs_gt):
            view_loss = view_loss + self.criterion(img_res, img_gt.to(img_res.device), dataset.img_kernel)
        base_loss = base_loss + view_loss / len(imgs_gt) * self.alpha
        pseudo_loss = map_res.new_zeros(())
        if pseudo_target is not None and self.pseudo_loss_weight > 0:
            target = pseudo_target.to(map_res.device, dtype=map_res.dtype)
            weights = pseudo_weight.to(map_res.device, dtype=map_res.dtype)
            if target.shape[-2:] != map_res.shape[-2:]:
                target = F.interpolate(target, size=map_res.shape[-2:], mode='bilinear', align_corners=False)
                weights = F.interpolate(weights, size=map_res.shape[-2:], mode='bilinear', align_corners=False)
            weighted_error = weights * (map_res - target).pow(2)
            active_count = (weights > 0).sum().clamp(min=1).to(map_res.dtype)
            pseudo_loss = self.pseudo_loss_weight * weighted_error.sum() / active_count
        return base_loss, pseudo_loss

    def _loss(self, map_res, map_gt, imgs_res, imgs_gt, dataset, pseudo_target=None, pseudo_weight=None):
        base_loss, pseudo_loss = self._loss_components(
            map_res, map_gt, imgs_res, imgs_gt, dataset, pseudo_target, pseudo_weight)
        return base_loss + pseudo_loss

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
        self.model.train()
        losses = 0
        base_losses = 0
        pseudo_losses = 0
        query_losses = 0
        consistency_losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        for batch_idx, batch in enumerate(data_loader):
            data, map_gt, imgs_gt, _ = batch[:4]
            pseudo_target, pseudo_weight = batch[4:6] if len(batch) >= 6 else (None, None)
            data = data.to(self._device(), non_blocking=True)
            optimizer.zero_grad()
            use_consistency = (
                self.consistency_criterion is not None
                and self.consistency_loss_weight > 0
                and self.camera_drop_prob > 0
            )
            teacher_map = None
            camera_mask = None
            if use_consistency:
                # Same-weight full-view teacher. Evaluation mode avoids a
                # second BatchNorm update; no_grad means only the dropped-view
                # student graph is retained for backward.
                self.model.eval()
                with torch.no_grad(), torch.autocast(
                    device_type='cuda', dtype=torch.float16, enabled=self.amp
                ):
                    teacher_map, _ = self.model(data)
                self.model.train()
                camera_mask = self._sample_camera_mask(
                    data.shape[0], data.shape[1], self.camera_drop_prob, data.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.amp):
                if self.query_criterion is not None and self.query_loss_weight > 0:
                    map_res, imgs_res, auxiliary = self.model(
                        data, return_aux=True, camera_mask=camera_mask)
                else:
                    if camera_mask is None:
                        map_res, imgs_res = self.model(data)
                    else:
                        map_res, imgs_res = self.model(data, camera_mask=camera_mask)
                    auxiliary = None
                t_f = time.time()
                t_forward += t_f - t_b
                base_loss, pseudo_loss = self._loss_components(
                    map_res, map_gt, imgs_res, imgs_gt, data_loader.dataset,
                    pseudo_target, pseudo_weight)
                query_loss = map_res.new_zeros(())
                if auxiliary is not None:
                    base_model = self.model.module if isinstance(
                        self.model, torch.nn.DataParallel) else self.model
                    query_progress = max(epoch - self.query_warmup_epochs, 0)
                    query_ramp = min(query_progress / self.query_ramp_epochs, 1.0)
                    if query_ramp > 0:
                        query_loss = (
                            self.query_loss_weight * query_ramp
                            * self.query_criterion(
                                auxiliary['queries'], map_gt, base_model.bev_xy_m)
                        )
                consistency_loss = map_res.new_zeros(())
                if use_consistency:
                    ramp = min(float(epoch) / self.consistency_ramp_epochs, 1.0)
                    consistency_loss = (
                        self.consistency_loss_weight * ramp
                        * self.consistency_criterion(map_res, teacher_map)
                    )
                loss = base_loss + pseudo_loss + query_loss + consistency_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            losses += loss.item()
            base_losses += base_loss.item()
            pseudo_losses += pseudo_loss.item()
            query_losses += query_loss.item()
            consistency_losses += consistency_loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
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
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
                      'Precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.1f} '
                      '(f{:.3f}+b{:.3f}), base: {:.6f}, pseudo: {:.6f}, '
                      'query: {:.6f}, consistency: {:.6f}, maxima: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1),
                    precision_s.avg * 100, recall_s.avg * 100,
                    t_epoch, t_forward / (batch_idx + 1), t_backward / (batch_idx + 1),
                    base_losses / (batch_idx + 1), pseudo_losses / (batch_idx + 1),
                    query_losses / (batch_idx + 1), consistency_losses / (batch_idx + 1),
                    map_res.max()))
                pass

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, '
              'Precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}, '
              'base: {:.6f}, pseudo: {:.6f}, query: {:.6f}, consistency: {:.6f}'.format(
            epoch, len(data_loader), losses / len(data_loader),
            precision_s.avg * 100, recall_s.avg * 100, t_epoch,
            base_losses / len(data_loader), pseudo_losses / len(data_loader),
            query_losses / len(data_loader), consistency_losses / len(data_loader)))

        return losses / len(data_loader), precision_s.avg * 100

    def test(self, data_loader, res_fpath=None, gt_fpath=None, visualize=False,
             score_cache_path=None):
        self.model.eval()
        losses = 0
        precision_s, recall_s = AverageMeter(), AverageMeter()
        all_res_list = []
        evaluated_frames = []
        cached_maps = []
        t0 = time.time()
        if res_fpath is not None:
            assert gt_fpath is not None
        for batch_idx, batch in enumerate(data_loader):
            data, map_gt, imgs_gt, frame = batch[:4]
            evaluated_frames.extend(int(value) for value in frame.reshape(-1).tolist())
            data = data.to(self._device(), non_blocking=True)
            with torch.no_grad():
                map_res, imgs_res = self.model(data)
            if score_cache_path is not None:
                cached_maps.append(map_res.detach().float().cpu().numpy()[:, 0])
            if res_fpath is not None:
                batch_maps = map_res.detach().cpu()
                for sample_index in range(batch_maps.shape[0]):
                    map_grid_res = batch_maps[sample_index, 0]
                    selected = map_grid_res > self.cls_thres
                    v_s = map_grid_res[selected].unsqueeze(1)
                    grid_ij = selected.nonzero()
                    if data_loader.dataset.base.indexing == 'xy':
                        grid_xy = grid_ij[:, [1, 0]]
                    else:
                        grid_xy = grid_ij
                    frame_column = torch.full_like(
                        v_s, int(frame.reshape(-1)[sample_index].item())
                    )
                    all_res_list.append(torch.cat([
                        frame_column,
                        grid_xy.float() * data_loader.dataset.grid_reduce,
                        v_s,
                    ], dim=1))

            loss = self._loss(map_res, map_gt, imgs_res, imgs_gt, data_loader.dataset)
            losses += loss.item()
            pred = (map_res > self.cls_thres).int().to(map_gt.device)
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
            # Visualization historically assumed batch_size=1 and squeezed
            # every singleton dimension. With DataParallel batch_size=2 that
            # leaves [B,H,W], which matplotlib cannot display. Visualize one
            # explicit sample/channel while metrics still use the full batch.
            map_visual = map_res[0, 0].detach().float().cpu().numpy()
            target_visual = self.criterion._traget_transform(
                map_res, map_gt, data_loader.dataset.map_kernel
            )[0, 0].detach().float().cpu().numpy()
            subplt0.imshow(map_visual)
            subplt1.imshow(target_visual)
            plt.savefig(os.path.join(self.logdir, 'map.jpg'))
            plt.close(fig)

            # visualizing the heatmap for per-view estimation
            heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            img0 = Image.fromarray((img0 * 255).astype('uint8'))
            head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            head_cam_result.save(os.path.join(self.logdir, 'cam1_head.jpg'))
            foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        moda = 0
        if score_cache_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(score_cache_path)), exist_ok=True)
            np.savez_compressed(
                score_cache_path,
                maps=np.concatenate(cached_maps, axis=0),
                frames=np.asarray(evaluated_frames, dtype=np.int64),
                grid_reduce=np.asarray(data_loader.dataset.grid_reduce),
                indexing=np.asarray(data_loader.dataset.base.indexing),
            )
        if res_fpath is not None:
            if len(all_res_list) == 0:
                all_res = torch.zeros((0, 4))
            else:
                all_res = torch.cat(all_res_list, dim=0)
            np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res.numpy(), '%.8f')
            res_list = []
            if all_res.numel() > 0:
                for frame in np.unique(all_res[:, 0].numpy()):
                    res = all_res[all_res[:, 0] == frame, :]
                    positions, scores = res[:, 1:3], res[:, 3]
                    ids, count = nms(
                        positions, scores, self.nms_radius_grid, np.inf
                    )
                    if count > 0:
                        res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')

            recall, precision, moda, modp = evaluate(
                os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
                data_loader.dataset.base.__name__, frames=evaluated_frames)

            # If you want to use the unofiicial python evaluation tool for convenient purposes.
            # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
            #                                             data_loader.dataset.base.__name__)

            print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                  format(moda, modp, precision, recall))

        print('Test, Loss: {:.6f}, Precision: {:.1f}%, Recall: {:.1f}%, \tTime: {:.3f}'.format(
            losses / len(data_loader), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

        return losses / len(data_loader), precision_s.avg * 100, moda


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
