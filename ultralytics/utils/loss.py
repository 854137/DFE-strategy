# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, ATSSAssigner,IoUAssigner,dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from . import LOGGER

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing
    on hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # ctrl + bbox_iou 进入metrics文件中的bbox_iou，里面有计算IoU的各种方法

        # WIoU
        # 版本二
        # 原代码是CIoU
        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, WIoU=True)
        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, Focal=False)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True,Focal=False)
        #  消融实验：EIoU,GIoU,DIoU,SIoU,Focal
        if type(iou) is tuple:
            if len(iou) == 2:
                # scale为False或者Focal为True
                loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
            else:
                # scale为True并且Focal为False
                loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
        else:
            # Focal为False
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum



        # DFL loss
        eps = 1e-8
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() /(target_scores_sum + eps)
            # 有改动：分母+eps
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class QualityFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred: [N, C] or [B, HW, C], raw logits (before sigmoid)
        target: [N, C] or [B, HW, C], soft labels in [0, 1] (e.g., IoU for positive, 0 for negative)
        """
        pred_sigmoid = pred.sigmoid()
        # 防止 log(0)
        pred_sigmoid = torch.clamp(pred_sigmoid, min=1e-4, max=1 - 1e-4)

        # 正负样本统一处理
        focal_weight = target * (1 - pred_sigmoid) ** self.gamma + \
                       (1 - target) * pred_sigmoid ** self.gamma

        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none'
        ) * focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        # 这里传入的model = DetectionModel(cfg)
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        # model.

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.qfl = QualityFocalLoss(gamma=2.0, reduction="none")
        # QFL用于替代分类损失函数的BCE
        # __call__中自动调用计算损失函数
        self.hyp = h
        # self.hyp存放超参数
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        # 任务对齐
        # ATSS策略
        # self.assigner = ATSSAssigner(topk=tal_topk, num_classes=self.nc)
        # 无对齐的策略
        # self.assigner = IoUAssigner(iou_threshold=0.5, num_classes=self.nc)
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2
        # 可能缓解分类和定位的矛盾，实际并没有dfl

        # TaskAlignedAssigner
        #
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            # # 可能缓解分类和定位的矛盾，实际并没有dfl
            pred_scores.detach().sigmoid(),

            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # # START
        # # QFL
        # iou_for_qfl = None  # 预定义避免未定义错误
        # # 确保 fg_mask 是2D布尔张量
        # if fg_mask.dim() > 2:
        #     fg_mask = fg_mask.squeeze(-1)
        #
        # if fg_mask.sum() > 0:
        #     # 提取正样本并确保是2D [num_pos, 4]
        #     pos_pred_bboxes = pred_bboxes[fg_mask].detach()
        #     pos_target_bboxes = target_bboxes[fg_mask]
        #     # 如果是3D张量，压缩维度
        #     if pos_pred_bboxes.dim() == 3:
        #         pos_pred_bboxes = pos_pred_bboxes.squeeze(1)
        #     if pos_target_bboxes.dim() == 3:
        #         pos_target_bboxes = pos_target_bboxes.squeeze(1)
        #     # 计算IoU - 确保返回1D张量
        #     iou_for_qfl = bbox_iou(
        #         pos_pred_bboxes,
        #         pos_target_bboxes,
        #         xywh=False,
        #         CIoU=False
        #     ).clamp(min=0.0, max=1.0)
        #     # 处理所有可能的异常形状
        #     if iou_for_qfl is not None:
        #         # 如果是3D且最后一维是1，压缩它
        #         if iou_for_qfl.dim() == 3 and iou_for_qfl.shape[-1] == 1:
        #             iou_for_qfl = iou_for_qfl.squeeze(-1)
        #         # 如果是3D且形状为 [N, N, 1]，提取对角线
        #         if iou_for_qfl.dim() == 3 and iou_for_qfl.shape[0] == iou_for_qfl.shape[1]:
        #             iou_for_qfl = iou_for_qfl.diagonal(dim1=0, dim2=1)
        #         # 如果是2D方阵，提取对角线
        #         if iou_for_qfl.dim() == 2 and iou_for_qfl.shape[0] == iou_for_qfl.shape[1]:
        #             iou_for_qfl = iou_for_qfl.diag()
        #         # 确保是1D张量
        #         iou_for_qfl = iou_for_qfl.view(-1)
        #         # 检查正样本数量是否匹配
        #         num_pos = fg_mask.sum().item()
        #         if iou_for_qfl.shape[0] != num_pos:
        #             print(f"警告: IoU数量({iou_for_qfl.shape[0]})与正样本数({num_pos})不匹配，使用平均值")
        #             iou_for_qfl = iou_for_qfl.mean().expand(num_pos)
        # # 安全赋值：只在有正样本时执行
        # if iou_for_qfl is not None:
        #     # 确保形状为 [num_pos, 1] 用于广播
        #     iou_for_qfl = iou_for_qfl.view(-1, 1)
        #     # target_scores[fg_mask] 形状: [num_pos, num_classes]
        #     target_scores[fg_mask] = target_scores[fg_mask] * iou_for_qfl
        # # END

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # loss[1] = self.qfl(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum # QFL

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # return loss_iou, loss_dfl
            # self.dfl_loss = DFLoss(reg_max)
            # """Compute IoU and DFL losses for bounding boxes."""
            # weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

            # target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            # loss_dfl = loss_dfl.sum() / target_scores_sum

            # 从这里可以看出，任务对齐体现在target_score上了：
            # _, target_bboxes, target_scores, fg_mask, _ = self.assigner(

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # 从超参数预定义中读取增益

        # 在basetrainer中的_do_train中有self.loss = loss.sum()
        # loss.sum = lamda1 * clsloss + lamda2 * dfl + lamda * cIoU

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)



class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()




class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]




# 下面是现加的
# 在 loss.py 中替换原有的 RetinaNetLoss 类

# ultralytics/utils/loss.py 中新增或替换

# ultralytics/utils/loss.py
#
# class RetinaNetLoss(nn.Module):
#     """RetinaNet loss using existing YOLOv8 modules for classification and regression."""
#
#     def __init__(self, model_or_hyp):
#         """
#         Initialize RetinaNetLoss with flexible input handling.
#
#         Args:
#             model_or_hyp: Can be a YOLO model instance, RetinaNetDetect instance, or hyperparameter dict.
#         """
#         super().__init__()
#
#         # ✅ 参数提取：确保 self.hyp 总是 dict 类型
#         if hasattr(model_or_hyp, 'args'):
#             # 标准 YOLO 模型（带 args 属性）
#             self.hyp = model_or_hyp.args if isinstance(model_or_hyp.args, dict) else {}
#             self.nc = getattr(model_or_hyp, 'nc', 80)
#             self.reg_max = getattr(model_or_hyp, 'reg_max', 4)
#             # ✅ 关键：从模型参数推断设备
#             try:
#                 self.device = next(model_or_hyp.parameters()).device
#             except StopIteration:
#                 self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         elif isinstance(model_or_hyp, dict):
#             # 直接的超参数字典
#             self.hyp = model_or_hyp
#             self.device = self.hyp.get('device', 'cuda')
#             self.nc = self.hyp.get('nc', 80)
#             self.reg_max = self.hyp.get('reg_max', 4)
#
#         else:
#             # ✅ 自定义模型（如 RetinaNetDetect）
#             self.hyp = {}  # 强制为 dict
#             self.nc = getattr(model_or_hyp, 'nc', 80)
#             self.reg_max = getattr(model_or_hyp, 'reg_max', 4)
#             # ✅ 关键：从模型参数推断设备
#             try:
#                 self.device = next(model_or_hyp.parameters()).device
#             except StopIteration:
#                 self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # ✅ 现在可以安全使用 .get() 方法
#         self.pos_iou_thr = self.hyp.get('pos_iou_thr', 0.5)
#         self.neg_iou_thr = self.hyp.get('neg_iou_thr', 0.4)
#         self.alpha = self.hyp.get('alpha', 0.25)
#         self.gamma = self.hyp.get('gamma', 2.0)
#
#         # ✅ 损失权重（提供默认值）
#         self.l1_loss_scale = self.hyp.get('l1_loss_scale', 0.1)
#         self.cls_loss_scale = self.hyp.get('cls_loss_scale', 1.0)
#         self.iou_loss_scale = self.hyp.get('iou_loss_scale', 1.0)
#
#         # ✅ 复用 YOLOv8 现有损失模块
#         self.focal_loss = FocalLoss(gamma=self.gamma, alpha=self.alpha)
#         self.l1_loss = nn.L1Loss(reduction='none')
#         self.bce_cls = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
#
#         # ✅ 修复：初始化这些属性
#         self.anchors_retinanet = None
#         self.strides_retinanet = None
#
#         # ✅ 关键：添加 num_anchors 属性
#         self.num_anchors = 9  # RetinaNet 默认每个位置9个anchor
#
#         # ✅ 确保 reg_max 正确传递
#         self.reg_max = getattr(self, 'reg_max', 4)
#
#         # ✅ 动态导入以获取模型参数
#         from ultralytics.nn.modules.retinanet import RetinaNetDetect
#
#         # 从模型对象中提取参数（如果有）
#         for m in model_or_hyp.modules():
#             if isinstance(m, RetinaNetDetect):
#                 self.nc = m.nc
#                 self.reg_max = m.reg_max
#                 self.num_anchors = m.num_anchors
#                 # 从检测头获取stride
#                 if hasattr(m, 'stride'):
#                     self.strides_retinanet = m.stride
#                 break
#
#         # 如果未从模型获取到stride，使用默认值
#                 # ✅ 添加strides（如果还没有的话）
#             if not hasattr(self, 'strides_retinanet') or self.strides_retinanet is None:
#                 self.strides_retinanet = torch.tensor([8, 16, 32, 64, 128], dtype=torch.float32)
#
#
#     def __call__(self, preds: list, batch: dict) -> tuple:
#         """
#         Compute RetinaNet loss for training.
#
#         Args:
#             preds: List of tensors from RetinaNetDetect, each (b, no, h*w)
#             batch: YOLOv8 batch dict with keys: img, batch_idx, cls, bboxes
#
#         Returns:
#             tuple: (total_loss, loss_items)
#         """
#         # ✅ 重组预测值
#         all_reg, all_cls = [], []
#         reg_channels = self.reg_max * 4
#         batch_size = preds[0].shape[0]
#         device = preds[0].device
#
#         # ✅ 修复：从图像尺寸和stride推断特征图尺寸
#         img_h, img_w = batch['img'].shape[2:]
#         level_shapes = []
#         for i, stride in enumerate(self.strides_retinanet):
#             h = int(img_h / stride.item())
#             w = int(img_w / stride.item())
#             level_shapes.append((h, w))
#
#         # ✅ 修复：为每一层生成anchors（不调用make_anchors）
#         anchors_list = []
#         for i, (h, w) in enumerate(level_shapes):
#             stride = self.strides_retinanet[i].item()
#             anchors_i = self._generate_anchors_single_level(h, w, stride, device)
#             anchors_list.append(anchors_i)
#
#         # 合并所有anchors
#         self.anchors_retinanet = torch.cat(anchors_list, dim=0)
#
#         # ✅ 处理预测值（原有逻辑）
#         total_anchors = 0
#         for i, pred in enumerate(preds):
#             b, c, hw = pred.shape
#             total_anchors += hw
#
#             reg = pred[:, :reg_channels, :].permute(0, 2, 1)  # (b, h*w, reg_channels)
#             cls = pred[:, reg_channels:, :].permute(0, 2, 1)  # (b, h*w, nc)
#
#             all_reg.append(reg)
#             all_cls.append(cls)
#
#         pred_reg = torch.cat(all_reg, dim=1)  # (b, total_anchors, reg_channels)
#         pred_cls = torch.cat(all_cls, dim=1)  # (b, total_anchors, nc)
#
#         # ✅ 重组标注数据
#         targets = torch.cat([
#             batch['batch_idx'].view(-1, 1).float(),
#             batch['cls'].view(-1, 1).float(),
#             batch['bboxes']
#         ], dim=1).to(device)
#
#         # ✅ 计算损失
#         loss_dict = self._compute_loss(pred_reg, pred_cls, targets, self.anchors_retinanet, self.nc)
#
#         # ✅ 返回YOLOv8格式
#         total_loss = loss_dict['tot_loss']
#         loss_items = torch.tensor([
#             loss_dict['l1_loss'].item(),
#             loss_dict['cls_loss'].item(),
#             loss_dict['iou_loss'].item()
#         ], device=device)
#
#         return total_loss, loss_items
#
#     def _generate_anchors_single_level(self, h, w, stride, device):
#         """为单个FPN层生成anchors"""
#         y, x = torch.meshgrid(
#             torch.arange(h, device=device),
#             torch.arange(w, device=device),
#             indexing='ij'
#         )
#
#         # 网格中心坐标
#         xc = (x + 0.5) * stride
#         yc = (y + 0.5) * stride
#
#         # RetinaNet anchor配置
#         ratios = [0.5, 1.0, 2.0]
#         scales = [1.0, 1.25, 1.6]
#         base_size = stride * 4
#
#         anchors = []
#         for r in ratios:
#             for s in scales:
#                 anchor_w = base_size * s * math.sqrt(r)
#                 anchor_h = base_size * s / math.sqrt(r)
#
#                 x1 = xc - anchor_w / 2
#                 y1 = yc - anchor_h / 2
#                 x2 = xc + anchor_w / 2
#                 y2 = yc + anchor_h / 2
#
#                 anchor_box = torch.stack([x1, y1, x2, y2], dim=-1)
#                 anchors.append(anchor_box)
#
#         return torch.cat(anchors, dim=-1).view(-1, 4)
#
#
#     def _generate_anchors(self, preds, batch, total_anchors_expected):
#         """生成 RetinaNet 风格的固定 anchors"""
#         device = preds[0].device
#         anchors = []
#         img_h, img_w = batch['img'].shape[2:]
#
#         # ✅ 修复：直接定义 stride 列表，不依赖动态计算
#         if self.strides_retinanet is None:
#             self.strides_retinanet = torch.tensor([8, 16, 32, 64, 128], device=device, dtype=torch.float32)
#
#         # RetinaNet anchor 配置
#         base_sizes = [32, 64, 128, 256, 512]
#         ratios = [0.5, 1.0, 2.0]
#         scales = [1.0, 1.25, 1.6]
#
#         total_generated = 0
#
#         for i, pred in enumerate(preds):
#             # ✅ 正确获取 stride
#             stride = self.strides_retinanet[i].item()
#
#             # ✅ 正确计算 feature map 尺寸
#             h = img_h // int(stride)
#             w = img_w // int(stride)
#
#             # 创建网格
#             y, x = torch.meshgrid(
#                 torch.arange(h, device=device),
#                 torch.arange(w, device=device),
#                 indexing='ij'
#             )
#
#             # 网格中心坐标
#             xc = (x + 0.5) * stride
#             yc = (y + 0.5) * stride
#
#             level_anchors = []
#             for r in ratios:
#                 for s in scales:
#                     anchor_w = base_sizes[i] * s * math.sqrt(r)
#                     anchor_h = base_sizes[i] * s / math.sqrt(r)
#
#                     x1 = xc - anchor_w / 2
#                     y1 = yc - anchor_h / 2
#                     x2 = xc + anchor_w / 2
#                     y2 = yc + anchor_h / 2
#
#                     anchor_box = torch.stack([x1, y1, x2, y2], dim=-1)
#                     level_anchors.append(anchor_box)
#
#             # ✅ 确保形状正确
#             level_anchors = torch.stack(level_anchors, dim=-1).view(h, w, -1, 4)
#             level_anchors = level_anchors.view(-1, 4)
#             anchors.append(level_anchors)
#             total_generated += level_anchors.shape[0]
#
#         self.anchors_retinanet = torch.cat(anchors, dim=0).to(device)
#
#         # ✅ 验证 anchor 数量
#         if self.anchors_retinanet.shape[0] != total_anchors_expected:
#             # 动态调整
#             if self.anchors_retinanet.shape[0] > total_anchors_expected:
#                 self.anchors_retinanet = self.anchors_retinanet[:total_anchors_expected]
#             else:
#                 pad_size = total_anchors_expected - self.anchors_retinanet.shape[0]
#                 self.anchors_retinanet = torch.cat([
#                     self.anchors_retinanet,
#                     torch.zeros(pad_size, 4, device=device)
#                 ], dim=0)
#
#     def _compute_loss(self, pred_reg, pred_cls, targets, anchors, num_classes):
#         """Inner loss computation with comprehensive safety checks."""
#         batch_size = pred_reg.shape[0]
#         device = pred_reg.device
#         l1_losses, iou_losses, cls_losses = [], [], []
#
#         for b in range(batch_size):
#             # 当前 batch 的标注
#             gt_ann = targets[targets[:, 0] == b]
#
#             if len(gt_ann) == 0:
#                 # 如果没有真值，只计算分类损失（全为负样本）
#                 cls_loss = self.focal_loss(pred_cls[b], torch.zeros_like(pred_cls[b]))
#                 cls_losses.append(cls_loss.sum())
#                 l1_losses.append(torch.tensor(0., device=device))
#                 iou_losses.append(torch.tensor(0., device=device))
#                 continue
#
#             gt_boxes = gt_ann[:, 2:]  # (N_gt, 4)
#             gt_labels = gt_ann[:, 1].long()  # (N_gt,)
#
#             # ✅ 关键修复：确保标签在有效范围内
#             gt_labels = torch.clamp(gt_labels, 0, num_classes - 1)
#
#             # 计算 IoU 矩阵 (N_gt, N_anchors)
#             iou = bbox_iou(gt_boxes.unsqueeze(1), anchors.unsqueeze(0), xywh=False)
#             iou_max, iou_idx = iou.max(dim=0)  # iou_max: (N_anchors,), iou_idx: (N_anchors,)
#
#             # 正负样本分配
#             pos_mask = iou_max >= self.pos_iou_thr  # (N_anchors,)
#             num_pos = pos_mask.sum().item()
#
#             # ✅ 创建分类目标 (N_anchors, num_classes)
#             target_cls = torch.zeros_like(pred_cls[b])
#
#             if num_pos > 0:
#                 # 获取正样本索引
#                 pos_indices = torch.where(pos_mask)[0]  # (num_pos,)
#                 matched_gt_indices = iou_idx[pos_mask]  # (num_pos,)
#                 matched_labels = gt_labels[matched_gt_indices]  # (num_pos,)
#
#                 # ✅ 确保是 1D 张量并检查边界
#                 matched_labels = matched_labels.view(-1)
#                 matched_labels = torch.clamp(matched_labels, 0, num_classes - 1)
#
#                 # ✅ 安全检查：确保索引数量匹配
#                 if len(pos_indices) == len(matched_labels):
#                     # 检查是否有越界索引
#                     if pos_indices.max() < target_cls.shape[0] and matched_labels.max() < num_classes:
#                         target_cls[pos_indices, matched_labels] = 1.0
#                     else:
#                         print(f"WARNING: 越界索引! pos_max={pos_indices.max()}, label_max={matched_labels.max()}, "
#                               f"target_cls.shape={target_cls.shape}")
#                         # 降级处理：只处理有效索引
#                         valid_mask = (pos_indices < target_cls.shape[0]) & (matched_labels < num_classes)
#                         if valid_mask.any():
#                             valid_pos = pos_indices[valid_mask]
#                             valid_labels = matched_labels[valid_mask]
#                             target_cls[valid_pos, valid_labels] = 1.0
#                 else:
#                     print(f"WARNING: 索引数量不匹配! pos={len(pos_indices)}, labels={len(matched_labels)}")
#
#             # ✅ 分类损失（Focal Loss）
#             cls_loss = self.focal_loss(pred_cls[b], target_cls)
#             cls_losses.append(cls_loss.sum() / max(num_pos, 1))
#
#             # ✅ 回归损失（仅正样本）
#             if num_pos > 0:
#                 # DFL 解码
#                 pred_dist = pred_reg[b][pos_mask].view(-1, 4, self.reg_max)
#                 pred_dist = pred_dist.softmax(dim=-1)
#                 proj = torch.arange(self.reg_max, device=device, dtype=torch.float32)
#                 pred_delta = pred_dist.matmul(proj)  # (num_pos, 4)
#
#                 # 解码为 bbox
#                 from ultralytics.utils.ops import xyxy2xywh, xywh2xyxy
#                 anchor_xywh = xyxy2xywh(anchors[pos_mask])  # (num_pos, 4)
#
#                 # RetinaNet 标准解码
#                 pred_xywh = anchor_xywh.clone()
#                 pred_xywh[:, :2] += pred_delta[:, :2] * anchor_xywh[:, 2:] * 0.1
#                 pred_xywh[:, 2:] *= torch.exp(pred_delta[:, 2:] * 0.2)
#                 pred_xyxy = xywh2xyxy(pred_xywh)
#
#                 # IoU 损失
#                 matched_gt = gt_boxes[matched_gt_indices]  # (num_pos, 4)
#                 iou = bbox_iou(pred_xyxy, matched_gt, xywh=False, CIoU=True)
#                 iou_losses.append((1.0 - iou).mean())
#
#                 # L1 损失
#                 gt_xywh = xyxy2xywh(matched_gt)
#                 gt_delta = torch.zeros_like(pred_delta)
#                 gt_delta[:, :2] = (gt_xywh[:, :2] - anchor_xywh[:, :2]) / anchor_xywh[:, 2:] / 0.1
#                 gt_delta[:, 2:] = torch.log(gt_xywh[:, 2:] / anchor_xywh[:, 2:]) / 0.2
#
#                 l1_loss = self.l1_loss(pred_delta, gt_delta)
#                 l1_losses.append(l1_loss.mean())
#             else:
#                 l1_losses.append(torch.tensor(0., device=device))
#                 iou_losses.append(torch.tensor(0., device=device))
#
#         # ✅ 返回平均损失
#         return {
#             "l1_loss": torch.stack(l1_losses).mean(),
#             "cls_loss": torch.stack(cls_losses).mean(),
#             "iou_loss": torch.stack(iou_losses).mean(),
#             "tot_loss": (
#                     torch.stack(l1_losses).mean() * self.l1_loss_scale +
#                     torch.stack(cls_losses).mean() * self.cls_loss_scale +
#                     torch.stack(iou_losses).mean() * self.iou_loss_scale
#             )
#         }