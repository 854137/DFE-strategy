# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from . import LOGGER
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy
from .torch_utils import TORCH_1_11

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class IoUAssigner(nn.Module):
    """
    修复版：基于IoU的简单Assigner，用于目标检测任务。

    修复内容：
    1. 完整的IoU计算实现
    2. 正确的正负样本分配逻辑
    3. 健壮的边界情况处理
    4. 与Ultralytics框架完全兼容
    """

    def __init__(self, iou_threshold: float = 0.3, num_classes: int = 80, eps: float = 1e-9):
        """
        初始化IoUAssigner对象

        Args:
            iou_threshold (float): IoU阈值，降低到0.3确保有足够的正样本
            num_classes (int): 类别数量
            eps (float): 防止除零的小值
        """
        super().__init__()
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes
        self.bg_idx = num_classes  # 背景类索引
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        计算IoU分配

        Args:
            pd_scores (torch.Tensor): 预测分类分数，形状为 (bs, num_total_anchors, num_classes)
            pd_bboxes (torch.Tensor): 预测边界框，形状为 (bs, num_total_anchors, 4)
            anc_points (torch.Tensor): 锚点，形状为 (num_total_anchors, 2)
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)
            mask_gt (torch.Tensor): 有效真实框的掩码，形状为 (bs, n_max_boxes, 1)

        Returns:
            target_labels (torch.Tensor): 目标标签，形状为 (bs, num_total_anchors)
            target_bboxes (torch.Tensor): 目标边界框，形状为 (bs, num_total_anchors, 4)
            target_scores (torch.Tensor): 目标分数，形状为 (bs, num_total_anchors, num_classes)
            fg_mask (torch.Tensor): 前景掩码，形状为 (bs, num_total_anchors)
            target_gt_idx (torch.Tensor): 目标真实框索引，形状为 (bs, num_total_anchors)
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        self.n_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full([self.bs, self.n_anchors], self.bg_idx, device=device),
                torch.zeros([self.bs, self.n_anchors, 4], device=device),
                torch.zeros([self.bs, self.n_anchors, self.num_classes], device=device),
                torch.zeros([self.bs, self.n_anchors], device=device),
                torch.zeros([self.bs, self.n_anchors], device=device)
            )

        # 计算IoU矩阵 [bs, n_max_boxes, n_anchors]
        overlaps = self.iou_calculator(gt_bboxes, pd_bboxes, mask_gt)

        # 创建正样本掩码：IoU大于阈值的anchor为正样本 [bs, n_max_boxes, n_anchors]
        pos_mask = overlaps > self.iou_threshold

        # 应用GT框掩码，只考虑有效的GT框
        valid_mask = mask_gt.squeeze(-1).bool()  # [bs, n_max_boxes]
        for b in range(self.bs):
            pos_mask[b, ~valid_mask[b]] = False

        # 计算每个anchor的前景掩码 [bs, n_anchors]
        fg_mask = pos_mask.sum(1) > 0  # 至少有一个GT框与之匹配

        # 确保至少有一些正样本，如果没有，使用top-k策略
        for b in range(self.bs):
            if fg_mask[b].sum() == 0:
                # 获取该批次所有IoU值
                batch_overlaps = overlaps[b]  # [n_max_boxes, n_anchors]
                valid_overlaps = batch_overlaps[valid_mask[b]]

                if valid_overlaps.numel() > 0:
                    # 取top-k个最高IoU的anchor作为正样本 (k=3)
                    topk = min(3, valid_overlaps.numel())
                    topk_values, _ = torch.topk(valid_overlaps.flatten(), topk)
                    min_iou = topk_values[-1]  # 第k大的IoU值

                    # 创建新的正样本掩码
                    new_pos_mask = batch_overlaps >= min_iou
                    new_pos_mask[~valid_mask[b]] = False

                    pos_mask[b] = new_pos_mask
                    fg_mask[b] = new_pos_mask.sum(0) > 0

        # 为每个anchor分配GT索引 [bs, n_anchors]
        target_gt_idx = torch.argmax(overlaps, dim=1)

        # 获取分配的目标
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, valid_mask
        )

        # 调试信息（可选）
        total_pos = fg_mask.sum().item()
        if total_pos == 0:
            print(f"⚠️ 警告：批次中没有正样本！最大IoU: {overlaps.max().item():.4f}")
        else:
            print(
                f"✅ 正样本数量: {total_pos}/{self.bs * self.n_anchors} ({total_pos / (self.bs * self.n_anchors) * 100:.2f}%)")

        return target_labels.long(), target_bboxes, target_scores, fg_mask, target_gt_idx

    def iou_calculator(self, gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor, mask_gt: torch.Tensor):
        """
        计算真实框与预测框之间的IoU

        Args:
            gt_bboxes (torch.Tensor): 真实框，形状为 (bs, n_max_boxes, 4) [x1, y1, x2, y2]
            pd_bboxes (torch.Tensor): 预测框，形状为 (bs, n_anchors, 4) [x1, y1, x2, y2]
            mask_gt (torch.Tensor): 有效GT框掩码，形状为 (bs, n_max_boxes, 1)

        Returns:
            torch.Tensor: IoU矩阵，形状为 (bs, n_max_boxes, n_anchors)
        """
        try:
            from ultralytics.utils.metrics import bbox_iou
        except ImportError:
            # 备用IoU计算，如果无法导入
            def bbox_iou(box1, box2, xywh=False, CIoU=False):
                # 简化的IoU计算
                if xywh:
                    # 转换为xyxy
                    box1 = torch.cat((box1[..., :2] - box1[..., 2:] / 2, box1[..., :2] + box1[..., 2:] / 2), -1)
                    box2 = torch.cat((box2[..., :2] - box2[..., 2:] / 2, box2[..., :2] + box2[..., 2:] / 2), -1)

                # 计算交集
                inter_x1 = torch.max(box1[..., 0], box2[..., 0])
                inter_y1 = torch.max(box1[..., 1], box2[..., 1])
                inter_x2 = torch.min(box1[..., 2], box2[..., 2])
                inter_y2 = torch.min(box1[..., 3], box2[..., 3])

                inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

                # 计算并集
                area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
                area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
                union_area = area1 + area2 - inter_area

                iou = inter_area / (union_area + self.eps)
                return iou

        bs, n_max_boxes, _ = gt_bboxes.shape
        _, n_anchors, _ = pd_bboxes.shape

        # 创建IoU矩阵 [bs, n_max_boxes, n_anchors]
        overlaps = torch.zeros((bs, n_max_boxes, n_anchors), device=gt_bboxes.device)

        # 逐批次计算IoU
        for b in range(bs):
            # 获取当前批次的有效GT
            valid_mask = mask_gt[b].squeeze(-1).bool()
            valid_gt_boxes = gt_bboxes[b][valid_mask]
            valid_gt_count = valid_gt_boxes.shape[0]

            if valid_gt_count == 0:
                continue

            # 获取当前批次的预测框
            batch_pd_bboxes = pd_bboxes[b]  # [n_anchors, 4]

            # 为每个有效的GT框计算与所有预测框的IoU
            for i, gt_box in enumerate(valid_gt_boxes):
                # 扩展gt_box以匹配预测框数量 [n_anchors, 4]
                expanded_gt = gt_box.unsqueeze(0).expand(n_anchors, -1)
                # 计算IoU [n_anchors]
                ious = bbox_iou(expanded_gt, batch_pd_bboxes, xywh=False, CIoU=False)
                # 填充到overlaps矩阵
                valid_idx = torch.where(valid_mask)[0][i]
                overlaps[b, valid_idx] = ious

        return overlaps

    def get_targets(self, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor,
                    target_gt_idx: torch.Tensor, fg_mask: torch.Tensor, valid_mask: torch.Tensor):
        """
        获取分配的目标标签、边界框和分数

        Args:
            gt_labels (torch.Tensor): 真实标签，形状为 (bs, n_max_boxes, 1)
            gt_bboxes (torch.Tensor): 真实边界框，形状为 (bs, n_max_boxes, 4)
            target_gt_idx (torch.Tensor): 分配的GT框索引，形状为 (bs, n_anchors)
            fg_mask (torch.Tensor): 前景掩码，形状为 (bs, n_anchors)
            valid_mask (torch.Tensor): 有效GT框掩码，形状为 (bs, n_max_boxes)

        Returns:
            target_labels (torch.Tensor): 目标标签，形状为 (bs, n_anchors)
            target_bboxes (torch.Tensor): 目标边界框，形状为 (bs, n_anchors, 4)
            target_scores (torch.Tensor): 目标分数，形状为 (bs, n_anchors, num_classes)
        """
        # 创建目标张量
        target_labels = torch.full((self.bs, self.n_anchors), self.bg_idx, device=gt_labels.device)
        target_bboxes = torch.zeros((self.bs, self.n_anchors, 4), device=gt_bboxes.device)
        target_scores = torch.zeros((self.bs, self.n_anchors, self.num_classes), device=gt_labels.device)

        # 为每个批次分配目标
        for b in range(self.bs):
            # 获取当前批次的有效前景索引
            fg_indices = torch.where(fg_mask[b])[0]

            if len(fg_indices) > 0:
                # 获取这些前景anchor对应的GT索引
                assigned_gt_indices = target_gt_idx[b, fg_indices]

                # 检查GT索引是否有效
                valid_assignments = valid_mask[b][assigned_gt_indices]
                valid_fg_indices = fg_indices[valid_assignments]
                valid_gt_indices = assigned_gt_indices[valid_assignments]

                if len(valid_fg_indices) > 0:
                    # 分配标签
                    target_labels[b, valid_fg_indices] = gt_labels[b, valid_gt_indices, 0].squeeze()

                    # 分配边界框
                    target_bboxes[b, valid_fg_indices] = gt_bboxes[b, valid_gt_indices]

                    # 分配分数 (one-hot编码)
                    for i, (fg_idx, gt_idx) in enumerate(zip(valid_fg_indices, valid_gt_indices)):
                        class_idx = int(gt_labels[b, gt_idx, 0].item())
                        if 0 <= class_idx < self.num_classes:
                            target_scores[b, fg_idx, class_idx] = 1.0

        return target_labels, target_bboxes, target_scores



# 完整版ATSS，但是要加东西
# class ATSSAssigner(nn.Module):
#     """
#     Adaptive Training Sample Selection Assigner for object detection.
#
#     This implementation follows the official ATSS paper more closely and
#     includes production-ready features like proper FPN level handling and
#     centerness calculation.
#     """
#
#     def __init__(
#             self,
#             topk: int = 9,
#             num_classes: int = 80,
#             num_levels: int = 5,
#             eps: float = 1e-9
#     ):
#         """
#         Initialize ATSSAssigner.
#
#         Args:
#             topk: Number of top candidates per FPN level for each GT box
#             num_classes: Number of object classes
#             num_levels: Number of FPN levels (P3-P7)
#             eps: Small value for numerical stability
#         """
#         super().__init__()
#         self.topk = topk
#         self.num_classes = num_classes
#         self.bg_idx = num_classes
#         self.num_levels = num_levels
#         self.eps = eps
#
#     @torch.no_grad()
#     def forward(
#             self,
#             pd_scores: torch.Tensor,
#             pd_bboxes: torch.Tensor,
#             anc_points: torch.Tensor,
#             gt_labels: torch.Tensor,
#             gt_bboxes: torch.Tensor,
#             mask_gt: torch.Tensor,
#             num_level_anchors: Optional[List[int]] = None
#     ) -> Tuple[torch.Tensor, ...]:
#         """
#         Compute ATSS assignment.
#
#         Args:
#             pd_scores: Predicted scores (bs, n_anchors, num_classes)
#             pd_bboxes: Predicted boxes (bs, n_anchors, 4) in xyxy format
#             anc_points: Anchor points (n_anchors, 2)
#             gt_labels: GT labels (bs, n_max_boxes, 1)
#             gt_bboxes: GT boxes (bs, n_max_boxes, 4) in xyxy format
#             mask_gt: Valid GT mask (bs, n_max_boxes, 1)
#             num_level_anchors: List of anchor counts per FPN level
#
#         Returns:
#             target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
#         """
#         self.bs = pd_scores.shape[0]
#         self.n_max_boxes = gt_bboxes.shape[1]
#         self.n_anchors = anc_points.shape[0]
#         device = gt_bboxes.device
#
#         # Handle empty GT case
#         if self.n_max_boxes == 0 or mask_gt.sum() == 0:
#             return self._handle_empty_gt(device)
#
#         # Use provided level info or auto-generate (fallback)
#         if num_level_anchors is None:
#             num_level_anchors = self._estimate_level_anchors()
#
#         # Validate level anchors sum
#         assert sum(num_level_anchors) == self.n_anchors, \
#             f"Sum of level anchors {sum(num_level_anchors)} != total anchors {self.n_anchors}"
#
#         # 1. Calculate distances between GT centers and anchor points
#         distances, ac_points = self._calculate_distances(gt_bboxes, anc_points)
#
#         # 2. Select top-k candidates per level
#         is_in_candidate = self._select_topk_candidates(
#             distances, num_level_anchors, mask_gt
#         )
#
#         # 3. Calculate IoU between GT and anchors (as points)
#         overlaps = self._calculate_overlaps(gt_bboxes, anc_points)
#
#         # 4. Compute dynamic IoU thresholds
#         overlaps_thr_per_gt = self._calculate_dynamic_thresholds(
#             overlaps, is_in_candidate
#         )
#
#         # 5. Select positive samples: IoU >= threshold AND inside GT
#         is_pos = (overlaps >= overlaps_thr_per_gt) & is_in_candidate
#         is_in_gts = self._select_candidates_in_gts(ac_points, gt_bboxes)
#         mask_pos = is_pos & is_in_gts & mask_gt.bool()
#
#         # 6. Resolve multi-GT assignment conflicts
#         target_gt_idx, fg_mask, mask_pos = self._resolve_assigned_matches(
#             mask_pos, overlaps
#         )
#
#         # 7. Get final targets
#         target_labels, target_bboxes, target_scores = self._get_targets(
#             gt_labels, gt_bboxes, target_gt_idx, fg_mask
#         )
#
#         # 8. Apply soft labels using predicted IoU (optional enhancement)
#         if pd_bboxes is not None:
#             pred_overlaps = self._calculate_pred_overlaps(gt_bboxes, pd_bboxes, mask_pos)
#             target_scores *= pred_overlaps
#
#         return (
#             target_labels.long(),
#             target_bboxes,
#             target_scores,
#             fg_mask.bool(),
#             target_gt_idx
#         )
#
#     def _handle_empty_gt(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
#         """Handle case with no ground truth objects."""
#         return (
#             torch.full([self.bs, self.n_anchors], self.bg_idx, device=device),
#             torch.zeros([self.bs, self.n_anchors, 4], device=device),
#             torch.zeros([self.bs, self.n_anchors, self.num_classes], device=device),
#             torch.zeros([self.bs, self.n_anchors], device=device, dtype=torch.bool),
#             torch.zeros([self.bs, self.n_anchors], device=device, dtype=torch.long)
#         )
#
#     def _estimate_level_anchors(self) -> List[int]:
#         """
#         Estimate anchor distribution across FPN levels.
#         In practice, this should be provided by the detector.
#         """
#         # Simple heuristic: geometric distribution (more anchors at high resolution)
#         total = self.n_anchors
#         base = int(total / (self.num_levels * 0.8))  # 80% base distribution
#
#         levels = []
#         remaining = total
#         for i in range(self.num_levels):
#             size = max(1, int(base * (0.75 ** i)))  # Decrease by 25% per level
#             size = min(size, remaining - (self.num_levels - i - 1))
#             levels.append(size)
#             remaining -= size
#
#         # Adjust last level to consume remaining anchors
#         levels[-1] += remaining
#         return levels
#
#     def _calculate_distances(
#             self,
#             gt_bboxes: torch.Tensor,
#             anc_points: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Calculate L2 distances between GT centers and anchor points."""
#         gt_centers = (gt_bboxes[..., :2] + gt_bboxes[..., 2:]) / 2.0
#
#         # Vectorized distance calculation
#         # gt_centers: (bs, n_max_boxes, 1, 2)
#         # anc_points: (1, 1, n_anchors, 2)
#         distances = torch.cdist(gt_centers, anc_points.unsqueeze(0), p=2)
#
#         return distances, anc_points
#
#     def _select_topk_candidates(
#             self,
#             distances: torch.Tensor,
#             num_level_anchors: List[int],
#             mask_gt: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Select top-k anchors per GT box for each FPN level.
#         Returns a boolean mask (bs, n_max_boxes, n_anchors).
#         """
#         mask_gt = mask_gt.bool()  # (bs, n_max_boxes, 1)
#         is_in_candidate = torch.zeros_like(distances, dtype=torch.bool)
#
#         start_idx = 0
#         for level_idx, n_level_anchors in enumerate(num_level_anchors):
#             end_idx = start_idx + n_level_anchors
#             level_distances = distances[..., start_idx:end_idx]
#
#             k = min(self.topk, n_level_anchors)
#             # Get top-k indices (bs, n_max_boxes, k)
#             _, topk_idxs = torch.topk(level_distances, k, dim=-1, largest=False)
#
#             # Convert to global indices
#             global_idxs = topk_idxs + start_idx
#
#             # Scatter to create level mask (bs, n_max_boxes, n_anchors_in_level)
#             level_mask = torch.zeros_like(level_distances, dtype=torch.bool)
#             level_mask.scatter_(-1, topk_idxs, True)
#
#             is_in_candidate[..., start_idx:end_idx] = level_mask
#             start_idx = end_idx
#
#         # Filter invalid GT boxes
#         return is_in_candidate & mask_gt
#
#     def _calculate_overlaps(
#             self,
#             gt_bboxes: torch.Tensor,
#             anc_points: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Calculate IoU between GT boxes and anchor points (as 1x1 boxes).
#         NOTE: In practice, consider using actual anchor boxes for better accuracy.
#         """
#         # Convert anchor points to tiny boxes for IoU calculation
#         anc_boxes = torch.cat([anc_points - 0.5, anc_points + 0.5], dim=-1)
#
#         # Vectorized IoU calculation
#         return self._bbox_iou(gt_bboxes, anc_boxes.unsqueeze(0))
#
#     def _calculate_pred_overlaps(
#             self,
#             gt_bboxes: torch.Tensor,
#             pred_bboxes: torch.Tensor,
#             mask_pos: torch.Tensor
#     ) -> torch.Tensor:
#         """Calculate IoU between GT and predicted boxes for soft labeling."""
#         overlaps = self._bbox_iou(gt_bboxes, pred_bboxes.unsqueeze(1))
#         overlaps = overlaps * mask_pos
#         # Max IoU per anchor (bs, n_anchors)
#         return overlaps.amax(dim=-2).unsqueeze(-1)
#
#     def _bbox_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
#         """Efficient IoU calculation using broadcasting."""
#         # boxes1: (bs, n_max_boxes, 4) or broadcastable
#         # boxes2: (bs, n_anchors, 4) or broadcastable
#
#         # Expand for broadcasting
#         boxes1 = boxes1.unsqueeze(2)  # (bs, n_max_boxes, 1, 4)
#         boxes2 = boxes2.unsqueeze(1)  # (bs, 1, n_anchors, 4)
#
#         # Intersection
#         lt = torch.maximum(boxes1[..., :2], boxes2[..., :2])  # x1, y1
#         rb = torch.minimum(boxes1[..., 2:], boxes2[..., 2:])  # x2, y2
#
#         wh = torch.clamp(rb - lt, min=0)
#         inter = wh[..., 0] * wh[..., 1]
#
#         # Union
#         area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
#         area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
#         union = area1 + area2 - inter
#
#         return inter / (union + self.eps)
#
#     def _calculate_dynamic_thresholds(
#             self,
#             overlaps: torch.Tensor,
#             is_in_candidate: torch.Tensor
#     ) -> torch.Tensor:
#         """Compute mean + std threshold for each GT based on candidate IoUs."""
#         # Mask overlaps to only consider candidates
#         candidate_overlaps = torch.where(is_in_candidate, overlaps, torch.tensor(0., device=overlaps.device))
#
#         # Use masked mean and std
#         # Sum and divide by count to ignore masked values
#         valid_mask = is_in_candidate.float()
#         overlaps_sum = (candidate_overlaps * valid_mask).sum(dim=-1, keepdim=True)
#         overlaps_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
#
#         overlaps_mean = overlaps_sum / overlaps_count
#         # For std, we need to center by mean
#         centered = (candidate_overlaps - overlaps_mean) * valid_mask
#         overlaps_std = torch.sqrt((centered ** 2).sum(dim=-1, keepdim=True) / overlaps_count)
#
#         return overlaps_mean + overlaps_std
#
#     def _select_candidates_in_gts(
#             self,
#             anc_points: torch.Tensor,
#             gt_bboxes: torch.Tensor
#     ) -> torch.Tensor:
#         """Check if anchor centers are inside GT boxes."""
#         # anc_points: (n_anchors, 2) -> (1, 1, n_anchors, 2)
#         # gt_bboxes: (bs, n_max_boxes, 4) -> (bs, n_max_boxes, 1, 4)
#         anc_expanded = anc_points.unsqueeze(0).unsqueeze(0)
#         gt_expanded = gt_bboxes.unsqueeze(2)
#
#         # Check left-top and right-bottom conditions
#         lt = anc_expanded[..., :2] >= gt_expanded[..., :2]  # x >= x1, y >= y1
#         rb = anc_expanded[..., :2] <= gt_expanded[..., 2:]  # x <= x2, y <= y2
#
#         return (lt & rb).all(dim=-1)  # (bs, n_max_boxes, n_anchors)
#
#     def _resolve_assigned_matches(
#             self,
#             mask_pos: torch.Tensor,
#             overlaps: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Resolve anchors assigned to multiple GTs by selecting highest IoU."""
#         # Count assignments per anchor
#         fg_mask = mask_pos.sum(dim=-2)  # (bs, n_anchors)
#
#         # Identify anchors with multiple assignments
#         multi_assign_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, self.n_max_boxes, -1)
#
#         if multi_assign_mask.any():
#             # Find GT with max IoU for each anchor
#             max_overlaps_idx = overlaps.argmax(dim=1)  # (bs, n_anchors)
#
#             # Create mask for highest IoU assignments only
#             is_max_overlap = torch.zeros_like(mask_pos, dtype=torch.bool)
#             is_max_overlap.scatter_(1, max_overlaps_idx.unsqueeze(1), True)
#
#             # Update mask: keep only max IoU for multi-assigned anchors
#             mask_pos = torch.where(multi_assign_mask, is_max_overlap, mask_pos)
#
#         # Get final GT index for each anchor
#         target_gt_idx = mask_pos.to(torch.long).argmax(dim=-2)  # (bs, n_anchors)
#         fg_mask = mask_pos.sum(dim=-2)  # Update foreground mask
#
#         return target_gt_idx, fg_mask, mask_pos
#
#     def _get_targets(
#             self,
#             gt_labels: torch.Tensor,
#             gt_bboxes: torch.Tensor,
#             target_gt_idx: torch.Tensor,
#             fg_mask: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Generate target labels, boxes and scores."""
#         batch_indices = torch.arange(self.bs, device=gt_labels.device).view(-1, 1)
#
#         # Flatten indices for gather
#         flat_idx = target_gt_idx + batch_indices * self.n_max_boxes
#
#         # Gather targets
#         target_labels = gt_labels.reshape(-1)[flat_idx]  # (bs, n_anchors)
#         target_bboxes = gt_bboxes.reshape(-1, 4)[flat_idx]  # (bs, n_anchors, 4)
#
#         # Set background labels
#         target_labels = torch.where(
#             fg_mask.bool(),
#             target_labels,
#             torch.full_like(target_labels, self.bg_idx)
#         )
#
#         # One-hot encode class labels
#         target_scores = F.one_hot(
#             target_labels.long(),
#             num_classes=self.num_classes + 1
#         )[..., :self.num_classes].float()  # Remove background dimension
#
#         return target_labels, target_bboxes, target_scores

class ATSSAssigner(nn.Module):
    """
    Adaptive Training Sample Selection Assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the ATSS strategy,
    which dynamically selects positive samples for each gt box using statistical methods.

    Attributes:
        topk (int): The number of top candidates to consider for each gt box.
        num_classes (int): The number of object classes.
        bg_idx (int): Background class index.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 9, num_classes: int = 80, eps: float = 1e-9):
        """
        Initialize an ATSSAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider for each gt box.
            num_classes (int, optional): The number of object classes.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes  # background class index
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the ATSS assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        self.n_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full([self.bs, self.n_anchors], self.bg_idx, device=device),
                torch.zeros([self.bs, self.n_anchors, 4], device=device),
                torch.zeros([self.bs, self.n_anchors, self.num_classes], device=device),
                torch.zeros([self.bs, self.n_anchors], device=device),
                torch.zeros([self.bs, self.n_anchors], device=device)
            )

        # Calculate IoU between all gt boxes and anchor points
        overlaps = self.iou_calculator(gt_bboxes, anc_points)
        # overlaps shape: (bs, n_max_boxes, n_anchors)

        # Calculate distances between gt centers and anchor points
        distances, ac_points = self.dist_calculator(gt_bboxes, anc_points)
        # distances shape: (bs, n_max_boxes, n_anchors)

        # Get level information for anchors (assuming a simple distribution)
        n_level_bboxes = self._get_anchor_levels(anc_points)

        # Select top-k candidates for each gt box
        is_in_candidate, candidate_idxs = self.select_topk_candidates(distances, n_level_bboxes, mask_gt)

        # Calculate dynamic IoU threshold for each gt box
        overlaps_thr_per_gt, candidate_overlaps = self.thres_calculator(is_in_candidate, candidate_idxs, overlaps)

        # Select positive samples: candidates with IoU >= threshold
        is_pos = torch.where(
            candidate_overlaps > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate,
            torch.zeros_like(is_in_candidate)
        )

        # Filter candidates that are inside gt boxes
        is_in_gts = self.select_candidates_in_gts(ac_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        # If an anchor is assigned to multiple gts, select the one with highest IoU
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Get assigned targets
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Apply soft labels using IoU
        if pd_bboxes is not None:
            # Calculate IoU between gt boxes and predicted boxes
            ious = self.iou_calculator_for_pred_boxes(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(dim=-2)[0].unsqueeze(-1)  # Get max IoU for each anchor
            target_scores *= ious

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _get_anchor_levels(self, anc_points: torch.Tensor) -> List[int]:
        """
        Estimate anchor levels based on anchor distribution.
        This is a simplified version assuming anchors are distributed across 3 levels.

        Args:
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).

        Returns:
            List[int]: Number of anchors at each level.
        """
        # Simple heuristic: divide anchors evenly among 3 levels
        n_anchors = anc_points.shape[0]
        n_level1 = n_anchors // 3
        n_level2 = n_anchors // 3
        n_level3 = n_anchors - n_level1 - n_level2
        return [n_level1, n_level2, n_level3]

    def iou_calculator(self, gt_bboxes: torch.Tensor, anc_points: torch.Tensor):
        """
        Calculate IoU between ground truth boxes and anchor points.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4) in (x1, y1, x2, y2) format.
            anc_points (torch.Tensor): Anchor points with shape (n_anchors, 2).

        Returns:
            torch.Tensor: IoU matrix with shape (bs, n_max_boxes, n_anchors).
        """
        from .metrics import bbox_iou  # 从当前包导入bbox_iou函数

        bs, n_max_boxes, _ = gt_bboxes.shape
        n_anchors, _ = anc_points.shape

        # 将anchor点转换为小的边界框 (x1, y1, x2, y2)
        # 为每个anchor点创建一个1x1的小框
        anchor_boxes = torch.cat([
            anc_points - 0.5,  # x1, y1
            anc_points + 0.5  # x2, y2
        ], dim=1)  # shape: (n_anchors, 4)

        # 扩展gt_bboxes和anchor_boxes以便计算所有组合的IoU
        # gt_bboxes: (bs, n_max_boxes, 4) -> (bs, n_max_boxes, 1, 4)
        gt_expanded = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)  # (bs, n_max_boxes, n_anchors, 4)
        # anchor_boxes: (n_anchors, 4) -> (1, 1, n_anchors, 4)
        anc_expanded = anchor_boxes.unsqueeze(0).unsqueeze(0).expand(bs, n_max_boxes, -1,
                                                                     -1)  # (bs, n_max_boxes, n_anchors, 4)

        # 重塑张量以便计算IoU
        gt_flat = gt_expanded.reshape(-1, 4)  # (bs * n_max_boxes * n_anchors, 4)
        anc_flat = anc_expanded.reshape(-1, 4)  # (bs * n_max_boxes * n_anchors, 4)

        # 计算IoU
        ious = bbox_iou(gt_flat, anc_flat, xywh=False, CIoU=False)

        # 重塑回原始形状
        return ious.reshape(bs, n_max_boxes, n_anchors)

    def iou_calculator_for_pred_boxes(self, gt_bboxes: torch.Tensor, pred_bboxes: torch.Tensor):
        """
        Calculate IoU between ground truth boxes and predicted boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4) in (x1, y1, x2, y2) format.
            pred_bboxes (torch.Tensor): Predicted boxes with shape (bs, n_anchors, 4) in (x1, y1, x2, y2) format.

        Returns:
            torch.Tensor: IoU matrix with shape (bs, n_max_boxes, n_anchors).
        """
        from .metrics import bbox_iou  # 从当前包导入bbox_iou函数

        bs, n_max_boxes, _ = gt_bboxes.shape
        _, n_anchors, _ = pred_bboxes.shape

        # 扩展gt_bboxes和pred_bboxes以便计算所有组合的IoU
        # gt_bboxes: (bs, n_max_boxes, 4) -> (bs, n_max_boxes, 1, 4)
        gt_expanded = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)  # (bs, n_max_boxes, n_anchors, 4)
        # pred_bboxes: (bs, n_anchors, 4) -> (bs, 1, n_anchors, 4)
        pred_expanded = pred_bboxes.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)  # (bs, n_max_boxes, n_anchors, 4)

        # 重塑张量以便计算IoU
        gt_flat = gt_expanded.reshape(-1, 4)  # (bs * n_max_boxes * n_anchors, 4)
        pred_flat = pred_expanded.reshape(-1, 4)  # (bs * n_max_boxes * n_anchors, 4)

        # 计算IoU
        ious = bbox_iou(gt_flat, pred_flat, xywh=False, CIoU=False)

        # 重塑回原始形状
        return ious.reshape(bs, n_max_boxes, n_anchors)

    def dist_calculator(self, gt_bboxes: torch.Tensor, anc_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate distances between ground truth box centers and anchor points.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4) in (x1, y1, x2, y2) format.
            anc_points (torch.Tensor): Anchor points with shape (n_anchors, 2).

        Returns:
            distances (torch.Tensor): Distance matrix with shape (bs, n_max_boxes, n_anchors).
            ac_points (torch.Tensor): Anchor points (unchanged).
        """
        bs, n_max_boxes, _ = gt_bboxes.shape
        n_anchors, _ = anc_points.shape

        # Calculate center points of gt boxes
        gt_centers = (gt_bboxes[:, :, :2] + gt_bboxes[:, :, 2:]) / 2.0  # (bs, n_max_boxes, 2)

        # Expand gt_centers to (bs, n_max_boxes, 1, 2) and anc_points to (1, 1, n_anchors, 2)
        gt_centers_expanded = gt_centers.unsqueeze(2).expand(-1, -1, n_anchors, -1)
        anc_points_expanded = anc_points.unsqueeze(0).unsqueeze(0).expand(bs, n_max_boxes, -1, -1)

        # Calculate distances
        distances = torch.sqrt(
            (gt_centers_expanded[..., 0] - anc_points_expanded[..., 0]) ** 2 +
            (gt_centers_expanded[..., 1] - anc_points_expanded[..., 1]) ** 2
        )

        return distances, anc_points

    def select_topk_candidates(self, distances: torch.Tensor, n_level_bboxes: List[int], mask_gt: torch.Tensor):
        """
        Select top-k candidates for each ground truth box based on distances.

        Args:
            distances (torch.Tensor): Distance matrix with shape (bs, n_max_boxes, n_anchors).
            n_level_bboxes (List[int]): Number of anchors at each feature level.
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            is_in_candidate (torch.Tensor): Boolean mask for candidates with shape (bs, n_max_boxes, n_anchors).
            candidate_idxs (torch.Tensor): Indices of selected candidates with shape (bs, n_max_boxes, topk).
        """
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()

        # Split distances by feature levels
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)

        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0

        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):
            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)

            # Get top-k closest anchors for each gt box
            _, per_level_topk_idxs = torch.topk(
                per_level_distances, selected_k, dim=-1, largest=False
            )

            # Adjust indices to global anchor indices
            candidate_idxs.append(per_level_topk_idxs + start_idx)

            # Apply mask to filter out invalid gt boxes
            per_level_topk_idxs = torch.where(
                mask_gt, per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs)
            )

            # Create one-hot encoding for selected candidates
            is_in_candidate = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(
                is_in_candidate > 1, torch.zeros_like(is_in_candidate), is_in_candidate
            )

            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self, is_in_candidate: torch.Tensor, candidate_idxs: torch.Tensor, overlaps: torch.Tensor):
        """
        Calculate dynamic IoU threshold for each ground truth box.

        Args:
            is_in_candidate (torch.Tensor): Boolean mask for candidates with shape (bs, n_max_boxes, n_anchors).
            candidate_idxs (torch.Tensor): Indices of selected candidates with shape (bs, n_max_boxes, topk).
            overlaps (torch.Tensor): IoU matrix with shape (bs, n_max_boxes, n_anchors).

        Returns:
            overlaps_thr_per_gt (torch.Tensor): IoU threshold for each gt box with shape (bs, n_max_boxes, 1).
            candidate_overlaps (torch.Tensor): IoU values for candidates with shape (bs, n_max_boxes, n_anchors).
        """
        n_bs_max_boxes = self.bs * self.n_max_boxes

        # Filter overlaps to only include candidates
        candidate_overlaps = torch.where(
            is_in_candidate > 0, overlaps, torch.zeros_like(overlaps)
        )

        # Reshape for easier indexing
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])

        # Create flattened indices for gathering candidate IoUs
        assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=candidate_idxs.device)
        assist_idxs = assist_idxs[:, None]
        flatten_idxs = candidate_idxs + assist_idxs

        # Gather IoUs for candidates
        candidate_iou_values = candidate_overlaps.reshape(-1)[flatten_idxs]
        candidate_iou_values = candidate_iou_values.reshape([self.bs, self.n_max_boxes, -1])

        # Calculate mean and standard deviation of IoUs for each gt box
        overlaps_mean_per_gt = candidate_iou_values.mean(dim=-1, keepdim=True)
        overlaps_std_per_gt = candidate_iou_values.std(dim=-1, keepdim=True)

        # Dynamic threshold = mean + std
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, candidate_overlaps

    def select_candidates_in_gts(self, xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9):
        """
        Select anchor centers that are inside ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (n_anchors, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (bs, n_max_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            torch.Tensor: Boolean mask with shape (bs, n_max_boxes, n_anchors).
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape

        # Calculate left-top and right-bottom points of gt boxes
        lt = gt_bboxes[:, :, :2].unsqueeze(2).expand(-1, -1, n_anchors, -1)
        rb = gt_bboxes[:, :, 2:].unsqueeze(2).expand(-1, -1, n_anchors, -1)
        xy_centers_expanded = xy_centers.unsqueeze(0).unsqueeze(0).expand(bs, n_boxes, -1, -1)

        # Calculate deltas
        bbox_deltas = torch.cat((xy_centers_expanded - lt, rb - xy_centers_expanded), dim=3)

        return bbox_deltas.amin(3).gt_(eps)

    def select_highest_overlaps(self, mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int):
        """
        If an anchor is assigned to multiple gts, select the one with highest IoU.

        Args:
            mask_pos (torch.Tensor): Positive mask with shape (bs, n_max_boxes, n_anchors).
            overlaps (torch.Tensor): IoU matrix with shape (bs, n_max_boxes, n_anchors).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned gt boxes with shape (bs, n_anchors).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, n_anchors).
            mask_pos (torch.Tensor): Updated positive mask with shape (bs, n_max_boxes, n_anchors).
        """
        # Sum over gt dimension to get number of gts assigned to each anchor
        fg_mask = mask_pos.sum(-2)

        # If an anchor is assigned to multiple gts
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            max_overlaps_idx = overlaps.argmax(1)  # Get index of max IoU for each anchor

            # Create mask for highest IoU assignments
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            # Update mask to keep only highest IoU assignments
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)

        # Get index of assigned gt for each anchor
        target_gt_idx = mask_pos.argmax(-2)

        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels: torch.Tensor, gt_bboxes: torch.Tensor, target_gt_idx: torch.Tensor,
                    fg_mask: torch.Tensor):
        """
        Compute target labels, bounding boxes, and scores for positive anchors.

        Args:
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            target_gt_idx (torch.Tensor): Indices of assigned gt boxes with shape (bs, n_anchors).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, n_anchors).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, n_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, n_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, n_anchors, num_classes).
        """
        # Create batch indices
        batch_idx = torch.arange(self.bs, device=gt_labels.device).unsqueeze(1)

        # Adjust target_gt_idx for flattened indexing
        target_gt_idx = target_gt_idx + batch_idx * self.n_max_boxes

        # Get target labels
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])

        # Set background labels for negative anchors
        target_labels = torch.where(
            fg_mask > 0,
            target_labels,
            torch.full_like(target_labels, self.bg_idx)
        )

        # Get target bounding boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # Get target scores (one-hot encoding)
        target_scores = torch.zeros(
            (self.bs, self.n_anchors, self.num_classes),
            dtype=torch.float32,
            device=gt_labels.device
        )

        # Only set scores for foreground anchors
        fg_indices = torch.nonzero(fg_mask, as_tuple=True)
        if fg_indices[0].numel() > 0:
            target_labels_fg = target_labels[fg_mask.bool()].long()
            target_scores[fg_indices[0], fg_indices[1], target_labels_fg] = 1.0

        return target_labels, target_bboxes, target_scores

class TaskAlignedAssigner(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).# 元素：每个检测框的四个方位
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).# 元素：每个预测框每个类的分数
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where
                topk is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos



class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    """Assigns ground-truth objects to rotated bounding boxes using a task-aligned metric."""

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Calculate IoU for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates with shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (b, n_boxes, 5).

        Returns:
            (torch.Tensor): Boolean mask of positive anchors with shape (b, n_boxes, h*w).
        """
        # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, n_boxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, n_boxes, h*w, 2)
        ap = xy_centers - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted rotated bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance with shape (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle with shape (bs, h*w, 1).
        anchor_points (torch.Tensor): Anchor points with shape (h*w, 2).
        dim (int, optional): Dimension along which to split.

    Returns:
        (torch.Tensor): Predicted rotated bounding boxes with shape (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
