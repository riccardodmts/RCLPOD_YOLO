import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

"""YOLOv8 loss + L2 for LwF"""
class LwFLoss(v8DetectionLoss):
    def __init__(self, h, m, device, lwf=3.0, new_classes=[]):  # model must be de-paralleled
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.lwf = lwf
        self.new_classes = new_classes

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # LwF on the output
        self.lwf_loss = torch.nn.MSELoss()
        self.last_yolo_loss = 0
        self.last_lwf_loss = 0

    def preprocess(self, targets, batch_size, scale_tensor):
        """
        Preprocesses the target counts and matches with the input batch size
        to output a tensor.
        """
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """
        Decode predicted object bounding box coordinates from anchor points and
        distribution.
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch, teacher_output):
        """
        Calculate the sum of the loss for box, cls and dfl multiplied by batch size.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        teacher_output = teacher_output[1] if isinstance(teacher_output, tuple) else teacher_output
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)

        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.clone().detach().sigmoid(),
            (pred_bboxes.clone().detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = (
               self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum 
            )


        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor


            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        lwf_loss = 0
        filter_idx = self.reg_max * 4 + self.new_classes[0]

        for i in range(3):
            lwf_loss += self.lwf_loss(feats[i][:, : filter_idx,:,:], teacher_output[i][:, :filter_idx,:,:].detach())
        lwf_loss /= 3

        #print(type(lwf_loss))
        total_loss = loss.sum() * batch_size + self.lwf * lwf_loss * batch_size
        self.last_yolo_loss = loss.sum().item()
        self.last_lwf_loss = lwf_loss.item()

        #return torch.tensor(0.0, requires_grad=True), loss.detach()  # loss(box, cls, dfl)
        return total_loss, loss.detach()
    