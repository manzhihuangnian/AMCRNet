import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from ..transformer.transformer import bbox_target

class CLS_Head(nn.Module):
    def __init__(
            self,
            in_channels=2048,
            num_classes=81,  #
            dropout_ratio=0,
            multilabel=True,
            stages=1,
    ):

        super().__init__()
        self.dropout_ratio=dropout_ratio
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.multilabel = multilabel

        # Class 0 is ignored when calculating accuracy,
        #      so topk cannot be equal to num_classes.
        in_channels = self.in_channels
        self.stages=stages
        if stages==1:
            if dropout_ratio > 0:
                self.dropout = nn.Dropout(dropout_ratio)
            self.fc_cls = nn.Linear(in_channels, num_classes)
        else:
            self.fc_cls=nn.Sequential(
                nn.Linear(in_channels, in_channels//2),
                nn.Dropout(dropout_ratio),
                nn.Linear(in_channels//2,num_classes)
            )

    def forward(self, x):
        if self.stages==1:
            if self.dropout_ratio > 0 :
                x = self.dropout(x)
            x = x.view(x.size(0), -1)
            cls_score = self.fc_cls(x)
        else:
            x = x.view(x.size(0), -1)
            cls_score = self.fc_cls(x)
        return cls_score

    def get_det_bboxes(self,
                       rois,
                       cls_score,
                       img_shape,
                       flip=False,
                       crop_quadruple=None,
                       cfg=None):

        # might be used by testing w. augmentation
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        # Handle Multi/Single Label
        if cls_score is not None:
            if self.multilabel:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = rois[:, 1:]
        assert bboxes.shape[-1] == 4

        # First reverse the flip
        img_h, img_w = img_shape
        if flip:
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        # Then normalize the bbox to [0, 1]
        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        def _bbox_crop_undo(bboxes, crop_quadruple):
            decropped = bboxes.clone()

            if crop_quadruple is not None:
                x1, y1, tw, th = crop_quadruple
                decropped[:, 0::2] = bboxes[..., 0::2] * tw + x1
                decropped[:, 1::2] = bboxes[..., 1::2] * th + y1

            return decropped

        bboxes = _bbox_crop_undo(bboxes, crop_quadruple)
        return bboxes, scores