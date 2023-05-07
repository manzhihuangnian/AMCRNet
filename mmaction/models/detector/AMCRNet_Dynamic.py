from ..builder import ACDETECTORS,build_backbone,build_head,build_posenet
from torch import nn
import torch
from collections import OrderedDict
import torch.distributed as dist
import numpy as np

@ACDETECTORS.register_module()
class AMCRNet_Dynamic(nn.Module):
    def __init__(self,
                 backbone,
                 pose_net=None,
                 head=None,
                 test_cfg=None,
                 ):
        super().__init__()

        self.backbone=build_backbone(backbone)
        if pose_net is not None:
            self.pose_net=build_posenet(pose_net)
        if test_cfg is not None:
            head.update(dict(test_cfg=test_cfg))
        self.head=build_head(head)
        self.test_cfg=test_cfg


    def extract_feat(self, img):
        x = self.backbone(img)
        pos_head_spatial = self.pose_net(x[0].shape)[0]
        return x, pos_head_spatial

    def forward_test(self, imgs, img_metas,**kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test_onestage(imgs[0], img_metas[0], **kwargs)


    def simple_test_onestage(self, img, img_metas, proposals=None, rescale=False):
        x, pos_head_spatial = self.extract_feat(img)
        proposal_list = proposals
        return self.head.simple_test_onestage(
            x, proposal_list, img_metas, rescale=rescale, pos=pos_head_spatial)

    def simple_test_twostage(self, img_metas ,LFB, roi):
        return self.head.simple_test_twostage(
            img_metas=img_metas, LFB=LFB, rois=roi)

    def forward(self,
                img=None,
                img_metas=None,
                return_loss=True,
                stage=1,
                img_meta=None,
                LFB=None,
                roi=None,
                **kwargs):
                
        if stage==1:
            return self.forward_test(img, img_metas, **kwargs)
        else:
            return self.simple_test_twostage(img_metas=img_metas, LFB=LFB, roi=roi)


