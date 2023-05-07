import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from einops import rearrange
from ..builder import ROI_EXTRACTORS
from ..transformer.transformer import bbox2roi

@ROI_EXTRACTORS.register_module()
class AMCRNet_roiextractor(nn.Module):
    def __init__(self,
                 roi_layer_type='RoIAlign',
                 featmap_stride=16,
                 output_size=16,
                 pool_mode='avg',
                 aligned=True,
                 temporal_pool_size=2,
                 sampling_ratio=0,
                 channel_ratio=8,
                 channels=2048,
                 temporal_down_type="avgpool",
                 spatial_down_type="avgpool",
                 test_cfg=None,
                 norm_cfg=None,
                 use_channel_ratio=False,
                 ):
        super().__init__()
        self.roi_layer_type = roi_layer_type
        assert self.roi_layer_type in ['RoIPool', 'RoIAlign']
        self.featmap_stride = featmap_stride
        self.spatial_scale = 1. / self.featmap_stride


        self.output_size = output_size
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.temporal_pool_size = temporal_pool_size
        self.temporal_down_type = temporal_down_type
        self.spatial_down_type = spatial_down_type
        self.test_cfg = test_cfg
        self.channels=channels
        self.channel_ratio=channel_ratio
        self.sampling_ratio=sampling_ratio
        self.norm_cfg=norm_cfg
        self.use_channel_ratio=use_channel_ratio

        ######[b,2048,4,h,w]->[b,2048,2,h,w]
        if self.temporal_down_type == "conv":
            self.temporal_downsample = ConvModule(in_channels=self.channels,
                                                  out_channels=self.channels,
                                                  kernel_size=(3, 1, 1),
                                                  conv_cfg=dict(type="Conv3d"),
                                                  norm_cfg=dict(type="GN",num_groups=16) if self.norm_cfg is None else self.norm_cfg,
                                                  act_cfg=dict(type="ReLU",
                                                               inplace=True),
                                                  stride=(4 // self.temporal_pool_size, 1, 1),
                                                  padding=(1, 0, 0)
                                                  )
        else:
            ######[2048,4,w,h]---->[2048,2,w,h]
            if self.temporal_down_type == "avgpool":
                self.temporal_downsample = nn.AdaptiveAvgPool3d((self.temporal_pool_size,None,None))
            else:
                self.temporal_downsample = nn.AdaptiveAvgPool3d((self.temporal_pool_size,None,None))

        ########[b,2048,2,w,h]->[b,2048/4,2,w,h]->[b,1024,w,h]
        self.channel_downsample = ConvModule(in_channels=self.channels,
                                             out_channels=self.channels // (self.channel_ratio if \
                                                                                self.use_channel_ratio else self.temporal_pool_size),
                                             kernel_size=(1, 1, 1),
                                             conv_cfg=dict(type="Conv3d"),
                                             norm_cfg=dict(type="GN",num_groups=16) if self.norm_cfg is None else self.norm_cfg,
                                             act_cfg=dict(type="ReLU",
                                                          inplace=True),
                                             stride=(1, 1, 1)
                                             )

        if self.spatial_down_type == "avgpool":
            self.spatial_downsample = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.spatial_downsample = nn.AdaptiveMaxPool2d((1, 1))

        try:
            from mmcv.ops import RoIAlign, RoIPool
        except (ImportError, ModuleNotFoundError):
            raise ImportError('Failed to import `RoIAlign` and `RoIPool` from '
                              '`mmcv.ops`. The two modules will be used in '
                              '`SingleRoIExtractor3D`! ')

        if self.roi_layer_type == 'RoIPool':
            self.roi_layer = RoIPool(self.output_size, self.spatial_scale)
        else:
            self.roi_layer = RoIAlign(
                self.output_size,
                self.spatial_scale,
                sampling_ratio=self.sampling_ratio,
                pool_mode=self.pool_mode,
                aligned=self.aligned)

    def trans_feat(self, feat):
        slow_feat, fast_feat = feat
        B, C, T, W, H = fast_feat.shape
        ratio = int(32 / slow_feat.shape[2])
        fast_feat = fast_feat.permute(0, 3, 4, 2, 1).reshape(B, W, H, T // ratio, 256 * ratio).contiguous()
        ######[B,1024/2048,8/4,W,H]
        fast_feat = fast_feat.permute(0, 4, 3, 1, 2).contiguous()
        ######[B,4096/3072,4/8,W,H]
        feat = torch.cat([slow_feat, fast_feat], axis=1).contiguous()

        feat = self.temporal_downsample(feat)
        #######[b,4096/2048,2,h,w]->[b,512,2,h,w]  channe_ratio=8 or 4
        feat = self.channel_downsample(feat)
        B, C, T, W, H = feat.shape
        #######[b,512,2,h,w]->[b,1024,h,w]
        feat = rearrange(feat, 'b c t w h -> b (c t) w h', b=B, c=C, t=T, w=W, h=H)
        return feat

    def extract_roi_feat(self, feat, rois):
        #######[n_rois,1024,8,8]
        rois_feat = self.roi_layer(feat, rois)
        rois_feat = self.spatial_downsample(rois_feat)
        return rois_feat

    def forward(self, x, rois, pos=None):
        batch_size = int(x[0].shape[0])
        feat = self.trans_feat(x)
        #######[num_rois,C,1,1]
        rois_feat = self.extract_roi_feat(feat, rois)

        #######[num_rois,C,1,1]
        rois_pos =self.extract_roi_feat(pos.repeat(batch_size,1,1,1), rois)
        NUM,C,H,W=rois_feat.shape
        roi_pos_list=[]
        roi_feat_list=[]
        for i in range(batch_size):
            roi_feat_list.append([])
            roi_pos_list.append([])

        for roi_feat , roi_pos, roi in zip(rois_feat, rois_pos, rois):
            roi_feat=roi_feat.reshape(-1, C)
            roi_feat_list[int(roi[0].item())].append(roi_feat)
            roi_pos_list[int(roi[0].item())].append(roi_pos.contiguous().view(1,C))

        for i, single_frame_roi_feat  in enumerate(roi_feat_list):
            roi_feat_list[i]=torch.cat(single_frame_roi_feat,dim=0)

        for i, single_frame_roi_pos in enumerate(roi_pos_list):
            roi_pos_list[i] = torch.cat(single_frame_roi_pos, dim=0)

        #####[B,C,H,W],[[num_roi_feame1,C],...]
        return feat, roi_feat_list,roi_pos_list


