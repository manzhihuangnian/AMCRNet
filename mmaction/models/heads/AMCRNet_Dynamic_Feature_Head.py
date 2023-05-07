import torch
import math
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import ConvModule
from einops import rearrange
from ..builder import HEADS,LFB,build_lfb,build_roiextractor
from ..transformer.transformer import bbox2roi
from ..transformer.transformer import Attention,DropPath,Mlp
from .AMCRNet_cls_head import CLS_Head
import numpy as np
from mmaction.core.bbox import bbox2result
from mmaction.apis.test_twostage import collect_results_cpu, collect_results_gpu


class BHOI_block(nn.Module):
    def __init__(self,dim,
                 num_heads,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_norm=norm_layer(dim)
        
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp_norm=norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_layer=DropPath(drop_path) if drop_path>0 else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        attn_value = self.attn(self.attn_norm(x), attn_mask.to(x.device))
        x=self.drop_layer(attn_value)+x
        out=self.drop_layer(self.mlp(self.mlp_norm(x)))+x
        return out


class BHOI_Module(nn.Module):
    def __init__(self,
                 type="MHSA",
                 deepth=3,
                 num_head=3,
                 dims=2048,
                 mlp_ratio=4,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.type=type
        self.deepth=deepth
        self.dims=dims
        self.mlp_ratio=mlp_ratio
        self.drop=drop
        self.attn_drop=attn_drop
        self.drop_path=drop_path
        self.num_head=num_head        
        self.block=BHOI_block
        print("depth",self.deepth)
        self.build_net()

    def build_net(self):
        self.blocks=nn.ModuleList()
        for i in range(self.deepth):
            self.blocks.append(self.block(dim=self.dims,
                                num_heads=self.num_head,
                                mlp_ratio=self.mlp_ratio,
                                 drop_path=self.drop_path,
                                 ))

    def forward(self,x, attn_mask):
        for block in self.blocks:
            x = block(x, attn_mask)
        return x


class BBOX_Head(nn.Module):
    def __init__(self):
        pass
    def get_target(self):
        pass


class CONTEXT_Module(nn.Module):
    def __init__(self,
                 channels=2048,
                 type="conv_pool",
                 convs=None,
                 poolings=None,
                 norm_cfg=None,
                 pool_ratio=None,
                 pool_type="avgpool",
                 layer_embed=False,
                 ):
        super().__init__()
        self.type=type
        self.convs=convs
        self.poolings=poolings
        self.channels=channels
        self.norm_cfg=norm_cfg
        self.pool_type=pool_type
        self.pool_ratio=pool_ratio
        self.layer_embed=layer_embed
        if self.pool_ratio:
            self.num_ration=len(self.pool_ratio)
            if self.layer_embed:
                self.layer_embedindg=nn.Parameter(torch.zeros(self.num_ration, self.channels), requires_grad=True)

        assert self.type in ["convs", "pool_out", "conv_pool", 'pool_kernel']
        if self.convs is not None:
            self.build_conv_context(convs=self.convs)
        if self.poolings is not None:
            self.build_pool_context(pools=self.poolings)
        if self.pool_ratio is None:
            print("context_module",self)
        else:
            print("context_module", self.pool_ratio)

    def build_conv_context(self,convs):
        self.conv_context_names=[]
        self.conv_pos_names=[]
        for i, conv_cfg in enumerate(zip(convs["size"],convs["stride"])):
            conv_context=ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=conv_cfg[0],
                stride=conv_cfg[1],
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="GN", num_groups=16) if self.norm_cfg is None else self.norm_cfg,
                act_cfg=dict(type="ReLU")
                            )
            self.add_module(f"conv_context{i}",conv_context)
            self.conv_context_names.append(f"conv_context{i}")
            
            conv_pos = ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=conv_cfg[0],
                stride=conv_cfg[1],
                conv_cfg=dict(type="Conv2d"),
                norm_cfg=dict(type="GN", num_groups=16) if self.norm_cfg is None else self.norm_cfg,
                act_cfg=dict(type="ReLU")
            )
            self.add_module(f"conv_pos{i}", conv_pos)
            self.conv_pos_names.append(f"conv_pos{i}")

    def build_pool_context(self,pools):
        self.pool_context_names=[]
        if self.type=="pool_out":
            for i, pool_out_size in enumerate(pools["size"]):
                if self.pool_type=="avgpool":
                    self.add_module(f"pool_context{i}", nn.AdaptiveAvgPool2d(pool_out_size))
                else:
                    self.add_module(f"pool_context{i}", nn.AdaptiveMaxPool2d(pool_out_size))
                self.pool_context_names.append(f"pool_context{i}")
        else:
            for i , pool_cfg in enumerate(zip(self.poolings["size"], self.poolings["stride"], self.poolings["padding"])):
                if self.pool_type=="avgpool":
                    pool_context=nn.AvgPool2d(
                        kernel_size=pool_cfg[0],
                        stride=pool_cfg[1],
                        padding=pool_cfg[2]
                    )
                else:
                    pool_context=nn.MaxPool2d(
                        kernel_size=pool_cfg[0],
                        stride=pool_cfg[1],
                        padding=pool_cfg[2]
                    )
                self.add_module(f"pool_context{i}",pool_context)
                self.pool_context_names.append(f"pool_context{i}")


    def forward(self,feat,pos=None):
        out_feat=[]
        out_pos=[]
        B, _, H, W = feat.shape
        if self.convs is not None:
            for conv_text_name, conv_pos_name in zip(self.conv_context_names,self.conv_pos_names):
                conv_context=getattr(self,conv_text_name)
                conv_pos=getattr(self, conv_pos_name)
                context_feat=conv_context(feat)
                context_pos=conv_pos(pos)
                out_feat.append(context_feat)
                out_pos.append(context_pos)

        if self.poolings is not None:
            for pool_context_name in self.pool_context_names:
                pool_context=getattr(self,pool_context_name)
                context_feat=pool_context(feat)
                context_pos=pool_context(pos)
                out_feat.append(context_feat)
                out_pos.append(context_pos)

        if self.pool_ratio is not None:
            for index, ratio in enumerate(self.pool_ratio):
                pool_context = F.adaptive_avg_pool2d(feat, (round(H / ratio[0]), round(W / ratio[1])))
                pool_pos = F.adaptive_avg_pool2d(pos, (round(H / ratio[0]), round(W / ratio[1])))
                if self.layer_embed:
                    single_layer_embed = self.layer_embedindg[index][None][None][None].permute(0, 3, 1,
                                                                                               2).contiguous()
                    pool_pos += single_layer_embed
                out_feat.append(pool_context)
                out_pos.append(pool_pos)

        for i, feat_single_scale in enumerate(out_feat):
            ####[B,C,H,W]
            B,C,H,W=feat_single_scale.shape
            feat_single_scale=feat_single_scale.contiguous().permute(0,2,3,1).view(B,H*W,C)
            out_feat[i]=feat_single_scale

        #####[B,num_context,C]
        out_feat=torch.cat(out_feat,dim=1)

        for i, pos_single_scale in enumerate(out_pos):
            ####[1,C,H,W]
            B, C, H, W = pos_single_scale.shape
            pos_single_scale=pos_single_scale.contiguous().permute(0, 2, 3, 1).view(B, H * W, C)
            out_pos[i]=pos_single_scale

        #####[1,num_context,C]
        out_pos=torch.cat(out_pos, dim=1)

        return out_feat, out_pos



class LOSS_EMA(object):
    def __init__(self,
                 decay=0.9,
                 updates=0):
        self.updates = updates
        self.loss=torch.tensor(0.0)
        self.decay = lambda x: decay * (1 - math.exp(-x / 200))  # decay exponential ramp (to help early epochs)

    def update(self, loss):
        self.updates += 1
        decay_now = self.decay(self.updates)
        self.loss*=decay_now
        self.loss += (1. - decay_now) * loss
        self.applay()

    def applay(self):
        self.loss=self.loss/(1-self.decay(self.updates)**self.updates)

@HEADS.register_module()
class AMCRNet_Dynamic_Head(nn.Module):
    def __init__(self,
                 roi_head=None,
                 mask_mode="zero",
                 grid=False,
                 bbox_head=None,
                 context_module=None,
                 BHOI_module_cfg=None,
                 Dynamic_LFB=None,
                 window_size=11,
                 BHOI_module_high_cfg=None,
                 BHOI_module_roi_cfg=None,
                 cls_head=None,
                 test_cfg=None,
                 LFB_path="./results/Dynamic_LFB"
                 ):
        super().__init__()
        self.mask_mode=mask_mode
        self.roi_head=build_roiextractor(roi_head)
        if bbox_head is not None and bbox_head["activate"]:
            self.bbox_head=BBOX_Head(**bbox_head)
        self.context_module=CONTEXT_Module(**context_module)
        self.BHOI_modlue=BHOI_Module(**BHOI_module_cfg)
        self.cls_head=CLS_Head(**cls_head)
        self.test_cfg=test_cfg
        self.grid=grid

        self.LFB_path = os.path.join(LFB_path, "lfb", "LFB.pkl")
        self.LFB_Extractor_Dynamic = build_lfb(Dynamic_LFB)
        if BHOI_module_high_cfg:
            self.BHOI_module_high = BHOI_Module(**BHOI_module_high_cfg)
        if BHOI_module_roi_cfg:
            self.BHOI_module_roi = BHOI_Module(**BHOI_module_roi_cfg)
        self.window_size = window_size
        self.temporal_embed = nn.Parameter(torch.zeros(self.window_size, BHOI_module_high_cfg["dims"]), requires_grad=False)
        print(self)

    @property
    def with_BHOI_high(self):
        return hasattr(self,"BHOI_module_high")

    @property
    def with_BHOI_roi(self):
        return hasattr(self,"BHOI_module_roi")
    
    @property
    def with_dynamic(self):
        return hasattr(self,"LFB_Extractor_Dynamic")
    
    @property
    def with_bbox(self):
        return hasattr(self, "bbox_head")

    @property
    def with_cls(self):
        return hasattr(self, "cls_head")


    def extract_long_term_features_onebranch(self,img_metas, forward_loss, LFB ):
        lt_high_feats, index_list=LFB.get_memory_feature_onebranch(img_metas, forward_loss_list=forward_loss)
        return lt_high_feats, index_list

    def extract_long_term_features_twobranch(self,img_metas, forward_loss,LFB):
        lt_high_feats, lt_roi_feats, index_list=LFB.get_memory_feature_twobranch(img_metas, forward_loss_list=forward_loss)
        return lt_high_feats, lt_roi_feats, index_list

    def simple_test_onestage(self,
                    x,
                    proposal_list,
                    img_metas,
                    rescale=False,
                    pos=None):
        """Defines the computation performed for simple testing."""
        assert self.with_cls, 'Cls head must be implemented.'
        if isinstance(x, tuple):
            x_shape = x[0].shape
        else:
            x_shape = x.shape

        assert x_shape[0] == 1, 'only accept 1 sample at test mode'
        assert x_shape[0] == len(img_metas) == len(proposal_list)

        features_dict, rois = self.simple_test_bboxes_onestage(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, pos=pos)

        return features_dict, img_metas, rois

    def simple_test_bboxes_onestage(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           pos=None):
        """Test only det bboxes without augmentation."""
        ###########[[[img_id,box],[img_id,box],..............]]
        rois = bbox2roi(proposals)
        video_id, timestamp=img_metas[0]["img_key"].split(",")
        timestamp=int(timestamp)
        if self.with_BHOI_roi:
            features_high, update_features=self._cls_forward_onestage(x, rois, img_metas, pos=pos)
            timestamp_dict = dict(loss_tag=torch.tensor(1), high_relation=list(features_high[0].cpu()),
                                 update_feature=list(update_features.cpu()))
        else:
            features_high =self._cls_forward_onestage(x, rois, img_metas, pos=pos)
            timestamp_dict = dict(loss_tag=torch.tensor(1), high_relation=list(features_high[0].cpu()))
        features_dict = {
            video_id: {
                timestamp: timestamp_dict
            }
        }

        return features_dict, rois


    def simple_test_twostage(self,
                             LFB=None,
                             img_metas=None,
                             rois=None):
        """Defines the computation performed for simple testing."""
        assert self.with_cls, 'Cls head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes_twostage(
                                                            LFB=LFB,
                                                            img_metas=img_metas,
                                                            rois=rois)

        bbox_results = bbox2result(
            det_bboxes,
            det_labels,
            self.cls_head.num_classes,
            thr=self.test_cfg.action_thr)
        return [bbox_results]

    def simple_test_bboxes_twostage(self,
                           LFB=None,
                           img_metas=None,
                           rois=None
                           ):

        cls_results = self._cls_forward_twostage(LFB=LFB, img_metas=img_metas)
        cls_score = cls_results['cls_score']

        img_shape = img_metas[0]['img_shape']
        crop_quadruple = np.array([0, 0, 1, 1])
        flip = False

        if 'crop_quadruple' in img_metas[0]:
            crop_quadruple = img_metas[0]['crop_quadruple']

        if 'flip' in img_metas[0]:
            flip = img_metas[0]['flip']

        det_bboxes, det_labels = self.cls_head.get_det_bboxes(
            rois,
            cls_score,
            img_shape,
            flip=flip,
            crop_quadruple=crop_quadruple,
            )

        return det_bboxes, det_labels


    def _cls_forward_onestage(self, x, rois, img_metas, pos=None):
        #########[b,1024,h,w],[n-rois,1024,1,1]
        trans_feats, roi_feat_list, roi_pos_list = self.roi_head(x, rois, pos)
        roi_update_features=roi_feat_list

        context_feat_local_list, pos_local = self.context_module(trans_feats, pos)

        B, C, H, W = trans_feats.shape

        img_index_list, rois_num_list = torch.unique(rois[:, 0], sorted=True, return_counts=True)

        context_feat_grid_list = [rearrange(single_frame_feat, 'c h w -> (h w) c', c=C, h=H, w=W) for single_frame_feat
                                  in trans_feats]

        BHOI_input_list = []
        for single_frame_roi_feat, single_frame_pos_roi, single_frame_context_feat_local, single_frame_context_feat_grid in \
                zip(roi_feat_list, roi_pos_list, context_feat_local_list, context_feat_grid_list):

            input_feat = []
            input_pos = []
            input_feat.append(single_frame_roi_feat)
            input_feat.append(single_frame_context_feat_local)

            ########[rois, local, grid]
            input_pos.append(single_frame_pos_roi)
            input_pos.append(pos_local.squeeze())
            #####pos[1,C,H,W]---->[H*W,C]

            if self.grid:
                input_feat.append(single_frame_context_feat_grid)
                input_pos.append(pos.squeeze().permute(1, 2, 0).contiguous().view(-1, C))

            input_feat = torch.cat(input_feat, dim=0)
            input_pos = torch.cat(input_pos, dim=0)
            BHOI_input_list.append(input_feat + input_pos)

        max_len = max([feat.shape[0] for feat in BHOI_input_list])

        attn_mask = []
        ###########feat[num_feat,C] 
        for index, feat in enumerate(BHOI_input_list):
            cur_len = feat.shape[0]
            if self.mask_mode:
                single_mask = torch.zeros((1, max_len))

            single_mask[:, cur_len:] = -math.inf
            padding = max_len - cur_len

            feat = F.pad(feat.permute(1, 0)[None], pad=(0, padding), mode="replicate")
            feat = feat[0].permute(1, 0).contiguous()
            BHOI_input_list[index] = feat[None]
            if padding:
                single_mask_padding=torch.zeros((1,max_len))
                single_mask_padding[:,:cur_len]=-math.inf
                ###########[1,max_len,max_len]
                attn_mask.append(torch.cat([single_mask[None].repeat(1, cur_len, 1),
                                            single_mask_padding[None].repeat(1,padding,1)],dim=1))
            else:
                attn_mask.append(single_mask[None].repeat(1,max_len,1))

        # [B,Max_len,C]
        BHOI_input_list = torch.cat(BHOI_input_list, dim=0)
        # [B,Max_len,Max_len]
        attn_mask = torch.cat(attn_mask, dim=0)

        BHOI_out = self.BHOI_modlue(BHOI_input_list, attn_mask)
        roi_high_relation = []
        for i in range(B):
            single_frame_BHOI_out = BHOI_out[i]
            single_frame_roi_num = rois_num_list[i]
            roi_high_relation.append(single_frame_BHOI_out[:single_frame_roi_num])
        return roi_high_relation

    def _cls_forward_twostage(self, LFB, img_metas):
        video_id, timestap = img_metas[0]["img_key"].split(",")
        timestap = int(timestap)
        cur_high_relations = LFB[video_id][timestap]["high_relation"]
        cur_high_relations = [cur_high_relations]

        if not self.with_BHOI_roi:
            lt_high_feats, index_list = self.extract_long_term_features_onebranch(img_metas=img_metas,
                                                                                  forward_loss=[torch.tensor(1)],
                                                                                  LFB=LFB)

        else:
            lt_high_feats, lt_roi_feat, index_list = self.extract_long_term_features_twobranch(img_metas=img_metas,
                                                                                  forward_loss=[torch.tensor(1)],
                                                                                  LFB=LFB)

            roi_update_features = LFB[video_id][timestap]["update_feature"]
            roi_update_features = [roi_update_features]

        rois_num_list=[]
        BHOI_input_list_high=[]
        for cur_high_relation, single_lt_high_features, single_index_list\
                in zip(cur_high_relations, lt_high_feats, index_list):
            single_clip_input_high=[]
            ###########[1, 1024]
            cur_clip_temporal_embed=self.temporal_embed[self.window_size//2][None]
            ###########[n_rois_now,1024]
            cur_high_relation=[feat[None].cuda() for feat in cur_high_relation]
            cur_high_relation=torch.cat(cur_high_relation,dim=0)+cur_clip_temporal_embed
            ###########[n_rois_now,1024]
            single_clip_input_high.append(cur_high_relation)
            rois_num_list.append(cur_high_relation.shape[0])

            if len(single_index_list):
                his_single_temporal_embed = self.temporal_embed[single_index_list]
                for his_clip_lt_high_features, his_clip_temporal_embed in zip(single_lt_high_features,list(his_single_temporal_embed)):

                    his_clip_lt_high_features=[feat[None] for feat in his_clip_lt_high_features]
                    #######[n_rois_clipn,C]
                    his_clip_lt_high_features=torch.cat(his_clip_lt_high_features,dim=0)+his_clip_temporal_embed[None]

                    single_clip_input_high.append(his_clip_lt_high_features)
            single_clip_input_high=torch.cat(single_clip_input_high,dim=0)
            BHOI_input_list_high.append(single_clip_input_high)

        if self.with_BHOI_roi:
            BHOI_input_list_roi = []
            for single_cur_roi_update_feature, single_lt_roi_features, single_index_list \
                    in zip(roi_update_features, lt_roi_feat, index_list):
                single_clip_input_roi = []
                ###########[1, 1024]
                cur_clip_temporal_embed = self.temporal_embed[self.window_size // 2][None]
                single_cur_roi_update_feature=[feat[None].cuda() for feat in single_cur_roi_update_feature]
                single_cur_roi_update_feature = torch.cat(single_cur_roi_update_feature,dim=0) + cur_clip_temporal_embed
                ###########[n_rois_now,1024]
                single_clip_input_roi.append(single_cur_roi_update_feature)
                if len(single_index_list):
                    his_single_temporal_embed = self.temporal_embed[single_index_list]
                    for his_clip_lt_roi_features, his_clip_temporal_embed in zip(single_lt_roi_features,
                                                                                 list(his_single_temporal_embed)):
                        his_clip_lt_roi_features = [feat[None] for feat in his_clip_lt_roi_features]
                        #######[n_rois_clipn,C]
                        his_clip_lt_roi_features = torch.cat(his_clip_lt_roi_features, dim=0) + his_clip_temporal_embed[
                            None]
                        #######[n_rois_clip1+n_rois_clip2+...,C]
                        single_clip_input_roi.append(his_clip_lt_roi_features)
                single_clip_input_roi = torch.cat(single_clip_input_roi, dim=0)
                BHOI_input_list_roi.append(single_clip_input_roi)

        # [B,Max_len,C]
        BHOI_input_list_high = torch.cat(BHOI_input_list_high, dim=0)
        BHOI_input_list_high = BHOI_input_list_high[None]
        attn_mask_high = torch.zeros((1, BHOI_input_list_high.shape[0], BHOI_input_list_high.shape[0]))

        BHOI_input_list_high_out = self.BHOI_module_high(BHOI_input_list_high, attn_mask_high)

        B=BHOI_input_list_high.shape[0]

        ###########
        high_relation_high_roi=[]

        if self.with_BHOI_roi:
            BHOI_input_list_roi = torch.cat(BHOI_input_list_roi, dim=0)
            BHOI_input_list_roi= BHOI_input_list_roi[None]
            BHOI_input_list_roi_out = self.BHOI_module_roi(BHOI_input_list_roi, attn_mask_high)

            for i in range(B):
                single_frame_BHOI_out_high, single_frame_BHOI_out_roi = BHOI_input_list_high_out[i], BHOI_input_list_roi_out[i]
                single_frame_roi_num = rois_num_list[i]
                high_relation_high_roi.append(
                    torch.cat([single_frame_BHOI_out_high[:single_frame_roi_num],single_frame_BHOI_out_roi],dim=1))
        else:
            for i in range(B):
                single_frame_BHOI_out_high = BHOI_input_list_high_out[i]
                single_frame_roi_num = rois_num_list[i]
                high_relation_high_roi.append(single_frame_BHOI_out_high[:single_frame_roi_num])

        high_relation_high_roi=torch.cat(high_relation_high_roi,dim=0)

        cls_score = self.cls_head(high_relation_high_roi)

        cls_results = dict(
            cls_score=cls_score)
        return cls_results
