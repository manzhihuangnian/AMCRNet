import torch.nn as nn
import torch
from ..builder import POSENETS,build_posenet
from mmcv.cnn import ConvModule
import numpy as np

@POSENETS.register_module()
class convpos(nn.Module):
    def __init__(self,
                 deepth=6,
                 out_channels=2048,
                 dims=[2,2, 4, 4, 4, 16, 128],
                 conv_size=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                 dilation=[[1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                 out_conv=[[3, 1, 1024], [3, 1, 1024]],
                 conv_cfg=None,
                 norm_cfg=dict(type="GN",num_groups=2),
                 act_cfg=dict(type="ReLU")
                 ):
        super().__init__()
        self.deepth=deepth
        self.dims=dims
        self.out_channels=out_channels
        self.conv_size=conv_size
        self.dilation=dilation
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.out_conv_cfg=out_conv
        self.build_net()


    def make_layer(self,
                   conv_sizes,
                   dilations,
                   dims
                   ):
        conv_list=[]
        for i, (conv_size,dilation) in enumerate(zip(conv_sizes,dilations)):
            conv_list.append(ConvModule(in_channels=dims[0] if i==0 else dims[1],
                            out_channels=dims[1] ,
                            kernel_size=conv_size,
                            dilation=dilation,
                            padding=((conv_size-1)*(dilation-1)+conv_size-1)//2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            stride=1
                            ))

        return conv_list


    def build_net(self):
        self.pos_net=nn.ModuleList()
        self.out_convs=[]
        for i in range(self.deepth):
            conv_size=self.conv_size[i]
            dilation=self.dilation[i]
            dims=self.dims[i:i+2]
            layer=self.make_layer(conv_sizes=conv_size,
                            dilations=dilation,
                            dims=dims,
                            )
            self.pos_net.append(nn.Sequential(*layer))
        for i ,out_conv_cfg in enumerate(self.out_conv_cfg):
            self.add_module(f"outconv{i}",
                            ConvModule(in_channels=self.dims[-1],
                                       out_channels=out_conv_cfg[-1],
                                       kernel_size=(out_conv_cfg[0],out_conv_cfg[0]),
                                       stride=(out_conv_cfg[1],out_conv_cfg[1]),
                                       padding=(out_conv_cfg[0]//2,out_conv_cfg[0]//2),
                                       conv_cfg=self.conv_cfg,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg
                                       ))
            self.out_convs.append(f"outconv{i}")

    def forward(self,size):
        outs=[]
        device= next(self.pos_net.parameters()).device
        inputs=torch.from_numpy(np.random.uniform(1,10,(1,1,size[-2],size[-1]))).float().to(device)
        for block in self.pos_net:
            inputs=block(inputs)
        for conv_name in self.out_convs:
            conv=getattr(self,conv_name)
            outs.append(conv(inputs))
        return outs




