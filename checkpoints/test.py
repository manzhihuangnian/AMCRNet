import torch
# from mmcv.cnn import ConvModule
# conv3d=ConvModule(conv_cfg=dict(type="Conv3d"),kernel_size=(3,3,3),in_channels=3,out_channels=3)
# conv=torch.nn.Conv3d(3,1,kernel_size=(3,3,3))
# data=torch.ones((4,3,7,7,7))
# conv.cuda()
# data.cuda()
# conv3d.cuda()


#
# import torch.nn as nn
# input=torch.ones((1,256,16,35))
#
# conv3=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=3,stride=2)
# conv5=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=5,stride=3)
# conv7=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=7,stride=4)
# conv3_1=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(3,2),stride=(2,1))
# conv1_3=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(2,3),stride=(1,2))
# conv5_3=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(5,3),stride=(3,2))
# conv3_5=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(3,5),stride=(2,3))
# conv7_5=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(7,5),stride=(4,3))
# conv5_7=nn.Conv2d(in_channels=256,out_channels=25,kernel_size=(5,7),stride=(3,4))
#
# pool3=nn.MaxPool2d(kernel_size=3,stride=2)
# pool5=nn.MaxPool2d(kernel_size=5,stride=3)
# pool7=nn.MaxPool2d(kernel_size=7,stride=4)
#
# print(conv3(input).shape)
# print(conv5(input).shape)
# print(conv7(input).shape)
#
# print(conv3_1(input).shape)
# print(conv1_3(input).shape)
# print(conv5_3(input).shape)
# print(conv3_5(input).shape)
# print(conv7_5(input).shape)
# print(conv5_7(input).shape)
#
# #####选择池化或者卷积
# print(pool3(input).shape)
# print(pool5(input).shape)
# print(pool7(input).shape)





import sys
import os

if __name__=="__main__":
    model_path=sys.argv[1]
    new_path=os.path.join(os.path.dirname(model_path),f"{os.path.splitext()[0]}_new.pth")
    state_dict=torch.load(model_path,map_location="cpu")["state_dict"]
    state_dict={"state_dict": state_dict}
    torch.save(state_dict,new_path)

