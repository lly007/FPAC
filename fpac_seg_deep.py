import torch.nn as nn
import torch
import torch.nn.functional as F
from pointnet_util import PointNetSetAbstraction,PointNetFeaturePropagation,PointNetFeaturePropagation2,PointNetFeaturePropagation_backup
from fpac import FPAC


class get_model(nn.Module):
    def __init__(self, num_classes, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        framepoint = torch.tensor([[1.0,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],[0,0,0]]).cuda()
        self.fpac1 = FPAC(npoint=1024, cin=3+additional_channel, cout=128, radius=0.1, nsample=16, m1=[3,9,1], m2=[1,32,128], framepoints=framepoint)
        self.fpac2 = FPAC(npoint=512, cin=128, cout=256, radius=0.2, nsample=32, m1=[3,9,1], m2=[1,32,128], framepoints=framepoint)
        self.fpac3 = FPAC(npoint=256, cin=256, cout=512, radius=0.3, nsample=32, m1=[3,9,1], m2=[1,32,128], framepoints=framepoint)
        self.fpac4 = FPAC(npoint=128, cin=512, cout=512, radius=0.4, nsample=64, m1=[3,64,1], m2=[1,128,512], framepoints=framepoint)
        self.fpac5 = FPAC(npoint=32, cin=512, cout=512, radius=0.6, nsample=64, m1=[3,9,1], m2=[1,128,512], framepoints=framepoint)
        self.fpac6 = FPAC(npoint=None, cin=512, cout=1024, radius=0.8, nsample=32, m1=[3,64,1], m2=[1,128,1024], framepoints=framepoint)
        self.fp6 = PointNetFeaturePropagation_backup(in_channel=1536, mlp=[512, 256])
        self.fp5 = PointNetFeaturePropagation_backup(in_channel=768, mlp=[256, 128])
        self.fp4 = PointNetFeaturePropagation_backup(in_channel=640, mlp=[256, 128])
        self.fp3 = PointNetFeaturePropagation_backup(in_channel=384, mlp=[256, 128])
        self.fp2 = PointNetFeaturePropagation_backup(in_channel=256, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation_backup(in_channel=128+16+6+additional_channel, mlp=[128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz, cls_label):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l0_points = l0_points.transpose(1,2)
        l0_xyz = l0_xyz.transpose(1,2)

        f1, s1 = self.fpac1(l0_xyz, l0_points)
        f2, s2 = self.fpac2(s1, f1)
        f3, s3 = self.fpac3(s2, f2)
        f4, s4 = self.fpac4(s3, f3)
        f5, s5 = self.fpac5(s4, f4)
        f6, s6 = self.fpac6(s5, f5)
        # Feature Propagation layers
        l0_points = l0_points.transpose(1,2)
        l0_xyz = l0_xyz.transpose(1,2)
        f1 = f1.transpose(1,2)
        f2 = f2.transpose(1,2)
        f3 = f3.transpose(1,2)
        f4 = f4.transpose(1,2)
        f5 = f5.transpose(1,2)
        f6 = f6.transpose(1,2)
        s1 = s1.transpose(1,2)
        s2 = s2.transpose(1,2)
        s3 = s3.transpose(1,2)
        s4 = s4.transpose(1,2)
        s5 = s5.transpose(1,2)
        s6 = s6.transpose(1,2)
        l5_points = self.fp6(s5, s6, f5, f6)
        l4_points = self.fp5(s4, s5, f4, l5_points)
        l3_points = self.fp4(s3, s4, f3, l4_points)
        l2_points = self.fp3(s2, s3, f2, l3_points)
        l1_points = self.fp2(s1, s2, f1, l2_points)
        cls_label_one_hot = cls_label.view(B,16,1).repeat(1,1,N)
        l0_points = self.fp1(l0_xyz, s1, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        # FC layers
        feat =  F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        # diff = diff1 + diff2 + diff3
        return x, f3


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss