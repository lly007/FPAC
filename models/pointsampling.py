import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class get_model(nn.Module):
    def __init__(self, npoint = 512):
        super(get_model, self).__init__()

        self.c1 = 3
        self.c2 = 64
        self.c3 = 256
        self.c4 = 512

        self.cs1 = self.c4
        self.cs2 = 512
        self.cs3 = 256
        self.cs4 = 128

        self.conv1 = nn.Conv1d(self.c1, self.c2, 1)
        self.conv2 = nn.Conv1d(self.c2, self.c3, 1)
        self.conv3 = nn.Conv1d(self.c3, self.c4, 1)

        self.bn1 = nn.BatchNorm1d(self.c2)
        self.bn2 = nn.BatchNorm1d(self.c3)
        self.bn3 = nn.BatchNorm1d(self.c4)

        self.convs1 = nn.Conv1d(self.cs1, self.cs2, 1)  # 考虑到与前三层编码的特征进行连接
        self.convs2 = nn.Conv1d(self.cs2, self.cs3, 1)
        self.convs3 = nn.Conv1d(self.cs3, self.cs4, 1)
        self.convs4 = nn.Conv1d(self.cs4, npoint, 1)

        self.bns1 = nn.BatchNorm1d(self.cs2)
        self.bns2 = nn.BatchNorm1d(self.cs3)
        self.bns3 = nn.BatchNorm1d(self.cs4)
        
        self.softmax = nn.Softmax(-1)

        self.npoint = npoint

    def forward(self, point_xyz):
        B, N, D = point_xyz.size()
        point_xyz = point_xyz.transpose(2, 1)  # 修改为pytorch的channel first格式

        out1 = F.relu(self.bn1(self.conv1(point_xyz)))
        #print('out1_size'+str(out1.size()))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        #print('out2_size'+str(out2.size()))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        #print('out3_size'+str(out3.size()))

        #out_max = torch.max(out3, 2, True)[0]
        #expand = out_max.repeat(1, 1, N)

        #concat = torch.cat([expand, out3, out2, out1, point_xyz], 1)
        #concat = torch.cat([out3, out2, out1, point_xyz], 1)
        #print('concat_size'+str(concat.size()))

        out4 = F.relu(self.bns1(self.convs1(out3)))
        out5 = F.relu(self.bns2(self.convs2(out4)))
        out6 = F.relu(self.bns3(self.convs3(out5)))

        out7 = self.convs4(out6)   #  [B,S,N]

        out8 = out7.transpose(2, 1).view(B, N, self.npoint)  # [B, N, S]

        result = self.softmax(out8)  # [B, N, S] 归一化
        # net = net.data.topk(spoint)[1]
        #print(net[0].shape)
        #print(net)
        return result  # [B, S]


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.lossf = nn.CrossEntropyLoss()

    # 损失函数的pred输入为[B,N,2],target的输入为[B,N]
    def forward(self, pred, target):
        # sampled_points = u.index_points(xyz, pred)
        B, N = pred.shape
        p = pred.view(-1, 5)
        t = target.view(-1)
        loss = self.lossf(p, t)
        return loss

