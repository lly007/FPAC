import torch
from torch._C import device
from torch.nn import Conv1d, ModuleList, Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
import pointnet_util as pu
import numpy as np


class FPAC(Module):
    def __init__(self, npoint, cin, cout, radius, nsample, m1, m2, mr, mid, framepoints):
        super(FPAC, self).__init__()
        self.device = framepoints.device
        self.npoint = npoint
        self.cin = cin + 3
        self.cout = cout
        self.radius = radius
        self.nsample = nsample
        self.m1 = m1
        self.m2 = m2
        self.mr = mr
        self.mid = mid
        self.framepoints = framepoints * radius
        self.m1_filter = ModuleList()
        self.m1_bn = ModuleList()
        self.m2_filter = ModuleList()
        self.m2_bn = ModuleList()
        self.mr_filter = ModuleList()
        self.mr_bn = ModuleList()
        self.m1_num = len(self.m1) - 1
        self.m2_num = len(self.m2) - 1
        self.mr_num = len(self.mr) - 1

        self.m1[0] = 6
        # self.m1[-1] = 1
        self.m1[-1] = self.mid  # 可以改成对全部权重值单独计算注意力a的值
        self.m2[0] = self.cin * self.cout
        self.m2[-1] = self.mid
        self.mr[0] = self.cin * self.mid
        self.mr[-1] = self.cout
        self.w = Parameter(torch.randn(self.framepoints.shape[0], self.cin, self.cout))  # 初始化Weight of Frame Points
        self.rotated_framepoints = self.rotate_frame_points(self.framepoints)  # [V, 3]

        for i in range(self.m1_num):
            self.m1_filter.append(Conv1d(self.m1[i], self.m1[i+1], 1))
            self.m1_bn.append(BatchNorm1d(self.m1[i+1]))
        for i in range(self.m2_num):
            self.m2_filter.append(Conv1d(self.m2[i], self.m2[i+1], 1))
            self.m2_bn.append(BatchNorm1d(self.m2[i+1]))
        for i in range(self.mr_num):
            self.mr_filter.append(Conv1d(self.mr[i], self.mr[i+1], 1))
            self.mr_bn.append(BatchNorm1d(self.mr[i+1]))

    def forward(self, xyz, f):
        device = xyz.device
        B, N, C = f.shape
        V = self.framepoints.shape[0]
        S = self.npoint
        n = self.nsample
        w = self.w

        if S is not None:
            new_xyz_idx = pu.farthest_point_sample(xyz.contiguous(), S)  # [B, S, 1]
            new_xyz = pu.index_points(xyz, new_xyz_idx)  # [B, S, 3]
            grouped_idx, mask = pu.query_ball_point_with_mask(self.radius, n, xyz, new_xyz)  # [B, S, nsample] 邻域点分组

        else:
            new_xyz = xyz.mean(dim = 1, keepdim = True) #[B, 1, 3]  S=1
            grouped_idx, mask = pu.query_ball_point_with_mask(10, n, xyz, new_xyz)  # [B, 1, nsample] 邻域点分组  S=1
            S = 1
            self.npoint = 1


        grouped_f = pu.index_points(f, grouped_idx)  # [B, S, n, cin]
        grouped_xyz = pu.index_points(xyz, grouped_idx)  # [B, S, n, 3]


        grouped_xyz = grouped_xyz - new_xyz.view(B, S, 1, 3).repeat(1, 1, n, 1) # [B, S, n, 3] Local pos

        
        grouped_f = torch.cat([grouped_xyz, grouped_f], -1)  # [B, S, n, 3+cin]

        framepoints = self.rotated_framepoints  # [V, 3]
        # framepoints = self.framepoints

        grouped_xyz = grouped_xyz.view(B * S * n, 3)  # [B*S*n,3]
        grouped_xyz = torch.cat([grouped_xyz, framepoints], 0)  # [B*N*nsample+V,3]
        grouped_xyz = grouped_xyz.view(B * S * n + V, 1, 3).repeat(1, V, 1)  # [B*S*n+V, V, 3]

        cocated_group = torch.cat([grouped_xyz, framepoints.view(1, V, 3).repeat(B * S * n + V, 1, 1)], -1).view(B * S * n + V, V, 6)  # [B*S*n+V, V, 6]

        a = cocated_group.transpose(1, 2)  # [B*S*n+V, 6, V] 转变成channel first

        # m1 in the paper
        for i, conv in enumerate(self.m1_filter):
            bn = self.m1_bn[i]
            a = F.relu(bn(conv(a)))
            # [B*S*n+V, 1, V]

        
        w = w.view(1, V, self.cin * self.cout).transpose(1, 2) # [1, in*out, V]

        # m2 in the paper
        for i, conv in enumerate(self.m2_filter):
            bn = self.m2_bn[i]
            w = F.relu(bn(conv(w)))
            # [1, mid, V]

        a = a.transpose(1, 2) # [B*S*n+V, V, 1]
        w = w.transpose(1, 2).view(V, self.mid)
        wl1 = w  # [V, mid]

        w = a * w  # [B*S*n+V,V,mid]
        w = torch.sum(w, 1)  # [B*S*n+V,mid]
        wl2 = w[-V:]  # [V, mid]
        w = w[:-V]  # [B*S*n, mid]
        w = w.view(B, S, n, self.mid)  # [B, S, n, mid]
        w[mask] = 0.0
        grouped_f[mask] = 0.0

        f = grouped_f.transpose(2, 3)  # [B, S, cin+3, n]
        f = torch.matmul(f, w)  # [B, S, cin, mid]

        # mr in the paper
        f = f.view(B, S, self.cin * self.mid).transpose(1, 2)  # [B, in*mid, S]
        for i, conv in enumerate(self.mr_filter):
            bn = self.mr_bn[i]
            f = F.relu(bn(conv(f)))
            # [B, out, S]
        f = f.view(B, S, self.cout) #修改为channel last

        # loss2
        # loss2 = torch.sum((wl2 - wl1) ** 2) / (V * self.mid)

        return f, new_xyz#, loss2

    # Random rotate the frame points
    def rotate_frame_points(self, framepoints):
        # define rotation angle
        rotation_angle = torch.rand(1, device=self.device) * 2 * np.pi
        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)
        # the rotation matrix
        rotation_matrix = torch.tensor([[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]], device=self.device)
        # rotate the frame point
        result = torch.matmul(framepoints, rotation_matrix)
        return result
