import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
#from linalg_utils import pdist2, PDist2Order
from collections import namedtuple
#import pytorch_utils as pt_utils
#import my_point_utils as point_utils
from typing import List, Tuple
import os


#from _ext import pointnet2
"""
try:
    import utils._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )
"""
os.environ['LD_LIBRARY_PATH'] = '/home/lly/miniconda3/lib/python3.8/site-packages/torch/lib'
try:
    import utils._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os
if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *



class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(
            X, theta, self.train, self.inplace
        )


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        """B, N, _ = xyz.size()

        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output
        )
        return output"""
        return _ext.furthest_point_sampling(xyz, npoint).long()

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()

        #output = torch.cuda.FloatTensor(B, C, npoint)

        #pointnet2.gather_points_wrapper(
        #    B, C, N, npoint, features, idx, output
        #)

        ctx.for_backwards = (idx, C, N)

        #return output
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data
        )"""
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        """
        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)
        """
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(
            ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        """output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(
            B, c, m, n, features, idx, weight, output
        )

        return output"""
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        #grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())

        #grad_out_data = grad_out.data.contiguous()
        #pointnet2.three_interpolate_grad_wrapper(
        #    B, c, n, m, grad_out_data, idx, weight, grad_features.data
        #)
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of points to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of points to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        #output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        #pointnet2.group_points_wrapper(
        #    B, C, N, nfeatures, nsample, features, idx, output
        #)

        ctx.for_backwards = (idx, N)
        return _ext.group_points(features, idx)
        #return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        #B, C, npoint, nsample = grad_out.size()
        #grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        #grad_out_data = grad_out.data.contiguous()
        #pointnet2.group_points_grad_wrapper(
        #   B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data
        #)
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(
            ctx, radius: float, nsample: int, xyz: torch.Tensor,
            new_xyz: torch.Tensor, fps_idx: torch.IntTensor
    ) -> torch.Tensor:
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        """B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(
            B, N, npoint, radius, nsample, new_xyz, xyz, fps_idx, idx
        )
        """
        idx = _ext.ball_query(new_xyz, xyz, radius, nsample)
        return torch.cat([fps_idx.unsqueeze(2), idx], dim = 2)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup2(nn.Module):  # 就是修改这里了！
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None,
            fps_idx: torch.IntTensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz, fps_idx)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(
            xyz_trans, idx
        )  # (B, 3, npoint, nsample)
        """xyz_flipped = xyz.permute(0,2,1).contiguous()
        new_xyz_flipped = new_xyz.permute(0,2,1).contiguous()
        idx = point_utils.query_ball_point(self.radius, self.nsample, xyz_flipped, new_xyz_flipped)
        grouped_xyz = point_utils.index_points(xyz_flipped, idx) # (B, 3, npoint, nsample)"""
        raw_grouped_xyz = grouped_xyz
        grouped_xyz -= new_xyz.transpose(1,2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            #grouped_features = point_utils.index_points(features, idx)
            if self.use_xyz:
                new_features = torch.cat([raw_grouped_xyz, grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3 + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim = 1)

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class QueryAndGroup(nn.Module):  # 就是修改这里了！
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, npoint: int,nsamples: List[int], use_xyz: bool = True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsamples, self.use_xyz = radius, nsamples, use_xyz
        self.npoint = npoint  # 多少个采样点
        if self.npoint is not None:
            print()
            self.sampling = pointsampling.get_model(npoint=npoint)

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None,
            fps_idx: torch.IntTensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        new_features = []
        # if GroupAll
        if self.npoint is None:
            #features = features.transpose(1,2)  #[B, C, N]

            grouped_xyz = xyz.transpose(1, 2).unsqueeze(2) 

            if features is not None:
                grouped_features = features.unsqueeze(2)

                if self.use_xyz:
                    new_features.append(torch.cat([grouped_xyz, grouped_features],
                                            dim=1))  # (B, 3 + C, 1, N)
                else:
                    new_features.append(grouped_features)
            else:
                new_features.append(grouped_xyz)

            return None, new_features

        device = xyz.device
        B, N, C = xyz.shape
        S = self.npoint
        net = self.sampling(point_xyz=xyz).contiguous()  #[B, N, S]

        sorted, indices = torch.sort(net, 1 ,descending=True)
        centers = indices[:, 0, :]
        new_xyz = index_points(xyz, centers)
        
        for nsample in self.nsamples:
            group_idx = indices[:, 0:nsample, :].transpose(1,2) #[B, S, nsample]
            group_xyz = index_points(xyz, group_idx)
            raw_grouped_xyz = group_xyz
            group_xyz = group_xyz - group_xyz[:,:,0,:].view(B,S,1,C).repeat([1,1,nsample,1]) #[B,S,n,3]
            torch.cuda.empty_cache() 
            if features is not None:
                features = features.permute(0,2,1)
                grouped_features = index_points(features, group_idx)
                if self.use_xyz:
                    new_feature = torch.cat([raw_grouped_xyz, group_xyz, grouped_features],
                                            dim=3) # (B, npoint, nsample, C + 3 + 3)
                else:
                    new_feature = grouped_features
            else:
                assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
                new_feature = torch.cat([raw_grouped_xyz, group_xyz], dim = 3)
            new_features.append(new_feature.permute(0,3,1,2))
        return new_xyz, new_features

