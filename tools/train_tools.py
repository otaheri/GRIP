
# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#


import torch
import numpy as np
# import chamfer_distance as chd
#
# def point2point_signed(
#         x,
#         y,
#         x_normals=None,
#         y_normals=None,
# ):
#     """
#     signed distance between two pointclouds
#
#     Args:
#         x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
#             with P1 points in each batch element, batch size N and feature
#             dimension D.
#         y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
#             with P2 points in each batch element, batch size N and feature
#             dimension D.
#         x_normals: Optional FloatTensor of shape (N, P1, D).
#         y_normals: Optional FloatTensor of shape (N, P2, D).
#
#     Returns:
#
#         - y2x_signed: Torch.Tensor
#             the sign distance from y to x
#         - y2x_signed: Torch.Tensor
#             the sign distance from y to x
#         - yidx_near: Torch.tensor
#             the indices of x vertices closest to y
#
#     """
#
#
#     N, P1, D = x.shape
#     P2 = y.shape[1]
#
#     if y.shape[0] != N or y.shape[2] != D:
#         raise ValueError("y does not have the correct shape.")
#
#     ch_dist = chd.ChamferDistance()
#
#     x_near, y_near, xidx_near, yidx_near = ch_dist(x,y)
#
#     xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
#     x_near = y.gather(1, xidx_near_expanded)
#
#     yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
#     y_near = x.gather(1, yidx_near_expanded)
#
#     x2y = x - x_near
#     y2x = y - y_near
#
#     if x_normals is not None:
#         y_nn = x_normals.gather(1, yidx_near_expanded)
#         in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
#         y2x_signed = y2x.norm(dim=2) * in_out
#
#     else:
#         y2x_signed = y2x.norm(dim=2)
#
#     if y_normals is not None:
#         x_nn = y_normals.gather(1, xidx_near_expanded)
#         in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
#         x2y_signed = x2y.norm(dim=2) * in_out_x
#     else:
#         x2y_signed = x2y.norm(dim=2)
#
#     return y2x_signed, x2y_signed, yidx_near


def v2v(x, y, mean=False):
    dist = (x - y).pow(2).sum(dim=-1).sqrt()
    if mean:
        return dist.mean()
    else:
        return dist

def v2v_2(x, y, mean=False):
    """
    Compute vertex-to-vertex distance between two point clouds x and y.
    
    :param x: Tensor of shape (..., N, 3) representing point clouds each with N points of dimension 3 (XYZ).
    :param y: Tensor of shape (..., M, 3) representing point clouds each with M points of dimension 3 (XYZ).
    :param mean: If True, return the mean vertex-to-vertex distance, otherwise return all distances.
    :return: Tensor representing the pairwise distances, or if mean is True, representing the mean distances.
    """
    
    dist = torch.norm(x - y, dim=-1)
    if mean:
        return torch.mean(dist)
    else:
        return dist
    

import torch.fft
import scipy.stats

def derivative_analysis(x, y, mean=False):
    velocity_x = x[1:, :, :, :] - x[:-1, :, :, :]
    velocity_y = y[1:, :, :, :] - y[:-1, :, :, :]
    
    acceleration_x = velocity_x[1:, :, :, :] - velocity_x[:-1, :, :, :]
    acceleration_y = velocity_y[1:, :, :, :] - velocity_y[:-1, :, :, :]
    
    velocity_mse = torch.mean((velocity_x - velocity_y)**2)
    acceleration_mse = torch.mean((acceleration_x - acceleration_y)**2)
    
    combined_metric = velocity_mse + acceleration_mse
    
    return combined_metric.mean() if mean else combined_metric

def total_variation(x, y, mean=False):
    tv_x = torch.sum(torch.abs(x[1:, :, :, :] - x[:-1, :, :, :]), dim=0)
    tv_y = torch.sum(torch.abs(y[1:, :, :, :] - y[:-1, :, :, :]), dim=0)
    
    tv_diff = torch.abs(tv_x - tv_y)
    
    return tv_diff.mean() if mean else tv_diff

def power_spectral_density(x, y, mean=False):
    fft_x = torch.fft.fft(x, dim=0)  # FFT over frames
    fft_y = torch.fft.fft(y, dim=0)
    
    psd_x = torch.abs(fft_x)**2
    psd_y = torch.abs(fft_y)**2
    
    psd_diff = psd_x - psd_y
    
    return psd_diff.mean() if mean else psd_diff

def spectral_entropy(x, y, mean=False):
    fft_x = torch.fft.fft(x, dim=0)
    fft_y = torch.fft.fft(y, dim=0)
    
    psd_x = torch.abs(fft_x)**2
    psd_y = torch.abs(fft_y)**2
    
    psd_x /= torch.sum(psd_x, dim=0, keepdim=True)
    psd_y /= torch.sum(psd_y, dim=0, keepdim=True)
    
    entropy_x = -torch.sum(psd_x * torch.log(psd_x), dim=0)
    entropy_y = -torch.sum(psd_y * torch.log(psd_y), dim=0)
    
    entropy_diff = torch.abs(entropy_x - entropy_y)
    
    return entropy_diff.mean() if mean else entropy_diff



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class WeightAnneal:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, start_w=0, end_w=1, start_batch=0, end_batch=1, **kwargs):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.start_w = start_w
        self.end_w = end_w
        self.start_batch = start_batch
        self.end_batch = end_batch

    def __call__(self, current_epoch):

        if current_epoch >= self.start_batch:
            anneal_weight = abs((current_epoch - self.start_batch) / (self.end_batch - self.start_batch))
        else:
            return 0.0

        anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight

        return self.start_w + anneal_weight * (self.end_w - self.start_w)