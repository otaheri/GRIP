
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
import sys, os

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from tools.utils import rotmat2aa
from tools.utils import d62rotmat
from tools.utils import batch_to
# from tools.train_tools import point2point_signed
cdir = os.path.dirname(sys.argv[0])


class ResBlock(nn.Module):

    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout

class hand_net(nn.Module):
    def __init__(self,
                 n_neurons=2048,
                 dec_in = 7543,
                 out_frames = 1,
                 drop_out = 0.3,
                 **kwargs):
        super().__init__()

        self.out_frames = out_frames

        self.dec_bn1 = nn.BatchNorm1d(dec_in)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons//2)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 2* 15 * 6*out_frames)

        self.dec_dist = nn.Linear(n_neurons, 2*99*out_frames)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

        self.f_ids = torch.from_numpy(np.load(f'{cdir}/../consts/feet_verts_ids_0512.npy')).to(torch.long)

    def forward(self, dec_x):

        X0 = self.dec_bn1(dec_x)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        X  = self.dec_rb3(X)
        X = self.dout(X)
        X = self.dec_rb4(X)

        # pose = self.sig(self.dec_pose(X))
        pose = self.dec_pose(X)
        int_field = self.sig(self.dec_dist(X))
        result = {'pose': pose, 'int_field': int_field}

        return result

class hand_net_emb(nn.Module):
    def __init__(self,
                 n_neurons=2048,
                 dec_in = 7543,
                 out_frames = 1,
                 drop_out = 0.3,
                 embed_dim = 256,
                 embed_active='ll',
                 **kwargs):
        super().__init__()

        self.out_frames = out_frames
        self.embed_dim = embed_dim
        self.embed_active = embed_active

        self.dec_bn1 = nn.BatchNorm1d(dec_in)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons//2)
        self.enc = nn.Linear(n_neurons//2, out_frames*embed_dim)
        self.enc_bn = nn.BatchNorm1d(out_frames*embed_dim)
        self.dec = nn.Linear(embed_dim, n_neurons//2)
        self.dec_bn = nn.BatchNorm1d(n_neurons//2)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons)

        # self.dec_pose = nn.Linear(n_neurons, 2* 15 * 6*out_frames)
        self.dec_pose = nn.Linear(n_neurons, 2* 15 * 6)

        # self.dec_trans = nn.Linear(n_neurons, 3*out_frames)
        # self.dec_xyz = nn.Linear(n_neurons, 2*99*3* out_frames)

        # self.dec_dist = nn.Linear(n_neurons, 2*99*out_frames)
        self.dec_dist = nn.Linear(n_neurons, 2*99)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()
        self.ll = nn.LeakyReLU(negative_slope=0.2)


        self.f_ids = torch.from_numpy(np.load(f'{cdir}/../consts/feet_verts_ids_0512.npy')).to(torch.long)

    def forward(self, dec_x):

        X0 = self.dec_bn1(dec_x)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        if self.embed_active=='ll':
            X_enc = self.ll(self.enc_bn(self.enc(X))).reshape(-1, self.out_frames, self.embed_dim)
        else:
            X_enc = self.sig(self.enc_bn(self.enc(X))).reshape(-1, self.out_frames, self.embed_dim)

        X_enc[:,1:] = X_enc[:,1:] + X_enc[:,:1]
        X_enc = X_enc.reshape(-1, self.embed_dim)

        X_dec = self.ll(self.dec_bn(self.dec(X_enc)))

        X  = self.dec_rb3(X_dec)
        X = self.dout(X)
        X = self.dec_rb4(X)

        # pose = self.sig(self.dec_pose(X))
        pose = self.dec_pose(X)
        int_field = self.sig(self.dec_dist(X))
        result = {'pose': pose, 'int_field': int_field}

        return result


class arm_net_emb(nn.Module):
    def __init__(self,
                 n_neurons=2048,
                 dec_in = 7543,
                 out_frames = 1,
                 drop_out = 0.3,
                 embed_dim = 64,
                 embed_active='ll',
                 **kwargs):
        super().__init__()

        self.out_frames = out_frames
        self.embed_dim = embed_dim
        self.embed_active = embed_active

        self.dec_bn1 = nn.BatchNorm1d(dec_in)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons//2)
        self.enc = nn.Linear(n_neurons//2, out_frames*embed_dim)
        self.enc_bn = nn.BatchNorm1d(out_frames*embed_dim)
        self.dec = nn.Linear(embed_dim, n_neurons//2)
        self.dec_bn = nn.BatchNorm1d(n_neurons//2)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons)

        # self.dec_pose = nn.Linear(n_neurons, 2* 15 * 6*out_frames)
        self.dec_pose = nn.Linear(n_neurons, 2* 3 * 4)

        # self.dec_trans = nn.Linear(n_neurons, 3*out_frames)
        self.dec_xyz = nn.Linear(n_neurons, 2*99*3)

        # self.dec_dist = nn.Linear(n_neurons, 2*99*out_frames)
        self.dec_dist = nn.Linear(n_neurons, 2*99)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()
        self.ll = nn.LeakyReLU(negative_slope=0.2)


        self.f_ids = torch.from_numpy(np.load(f'{cdir}/../consts/feet_verts_ids_0512.npy')).to(torch.long)

    def forward(self, dec_x):

        X0 = self.dec_bn1(dec_x)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        if self.embed_active=='ll':
            X_enc = self.ll(self.enc_bn(self.enc(X))).reshape(-1, self.out_frames, self.embed_dim)
        else:
            X_enc = self.sig(self.enc_bn(self.enc(X))).reshape(-1, self.out_frames, self.embed_dim)

        rel_glob = False
        rel_rel = True
        cumsum= True
        
        if rel_glob:
            X_enc[:,1:] = X_enc[:,1:] + X_enc[:,:1]
        elif rel_rel:
            if not cumsum:
                X_enc[:,1:] = X_enc[:,1:] + X_enc[:,:-1]
            else:
                X_enc_cumsum = torch.cumsum(X_enc, dim=1)
                X_enc[:,1:] = X_enc[:,1:] + X_enc_cumsum[:,:-1]
                    
        # X_enc[:,1:] = X_enc[:,1:] + X_enc[:,:1]
        X_enc = X_enc.reshape(-1, self.embed_dim)

        X_dec = self.ll(self.dec_bn(self.dec(X_enc)))

        X  = self.dec_rb3(X_dec)
        X = self.dout(X)
        X = self.dec_rb4(X)

        # pose = self.sig(self.dec_pose(X))
        pose = self.dec_pose(X)
        int_field = self.sig(self.dec_dist(X))
        xyz = self.dec_xyz(X)
        result = {'pose': pose, 'int_field': int_field, 'xyz': xyz}

        return result
    

class mnet_model(nn.Module):
    def __init__(self,
                 n_neurons=2048,
                 dec_in = 7543,
                 out_frames = 10,
                 drop_out = 0.3,
                 **kwargs):
        super().__init__()

        self.out_frames = out_frames

        self.dec_bn1 = nn.BatchNorm1d(dec_in)  # normalize the bps_torch for object
        self.dec_rb1 = ResBlock(dec_in, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + dec_in, n_neurons//2)
        self.dec_rb3 = ResBlock(n_neurons//2, n_neurons//2)
        self.dec_rb4 = ResBlock(n_neurons//2, n_neurons)

        self.dec_pose = nn.Linear(n_neurons, 55 * 6* out_frames)

        self.dec_trans = nn.Linear(n_neurons, 3*out_frames)

        self.dec_xyz = nn.Linear(n_neurons, 400*3* out_frames)

        self.dec_dist = nn.Linear(n_neurons, 99*3* out_frames)

        self.dout = nn.Dropout(p=drop_out, inplace=False)
        self.sig = nn.Sigmoid()

        self.f_ids = torch.from_numpy(np.load(f'{cdir}/../consts/feet_verts_ids_0512.npy')).to(torch.long)

    def forward(self, dec_x):

        X0 = self.dec_bn1(dec_x)
        X  = self.dec_rb1(X0, True)
        X  = self.dout(X)
        X  = self.dec_rb2(torch.cat([X0, X], dim=1), True)
        # X  = self.dec_rb2(X)
        X = self.dout(X)
        X  = self.dec_rb3(X)
        X = self.dout(X)
        X = self.dec_rb4(X)

        # pose = self.sig(self.dec_pose(X))
        pose = self.dec_pose(X)
        trans = self.dec_trans(X)

        xyz = self.dec_xyz(X)
        rh2last = self.dec_dist(X)

        return pose, trans, xyz, rh2last
#############################################

###################################################################################

def parms_decode_full(pose,trans):

    bs = trans.shape[0]

    pose_full = d62rotmat(pose)
    pose = pose_full.reshape([bs, 1, -1, 9])
    pose = rotmat2aa(pose).reshape(bs, -1)

    body_parms = full2bone(pose,trans)
    pose_full = pose_full.reshape([bs, -1, 3, 3])
    body_parms['fullpose'] = pose_full

    return body_parms

def full2bone(pose,trans):

    bs = trans.shape[0]
    if pose.ndim>2:
        pose = pose.reshape([bs, 1, -1, 9])
        pose = rotmat2aa(pose).view(bs, -1)

    global_orient = pose[:, :3]
    body_pose = pose[:, 3:66]
    jaw_pose  = pose[:, 66:69]
    leye_pose = pose[:, 69:72]
    reye_pose = pose[:, 72:75]
    left_hand_pose = pose[:, 75:120]
    right_hand_pose = pose[:, 120:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms


def parms_decode(pose,trans):

    bs = trans.shape[0]

    pose_full = d62rotmat(pose)
    pose = pose_full.view([bs, 1, -1, 9])
    pose = rotmat2aa(pose).view(bs, -1)

    global_orient = pose[:, :3]
    body_pose = pose[:, 3:66]
    left_hand_pose = pose[:, 66:111]
    right_hand_pose = pose[:, 111:]
    pose_full = pose_full.view([bs, -1, 3, 3])

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'fullpose': pose_full, 'transl': trans }

    return body_parms
