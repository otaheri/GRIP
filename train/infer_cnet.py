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



import os
import re
import shutil
import sys

from zmq import device
sys.path.append('/is/ps2/otaheri/code_repos/GRIP_adobe/project/')
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import argparse

from smplx import SMPLXLayer


from datetime import datetime
from tools.train_tools import EarlyStopping


from torch import nn, optim

from pytorch3d.structures import Meshes
from tensorboardX import SummaryWriter

import glob, time

from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.lines import Lines

from psbody.mesh.colors import name_to_rgb
from tools.objectmodel import ObjectModel

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video
from loguru import logger

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate

from tools.utils import smplx_loc2glob

from bps_torch.bps import bps_torch


from omegaconf import OmegaConf

from models.nets import hand_net_emb, hand_net

from losses import build_loss
from optimizers import build_optimizer
from data.dataloader_static import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full, parms_decode_full
from tools.train_tools import v2v, WeightAnneal
from tqdm import tqdm

from tools.vis_tools import sp_animation, get_ground, points_to_spheres
from tools.utils import LOGGER_DEFAULT_FORMAT
cdir = os.path.dirname(sys.argv[0])

import trimesh


class Trainer:

    def __init__(self,cfg, inference=False):

        
        self.dtype = torch.float32
        self.cfg = cfg
        self.is_inference = inference

        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, '%s_%s.log' % (cfg.expr_ID, 'train' if not inference else 'test')), isfile=True)

        logger.add(logger_path,  backtrace=True, diagnose=True)
        logger.add(lambda x:x,
                   level=cfg.logger_level.upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info

        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        self.logger('[%s] - Started training XXX, experiment code %s' % (cfg.expr_ID, starttime))
        self.logger('tensorboard --logdir=%s' % summary_logdir)
        self.logger('Torch Version: %s\n' % torch.__version__)

        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0], os.path.join(cfg.work_dir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(stime, '%Y%m%d_%H%M'))))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device(f"cuda:{cfg.cuda_id}" if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = cfg.num_gpus
        if use_cuda:
            self.logger('Using %d CUDA cores [gpu_id: %d] [%s] for training!' % (gpu_count,cfg.cuda_id, gpu_brand))

        self.data_info = {}
        self.load_data(cfg, inference)


        self.body_model_cfg = cfg.body_model

        self.predict_offsets = cfg.get('predict_offsets', False)
        self.logger(f'Predict offsets: {self.predict_offsets}')

        self.use_exp = cfg.get('use_exp', 0)
        self.logger(f'Use exp function on distances: {self.use_exp}')

        model_path = os.path.join(self.body_model_cfg.get('model_path', 'data/models'), 'smplx')

        self.body_model = SMPLXLayer(
            model_path=model_path,
            gender='neutral',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.female_model = SMPLXLayer(
            model_path=model_path,
            gender='female',
            use_pca=False,
            flat_hand_mean=True,
        ).to(self.device)
        self.male_model = SMPLXLayer(
            model_path=model_path,
            gender='male',
            use_pca=False,
            flat_hand_mean=True,
        ).to(self.device)

        self.object_model = ObjectModel().to(self.device)

        if 'future' in cfg.expr_ID:
            self.fu = int(self.cfg.expr_ID.split('future_')[-1]) + 1
        else:
            self.fu = 10
        
        if not inference:
            cfg.network.hand_net.dec_in -= (10-self.fu)*4
        # Create the network
        self.network = hand_net_emb(**cfg.network.hand_net).to(self.device)
        self.T = cfg.network.hand_net.out_frames
        
        ref_cfg = cfg.network.ref_net
        ref_cfg.dec_in = 1890
        self.network_ref = hand_net(**ref_cfg).to(self.device)

        # Setup the training losses
        self.loss_setup()


        if cfg.num_gpus > 1:
            self.network = nn.DataParallel(self.network)
            self.network_ref = nn.DataParallel(self.network_ref)
            self.logger("Training on Multiple GPU's")

        vars_network = [var[1] for var in self.network.named_parameters()]
        vars_network += [var[1] for var in self.network_ref.named_parameters()]

        n_params = sum(p.numel() for p in vars_network if p.requires_grad)

        self.logger('Total Trainable Parameters for network is %2.2f M.' % ((n_params) * 1e-6))

        self.configure_optimizers()

        self.best_loss = np.inf
        self.best_loss_ref = np.inf

        self.epochs_completed = 0
        self.cfg = cfg
        self.network.cfg = cfg

        if inference and cfg.best_model is None:
            cfg.best_model = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        if cfg.best_model is not None:
            self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
            self.logger('Restored hnet trained model from %s' % cfg.best_model)
        
        if inference and cfg.best_model_ref is None:
            cfg.best_model_ref = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model_ref.pt')))[-1]
        if cfg.best_model_ref is not None:
            self._get_network(self.network_ref).load_state_dict(torch.load(cfg.best_model_ref, map_location=self.device), strict=False)
            self.logger('Restored hnet_ref trained model from %s' % cfg.best_model_ref)

        if cfg.resume and not inference:
            best_model = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))
            epoch_hnet = epoch_hnet_ref = 0
            if len(best_model) > 0:
                cfg.best_model = best_model[-1]
                self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
                self.logger('Restored hnet trained model from %s' % cfg.best_model)
                epoch_hnet = int(cfg.best_model.split('/')[-1].split('_')[0][2:])
        
            best_model_ref = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model_ref.pt')))
            if len(best_model_ref) > 0:
                cfg.best_model_ref = best_model_ref[-1]
                self._get_network(self.network_ref).load_state_dict(torch.load(cfg.best_model_ref, map_location=self.device), strict=False)
                self.logger('Restored hnet_ref trained model from %s' % cfg.best_model_ref)
                epoch_hnet_ref = int(cfg.best_model_ref.split('/')[-1].split('_')[0][2:])

            # self.epochs_completed = max(epoch_hnet, epoch_hnet_ref)

        self.bps_torch = bps_torch()
        self.sig = nn.Sigmoid()

    def loss_setup(self):

        self.logger('Configuring the losses!')

        loss_cfg = self.cfg.get('losses', {})

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        self.Lossbce = nn.BCELoss(reduction='mean')

        # Edge loss
        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.edge_loss_weight = edge_loss_cfg.get('weight', 0.0)
        self.logger(f'Edge loss, weight: {self.edge_loss}, {self.edge_loss_weight}')

        # Vertex loss
        # TODO: Add denser vertex sampling
        vertex_loss_cfg = loss_cfg.get('vertices', {})
        self.vertex_loss_weight = vertex_loss_cfg.get('weight', 0.0)
        self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex loss, weight: {self.vertex_loss},'
                    f' {self.vertex_loss_weight}')

        vertex_consist_loss_cfg = loss_cfg.get('vertices_consist', {})
        self.vertex_consist_loss_weight = vertex_consist_loss_cfg.get('weight', 0.0)
        # self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex consist loss weight: {self.vertex_consist_loss_weight}')

        rh_vertex_loss_cfg = loss_cfg.get('rh_vertices', {})
        self.rh_vertex_loss_weight = rh_vertex_loss_cfg.get('weight', 0.0)
        self.rh_vertex_loss = build_loss(**rh_vertex_loss_cfg)
        self.logger(f'Right Hand Vertex loss, weight: {self.rh_vertex_loss},'
                     f' {self.rh_vertex_loss_weight}')

        feet_vertex_loss_cfg = loss_cfg.get('feet_vertices', {})
        self.feet_vertex_loss_weight = feet_vertex_loss_cfg.get('weight', 0.0)
        self.feet_vertex_loss = build_loss(**feet_vertex_loss_cfg)
        self.logger(f'Feet Vertex loss, weight: {self.feet_vertex_loss},'
                     f' {self.feet_vertex_loss_weight}')

        pose_loss_cfg = loss_cfg.get('pose', {})
        self.pose_loss_weight = pose_loss_cfg.get('weight', 0.0)
        self.pose_loss = build_loss(**pose_loss_cfg)
        self.logger(f'Pose loss, weight: {self.pose_loss},'
                    f' {self.pose_loss}')

        velocity_loss_cfg = loss_cfg.get('velocity', {})
        self.velocity_loss_weight = velocity_loss_cfg.get('weight', 0.0)
        self.velocity_loss = build_loss(**velocity_loss_cfg)

        self.logger(f'Velocity loss, weight: {self.velocity_loss},'
                    f' {self.velocity_loss_weight}')

        acceleration_loss_cfg = loss_cfg.get('acceleration', {})
        self.acceleration_loss_weight = acceleration_loss_cfg.get('weight', 0.0)
        self.acceleration_loss = build_loss(**acceleration_loss_cfg)
        self.logger(
            f'Acceleration loss, weight: {self.acceleration_loss},'
            f' {self.acceleration_loss_weight}')

        contact_loss_cfg = loss_cfg.get('contact', {})
        self.contact_loss_weight = contact_loss_cfg.get('weight', 0.0)
        self.logger(
            f'Contact loss, weight: '
            f' {self.contact_loss_weight}')

        kl_loss_cfg = loss_cfg.get('kl_loss', {})
        self.kl_loss_weight = kl_loss_cfg.get('weight', 0.0)
        self.logger(
            f'KL loss, weight: '
            f' {self.kl_loss_weight}')

        self.verts_ids = to_tensor(np.load(f'{cdir}/../consts/verts_ids_0512.npy'), dtype=torch.long)
        self.rhand_idx = torch.from_numpy(np.load(f'{cdir}/../consts/rhand_smplx_ids.npy'))
        self.rh_verts_ids = to_tensor(np.load(f'{cdir}/../consts/rhand_smplx_ids.npy'), dtype=torch.long)
        self.rh_faces_ids = to_tensor(np.load(f'{cdir}/../consts/rhand_faces.npy'), dtype=torch.long)

        self.lhand_idx = torch.from_numpy(np.load(f'{cdir}/../consts/lhand_smplx_ids.npy'))
        self.lh_verts_ids = to_tensor(np.load(f'{cdir}/../consts/lhand_smplx_ids.npy'), dtype=torch.long)
        self.lh_faces_ids = to_tensor(np.load(f'{cdir}/../consts/lhand_faces.npy'), dtype=torch.long)

        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)
        self.rh_ids_sampled1 = torch.tensor(np.where([id in self.verts_ids for id in self.rhand_idx])[0]).to(torch.long)
        self.lh_ids_sampled = torch.tensor(np.where([id in self.lhand_idx for id in self.verts_ids])[0]).to(torch.long)
        self.lh_ids_sampled1 = torch.tensor(np.where([id in self.verts_ids for id in self.lhand_idx])[0]).to(torch.long)

    def load_data(self,cfg, inference):

        self.logger('Base dataset_dir is %s' % self.cfg.datasets.dataset_dir)

        ds_name = 'test'
        self.data_info[ds_name] = {}
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_test.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_test.frame_sbjs
        self.data_info[ds_name]['frame_objs'] = ds_test.frame_objs
        # self.data_info['body_vtmp'] = ds_test.sbj_vtemp
        # self.data_info['body_betas'] = ds_test.sbj_betas
        # self.data_info['obj_verts'] = ds_test.obj_verts
        self.data_info['obj_info'] = ds_test.obj_info
        self.data_info['sbj_info'] = ds_test.sbj_info
        self.ds_test = build_dataloader(ds_test, split='test', cfg=self.cfg.datasets, batch_size=self.cfg.datasets.batch_size_test)

        if not inference:
        # if True:


            ds_name = 'train'
            self.data_info[ds_name] = {}
            ds_train = LoadData(self.cfg.datasets, split_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_train.frame_names
            self.data_info[ds_name]['frame_sbjs'] = ds_train.frame_sbjs
            self.data_info[ds_name]['frame_objs'] = ds_train.frame_objs
            self.data_info['body_vtmp'] = ds_train.sbj_vtemp
            self.data_info['body_betas'] = ds_train.sbj_betas
            self.data_info['obj_verts'] = ds_train.obj_verts
            # self.ds_train = build_dataloader(ds_train, split=ds_name, cfg=self.cfg.datasets, batch_size=self.cfg.datasets.batch_size)
            self.ds_train = build_dataloader(ds_train, split=ds_name, cfg=self.cfg.datasets, batch_size=self.cfg.datasets.batch_size)

        ds_name = 'val'
        self.data_info[ds_name] = {}
        ds_val = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_val.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_val.frame_sbjs
        self.data_info[ds_name]['frame_objs'] = ds_val.frame_objs
        self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets, batch_size=self.cfg.datasets.batch_size)

        self.bps = ds_test.bps
        self.bps_torch = bps_torch()

        self.mean_hand_pose = to_tensor(np.load(f'{cdir}/../consts/mean_hand_pose_rotmat.npy'), dtype=torch.float32).to(self.device)

        if not inference:
            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])
    
    def _get_network(self, model=None):
        if model is None:
            model = self.network
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def save_network(self, model, path):
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), path)
        # torch.save(self.network_ref.module.state_dict() if isinstance(self.network_ref, torch.nn.DataParallel) else self.network_ref.state_dict(), self.cfg.best_model.replace('.pt', '_ref.pt'))

    def forward(self, x):

        ##############################################
        # batch = {k:v.clone() for k,v in x.items()}
        # x     = {k:v[:,-1].clone() for k,v in x.items()}

        i = 1 # current frame
        vox_threshold = 0.005
        nf = self.T

        bs = x['transl'].shape[0]

        enc_x = {}
        

        fullpose = x['fullpose_rotmat'].clone()
        h_pose = torch.eye(3, dtype=fullpose.dtype, device=fullpose.device).reshape(-1,3,3).repeat(bs*nf*2*15,1,1).reshape(bs,nf,2*15,3,3)

        fullpose[:,i:i+nf,25:] = h_pose

        if not 'no_pose' in self.cfg.expr_ID:
            enc_x['fullpose'] = fullpose[:,i:i+nf,1:25,:2,:] # 1: --> ignore the global orientation
        # enc_x['transl'] = x['transl']

        enc_x['betas'] = x['betas'][:,i]




        if self.use_exp != 0 and self.use_exp != -1:
            # enc_x['verts2obj_exp'] = torch.exp(-self.use_exp * x['verts2obj'])
            if not 'no_proximity_sensor' in self.cfg.expr_ID:
                enc_x['rh2obj_exp'] = torch.exp(-self.use_exp * x['rh2obj_h'][:,i:i+nf].norm(dim=-1))
                enc_x['lh2obj_exp'] = torch.exp(-self.use_exp * x['lh2obj_h'][:,i:i+nf].norm(dim=-1))

            # enc_x['bps_obj'] = x['bps_obj_glob'][:,i]
            if not 'no_hand_sensor' in self.cfg.expr_ID:
                enc_x['bps_obj_rh'] = torch.exp(-0.1*self.use_exp * x['bps_obj_rh'][:,i:i+nf])
                enc_x['bps_obj_lh'] = torch.exp(-0.1*self.use_exp * x['bps_obj_lh'][:,i:i+nf])
            
            if 'voxelgrid' in self.cfg.expr_ID:
                enc_x['bps_obj_rh'] = (x['bps_obj_rh'][:,i:i+nf] < vox_threshold).to(torch.float32)
                enc_x['bps_obj_lh'] = (x['bps_obj_lh'][:,i:i+nf] < vox_threshold).to(torch.float32)

        else:
            # enc_x['verts2obj'] = x['verts2obj']
            if not 'no_proximity_sensor' in self.cfg.expr_ID:
                enc_x['rh2obj'] = x['rh2obj_h'][:,i:i+nf].norm(dim=-1)
                enc_x['lh2obj'] = x['lh2obj_h'][:,i:i+nf].norm(dim=-1)

            # enc_x['bps_obj'] = x['bps_obj_glob'][:,i]
            if not 'no_hand_sensor' in self.cfg.expr_ID:
                enc_x['bps_obj_rh'] = x['bps_obj_rh'][:,i:i+nf]
                enc_x['bps_obj_lh'] = x['bps_obj_lh'][:,i:i+nf]
            
            if 'voxelgrid' in self.cfg.expr_ID:
                enc_x['bps_obj_rh'] = (x['bps_obj_rh'][:,i:i+nf] < vox_threshold).to(torch.float32)
                enc_x['bps_obj_lh'] = (x['bps_obj_lh'][:,i:i+nf] < vox_threshold).to(torch.float32)

 
        fu = self.fu
        
        enc_x['rh2obj_h'] = x['rh2obj_h'][:,i:i+fu].norm(dim=-1).mean(dim=-1) #vid 7564
        enc_x['lh2obj_h'] = x['lh2obj_h'][:,i:i+fu].norm(dim=-1).mean(dim=-1)

        enc_x['rh2obj_h_vel'] = x['rh2obj_h_vel'][:,i:i+fu].mean(dim=-1)
        enc_x['lh2obj_h_vel'] = x['lh2obj_h_vel'][:,i:i+fu].mean(dim=-1)

        #####################################################
        enc_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in enc_x.values()], dim=1)

        net_results = self.network(enc_x)
        # rh_pose = self.network(enc_x)
        h_pose = net_results['pose'].reshape(bs,nf,-1)
        int_field = net_results['int_field'].reshape(bs,nf,-1)

        h_pose_rotmat = d62rotmat(h_pose).reshape(bs,nf, -1, 3, 3)

        #make global with respect to frame i
        relative_pose = False
        if relative_pose:
            h_pose_rotmat_f = torch.matmul(h_pose_rotmat[:,1:], h_pose_rotmat[:,:1])
            h_pose_rotmat[:,1:] = h_pose_rotmat_f
            
        pose = fullpose[:,i:i+nf].clone()
        pose[:,:,25:] = h_pose_rotmat

        pose = pose.reshape(bs*nf, -1)
        trans = x['transl'][:,i:i+nf].clone().reshape(bs*nf, -1)
        d62rot = pose.shape[-1] == 2*330
        body_params = parms_6D2full(pose, trans, d62rot= d62rot)

        results = {}
        results['body_params'] = body_params
        results['int_field'] = int_field

        return  results

    def forward_ref(self, x):

        ##############################################

        i = 1 # current frame

        bs = x['transl'].shape[0]

        enc_x = {}

        fullpose = x['fullpose_rotmat'].clone()
        h_pose_init = fullpose[:,25:].clone()

        enc_x['fullpose'] = fullpose[:,1:,:2,:] # 1: --> ignore the global prientation

        enc_x['betas'] = x['betas']


        if self.use_exp != 0 and self.use_exp != -1:
            enc_x['rh2obj_exp'] = torch.exp(-self.use_exp * x['rh2obj'])
            enc_x['lh2obj_exp'] = torch.exp(-self.use_exp * x['lh2obj'])
        else:
            enc_x['rh2obj'] = x['rh2obj']
            enc_x['lh2obj'] = x['lh2obj']


        #####################################################
        enc_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in enc_x.values()], dim=1)

        net_results = self.network_ref(enc_x)
        # rh_pose = self.network(enc_x)
        h_pose = net_results['pose']
        int_field = net_results['int_field']

        h_pose_rotmat = d62rotmat(h_pose).reshape(bs, -1, 3, 3)

        h_pose = torch.matmul(h_pose_rotmat,h_pose_init)
            
        pose = fullpose.clone()
        pose[:,25:] = h_pose_rotmat
        pose = pose.reshape(bs, -1)
        trans = x['transl'].clone().reshape(bs, -1)
        d62rot = pose.shape[-1] == 2*330
        body_params = parms_6D2full(pose, trans, d62rot= d62rot)

        results = {}
        results['body_params'] = body_params
        results['int_field'] = int_field

        return  results

    def prepare_data(self, batch, use_pert=False, use_mean=False):

        cur_id = 1

        genders = batch['gender'][:,cur_id]
        males = genders == 1
        females = ~males

        B = batch['transl_obj'].shape[0]
        device = batch['transl_obj'].device
        v_template = batch['sbj_vtemp'][:,cur_id]

        FN = sum(females)
        MN = sum(males)

        params_gt_c = parms_6D2full(batch['fullpose_rotmat'][:,cur_id],
                                    batch['transl'][:,cur_id],
                                    d62rot=False)

        if use_pert:
            params_gt = parms_6D2full(batch['fullpose_rotmat_p'][:,cur_id-1],
                                        batch['transl_p'][:,cur_id-1],
                                        d62rot=False)
        else:
            if use_mean:
                fullpose = batch['fullpose_rotmat'][:,cur_id-1]
                fullpose[:,25:] = self.mean_hand_pose.clone().repeat(B,2,1,1)
            else:
                fullpose = batch['fullpose_rotmat'][:,cur_id-1]

            params_gt = parms_6D2full(batch['fullpose_rotmat'][:,cur_id-1],
                                        batch['transl'][:,cur_id-1],
                                        d62rot=False)

        params_gt_c['right_hand_pose'] = params_gt['right_hand_pose'].clone()
        params_gt_c['left_hand_pose'] = params_gt['left_hand_pose'].clone()
        
        new_batch_cont = {}


        if FN > 0: 
            
            self.female_model.v_template = v_template[females].clone()

            f_params_gt = {k: v[females].clone().detach() for k, v in params_gt.items()}
            f_output_gt = self.female_model(**f_params_gt)
            f_verts_gt = f_output_gt.vertices
            f_joints_gt = f_output_gt.joints

            f_params_gt_c = {k: v[females].clone().detach() for k, v in params_gt_c.items()}
            f_output_gt_c = self.female_model(**f_params_gt_c)
            f_verts_gt_c = f_output_gt_c.vertices
            f_joints_gt_c = f_output_gt_c.joints


        if MN > 0:
            self.male_model.v_template = v_template[males].clone()

            m_params_gt = {k: v[males].clone() for k, v in params_gt.items()}
            m_output_gt = self.male_model(**m_params_gt)
            m_verts_gt = m_output_gt.vertices
            m_joints_gt = m_output_gt.joints

            m_params_gt_c = {k: v[males].clone() for k, v in params_gt_c.items()}
            m_output_gt_c = self.male_model(**m_params_gt_c)
            m_verts_gt_c = m_output_gt_c.vertices
            m_joints_gt_c = m_output_gt_c.joints

        bps_type = 'deltas'

        new_batch_cont['rh2obj_cont'] = torch.zeros([B, 778,3]).to(device)
        new_batch_cont['lh2obj_cont'] = torch.zeros([B, 778,3]).to(device)

        new_batch_cont['rh2obj_gt'] = torch.zeros([B,778,3]).to(device)
        new_batch_cont['lh2obj_gt'] = torch.zeros([B,778,3]).to(device)

        new_batch_cont['rh2obj_ids'] = torch.zeros([B, 778,3], dtype=torch.long).to(device)
        new_batch_cont['lh2obj_ids'] = torch.zeros([B, 778,3], dtype=torch.long).to(device)

        if FN>0:

            obj_verts_gt = batch['verts_obj'][:,cur_id-1].reshape(B,-1, 3)[females]
            obj_verts_gt_c = batch['verts_obj'][:,cur_id].reshape(B,-1, 3)[females]

            rh2obj = self.bps_torch.encode(x = obj_verts_gt,
                                            feature_type=[bps_type, 'closest'],
                                            custom_basis=f_verts_gt[:,self.rhand_idx])
            
            lh2obj = self.bps_torch.encode(x = obj_verts_gt,
                                            feature_type=[bps_type, 'closest'],
                                            custom_basis=f_verts_gt[:,self.lhand_idx])

            rh2obj_dist = rh2obj[bps_type]
            lh2obj_dist = lh2obj[bps_type]

            new_batch_cont['rh2obj_gt'][females] = rh2obj_dist.clone().detach()
            new_batch_cont['lh2obj_gt'][females] = lh2obj_dist.clone().detach()
            
            # rh2obj_closest = rh2obj['closest'].reshape(-1, 778, 3)
            # lh2obj_closest = lh2obj['closest'].reshape(-1, 778, 3)

            rh2obj_closest_ids = rh2obj['closest_ids']
            lh2obj_closest_ids = lh2obj['closest_ids']

            new_batch_cont['rh2obj_ids'][females] = rh2obj_closest_ids.clone().detach()
            new_batch_cont['lh2obj_ids'][females] = lh2obj_closest_ids.clone().detach()


            rh_f_verts_gt_c = f_verts_gt_c[:,self.rhand_idx].reshape(-1, 778, 3)
            lh_f_verts_gt_c = f_verts_gt_c[:,self.lhand_idx].reshape(-1, 778, 3)

            rh2obj_cont_dist = (rh_f_verts_gt_c - obj_verts_gt_c.gather(1,rh2obj_closest_ids))
            lh2obj_cont_dist = (lh_f_verts_gt_c - obj_verts_gt_c.gather(1,lh2obj_closest_ids))

            new_batch_cont['rh2obj_cont'][females] = rh2obj_cont_dist.clone().detach()
            new_batch_cont['lh2obj_cont'][females] = lh2obj_cont_dist.clone().detach()

        
        if MN>0:
            obj_verts_gt = batch['verts_obj'][:,cur_id-1].reshape(B,-1, 3)[males]
            obj_verts_gt_c = batch['verts_obj'][:,cur_id].reshape(B,-1, 3)[males]

            rh2obj = self.bps_torch.encode(x = obj_verts_gt,
                                            feature_type=[bps_type, 'closest'],
                                            custom_basis=m_verts_gt[:,self.rhand_idx])
            
            lh2obj = self.bps_torch.encode(x = obj_verts_gt,
                                            feature_type=[bps_type, 'closest'],
                                            custom_basis=m_verts_gt[:,self.lhand_idx])

            rh2obj_dist = rh2obj[bps_type]
            lh2obj_dist = lh2obj[bps_type]

            new_batch_cont['rh2obj_gt'][males] = rh2obj_dist.clone().detach()
            new_batch_cont['lh2obj_gt'][males] = lh2obj_dist.clone().detach()

            # rh2obj_closest = rh2obj['closest'].reshape(-1, 778, 3)
            # lh2obj_closest = lh2obj['closest'].reshape(-1, 778, 3)

            rh2obj_closest_ids = rh2obj['closest_ids']
            lh2obj_closest_ids = lh2obj['closest_ids']

            new_batch_cont['rh2obj_ids'][males] = rh2obj_closest_ids.clone().detach()
            new_batch_cont['lh2obj_ids'][males] = lh2obj_closest_ids.clone().detach()

            rh_m_verts_gt_c = m_verts_gt_c[:,self.rhand_idx].reshape(-1, 778, 3)
            lh_m_verts_gt_c = m_verts_gt_c[:,self.lhand_idx].reshape(-1, 778, 3)

            rh2obj_cont_dist = (rh_m_verts_gt_c - obj_verts_gt_c.gather(1,rh2obj_closest_ids))
            lh2obj_cont_dist = (lh_m_verts_gt_c - obj_verts_gt_c.gather(1,lh2obj_closest_ids))

            new_batch_cont['rh2obj_cont'][males] = rh2obj_cont_dist.clone().detach()
            new_batch_cont['lh2obj_cont'][males] = lh2obj_cont_dist.clone().detach()


        return new_batch_cont

    def train(self):

        self.network.train()
        save_every_it = len(self.ds_train) / self.cfg.summary_steps
        train_loss_dict = {}
        train_loss_dict_ref = {}

        for it, batch in enumerate(self.ds_train):
            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            self.optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            output = self.forward(batch)

            loss_total, losses_dict, new_batch, new_batch_p = self.get_loss(batch, it, output)


            if self.fit_hnet:

                loss_total.backward()
                self.optimizer.step()

                train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                    train_msg = self.create_loss_message(cur_train_loss_dict,
                                                        expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed,
                                                        model_name='hnet',
                                                        it=it,
                                                        try_num=0,
                                                        mode='train')

                    self.logger(train_msg)


            if self.fit_ref:
                self.optimizer_ref.zero_grad()
                self.optimizer.zero_grad()

                output_ref = self.forward_ref(new_batch)
                output_ref_p = self.forward_ref(new_batch_p)

                loss_total_ref, losses_dict_ref = self.get_loss_ref(batch, new_batch, it, output_ref)
                loss_total_ref_p, losses_dict_ref_p = self.get_loss_ref(batch, new_batch_p, it, output_ref_p)

                loss_total_ref = (loss_total_ref_p + loss_total_ref)/2
                losses_dict_ref = {k: (losses_dict_ref.get(k, 0.0) + losses_dict_ref_p.get(k, 0.0))/2 for k in losses_dict_ref.keys()}

                loss_total_ref.backward()
                self.optimizer_ref.step()

                train_loss_dict_ref = {k: train_loss_dict_ref.get(k, 0.0) + v.item() for k, v in losses_dict_ref.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_ref = {k: v / (it + 1) for k, v in train_loss_dict_ref.items()}
                    train_msg_ref = self.create_loss_message(cur_train_loss_dict_ref,
                                                        expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed,
                                                        model_name='hnet_ref',
                                                        it=it,
                                                        try_num=0,
                                                        mode='train')

                    self.logger(train_msg_ref)
            
        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        train_loss_dict_ref = {k: v / len(self.ds_train) for k, v in train_loss_dict_ref.items()}

        return train_loss_dict, train_loss_dict_ref

    def evaluate(self, ds_name='val', eval_test=False):
        self.network.eval()
        self.network_ref.eval()

        eval_loss_dict = {}
        eval_loss_dict_ref = {}
        eval_loss_dict_p = {}

        if eval_test:
            self.fit_hnet = True
            self.fit_ref = True

        data = self.ds_val if ds_name == 'val' else self.ds_test

        with torch.no_grad():
            for it, batch in enumerate(data):

                batch = {k: batch[k].to(self.device) for k in batch.keys()}

                self.optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)
                
                output = self.forward(batch)

                loss_total, losses_dict, new_batch, new_batch_p = self.get_loss(batch, it, output, eval_test=eval_test)
                if self.fit_hnet:
                    eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}


                if self.fit_ref:

                    self.optimizer.zero_grad()

                    # new_batch_cont = self.prepare_data(batch)
                    output_ref = self.forward_ref(new_batch)
                    output_ref_p = self.forward_ref(new_batch_p)


                    loss_total_ref, losses_dict_ref = self.get_loss_ref(batch, new_batch, it, output_ref, eval_test=eval_test)
                    loss_total_ref_p, losses_dict_ref_p = self.get_loss_ref(batch, new_batch_p, it, output_ref_p, eval_test=eval_test)


                    eval_loss_dict_ref = {k: eval_loss_dict_ref.get(k, 0.0) + v.item() for k, v in losses_dict_ref.items()}
                    eval_loss_dict_p = {k: eval_loss_dict_p.get(k, 0.0) + v.item() for k, v in losses_dict_ref_p.items()}
                
            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}
            eval_loss_dict_ref = {k: v / len(data) for k, v in eval_loss_dict_ref.items()}
            eval_loss_dict_p = {k: v / len(data) for k, v in eval_loss_dict_p.items()}

            if eval_test:
                return eval_loss_dict, eval_loss_dict_ref, eval_loss_dict_p
            eval_loss_dict_ref = {k: (eval_loss_dict_ref.get(k, 0.0) + eval_loss_dict_p.get(k, 0.0))/2 for k in eval_loss_dict_ref.keys()}
        return eval_loss_dict, eval_loss_dict_ref

    def get_loss(self, batch, batch_idx, results, eval_test=False):

        cur_id = 1
        B = batch['transl_obj'].shape[0]
        T = self.T
        device = batch['transl_obj'].device


        bparams = results['body_params']

        new_batch = {k:v.clone().detach() for k,v in bparams.items()}
        new_batch_p = parms_6D2full(batch['fullpose_rotmat_p'][:,cur_id:cur_id+T].reshape(B*T,-1),
                                  batch['transl_p'][:,cur_id:cur_id+T].reshape(B*T,-1),
                                  d62rot=False)
        params_p = {k:v.clone().detach() for k,v in new_batch_p.items()}

        new_batch['betas'] = batch['betas'][:,cur_id:cur_id+T].clone().reshape(B*T,1,-1)
        new_batch_p['betas'] = batch['betas'][:,cur_id:cur_id+T].clone().reshape(B*T,1,-1)

        genders = batch['gender'][:,cur_id:cur_id+T].reshape(-1)
        males = genders == 1
        females = ~males

        v_template = batch['sbj_vtemp'][:,cur_id:cur_id+T].reshape(B*T,-1,3)

        FN = sum(females)
        MN = sum(males)

        params_gt = parms_6D2full(batch['fullpose_rotmat'][:,cur_id:cur_id+T].reshape(B*T,-1),
                                  batch['transl'][:,cur_id:cur_id+T].reshape(B*T,-1),
                                  d62rot=False)

        if self.fit_ref:
            params_gt_prev = parms_6D2full(batch['fullpose_rotmat'][:,cur_id-1],
                                            batch['transl'][:,cur_id-1],
                                            d62rot=False)


        if FN > 0:
            
            self.female_model.v_template = v_template[females].clone()

            f_params_gt = {k: v[females].clone().detach() for k, v in params_gt.items()}
            f_output_gt = self.female_model(**f_params_gt)
            f_verts_gt = f_output_gt.vertices
            f_joints_gt = f_output_gt.joints

            s = time.time()
            f_params = {k: v[females].clone().detach() for k, v in bparams.items()}
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices
            f_joints = f_output.joints
            e = time.time()
            # print('time for smplx female forward pass: ', e-s)


            if self.fit_ref:

                f_params_p = {k: v[females].clone().detach() for k, v in params_p.items()}
                f_output_p = self.female_model(**f_params_p)
                f_verts_p = f_output_p.vertices
                f_joints_p = f_output_p.joints

                new_batch_p['f_verts'] = f_verts_p.clone().detach()
                new_batch_p['f_joints'] = f_joints_p.clone().detach()
                new_batch_p['f_params_gt'] = f_params_p

                new_batch_p['f_verts_gt'] = f_verts_gt
                new_batch_p['f_joints_gt'] = f_joints_gt

                new_batch['f_verts_gt'] = f_verts_gt
                new_batch['f_joints_gt'] = f_joints_gt
                new_batch['f_verts'] = f_verts.clone().detach()
                new_batch['f_joints'] = f_joints.clone().detach()
                new_batch['f_params_gt'] = f_params_gt

        if MN > 0:
            self.male_model.v_template = v_template[males].clone()

            m_params_gt = {k: v[males].clone() for k, v in params_gt.items()}
            m_output_gt = self.male_model(**m_params_gt)
            m_verts_gt = m_output_gt.vertices
            m_joints_gt = m_output_gt.joints


            s = time.time()
            m_params = {k: v[males].clone() for k, v in bparams.items()}
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            m_joints = m_output.joints
            e = time.time()
            # print('time for smplx male forward pass: ', e-s)

            if self.fit_ref:

                m_params_p = {k:v[males].clone() for k,v in params_p.items()}
                m_output_p = self.male_model(**m_params_p)
                m_verts_p = m_output_p.vertices
                m_joints_p = m_output_p.joints

                new_batch_p['m_verts_gt'] = m_verts_gt
                new_batch_p['m_joints_gt'] = m_joints_gt
                new_batch_p['m_verts'] = m_verts_p.clone().detach()
                new_batch_p['m_joints'] = m_joints_p.clone().detach()
                new_batch_p['m_params_gt'] = m_params_p

                new_batch['m_verts_gt'] = m_verts_gt
                new_batch['m_joints_gt'] = m_joints_gt
                new_batch['m_verts'] = m_verts.clone().detach()
                new_batch['m_joints'] = m_joints.clone().detach()
                new_batch['m_params_gt'] = m_params_gt

        losses = {}
        losses_w = {}

        if self.pose_loss_weight > 0:
            losses['pose'] = 0
            if FN>0:
                losses['pose'] += self.LossL2(f_params_gt['fullpose_rotmat'], f_params['fullpose_rotmat'])
            if MN>0:
                losses['pose'] += self.LossL2(m_params_gt['fullpose_rotmat'], m_params['fullpose_rotmat'])

            losses_w['pose'] = self.pose_loss_weight*losses['pose']


        rh2obj_gt = batch['rh2obj_gt'][:,cur_id:cur_id+T].norm(dim=-1).reshape(B*T, -1, 1)
        # rh2obj_w = torch.exp(-self.use_exp*rh2obj_gt)
        rh2obj_w = torch.ones_like(rh2obj_gt)

        int_fields = results['int_field'].reshape(B*T, -1)
        int_field = int_fields[:,:99].reshape(rh2obj_gt.shape)

        if self.use_exp != 0 and self.use_exp != -1:
            losses['rh2obj_inf'] = self.LossL1(torch.exp(-self.use_exp*rh2obj_gt), torch.exp(-self.use_exp*int_field))
            losses_w['rh2obj_inf'] = self.vertex_loss_weight*losses['rh2obj_inf']
        else:
            losses['rh2obj_inf'] = self.LossL1(rh2obj_w*rh2obj_gt, rh2obj_w*int_field)
            losses_w['rh2obj_inf'] = self.vertex_loss_weight*losses['rh2obj_inf']

        lh2obj_gt = batch['lh2obj_gt'][:,cur_id:cur_id+T].norm(dim=-1).reshape(B*T, -1, 1)
        # lh2obj_w = torch.exp(-self.use_exp*lh2obj_gt)
        lh2obj_w = torch.ones_like(lh2obj_gt)

        # int_field = results['int_field'][:,99:].reshape(lh2obj_gt.shape)
        int_field = int_fields[:,99:].reshape(lh2obj_gt.shape)


        if self.use_exp != 0 and self.use_exp != -1:
            losses['lh2obj_inf'] = self.LossL1(torch.exp(-self.use_exp*lh2obj_gt), torch.exp(-self.use_exp*int_field))
            losses_w['lh2obj_inf'] = self.vertex_loss_weight*losses['lh2obj_inf']
        else:
            losses['lh2obj_inf'] = self.LossL1(lh2obj_w*lh2obj_gt, lh2obj_w*int_field)
            losses_w['lh2obj_inf'] = self.vertex_loss_weight*losses['lh2obj_inf']
        
        
        rh2obj_gt = batch['rh2obj_gt'][:,cur_id:cur_id+T].norm(dim=-1).reshape(B*T, -1, 1)
        rh2obj_w = torch.exp(-self.use_exp*rh2obj_gt)

        lh2obj_gt = batch['lh2obj_gt'][:,cur_id:cur_id+T].norm(dim=-1).reshape(B*T,-1, 1)
        lh2obj_w = torch.exp(-self.use_exp*lh2obj_gt)


        # right hand vertex loss
        rh_ids = self.rhand_idx
        rh_ids1 = self.verts_ids[self.rh_ids_sampled]

        lh_ids = self.lhand_idx
        lh_ids1 = self.verts_ids[self.lh_ids_sampled]

        if self.rh_vertex_loss_weight > 0:
            losses['rh_vertices'] = 0
            if FN > 0:
                losses['rh_vertices'] += self.vertex_loss(f_verts_gt[:, rh_ids], f_verts[:, rh_ids])
                losses['rh_vertices'] += self.vertex_loss(rh2obj_w[females]*f_verts_gt[:, rh_ids1], rh2obj_w[females]*f_verts[:, rh_ids1])
            if MN > 0:
                losses['rh_vertices'] += self.vertex_loss(m_verts_gt[:, rh_ids], m_verts[:, rh_ids])
                losses['rh_vertices'] += self.vertex_loss(rh2obj_w[males]*m_verts_gt[:, rh_ids1], rh2obj_w[males]*m_verts[:, rh_ids1])

            losses_w['rh_vertices'] = self.rh_vertex_loss_weight*losses['rh_vertices']


            losses['lh_vertices'] = 0
            if FN > 0:
                losses['lh_vertices'] += self.vertex_loss(f_verts_gt[:, lh_ids], f_verts[:, lh_ids])
                losses['lh_vertices'] += self.vertex_loss(lh2obj_w[females]*f_verts_gt[:, lh_ids1], lh2obj_w[females]*f_verts[:, lh_ids1])
            if MN > 0:
                losses['lh_vertices'] += self.vertex_loss(m_verts_gt[:, lh_ids], m_verts[:, lh_ids])
                losses['lh_vertices'] += self.vertex_loss(lh2obj_w[males]*m_verts_gt[:, lh_ids1], lh2obj_w[males]*m_verts[:, lh_ids1])

            losses_w['lh_vertices'] = self.rh_vertex_loss_weight*losses['lh_vertices']
        
        

            if self.fit_ref:
                new_batch['lh2obj'] = torch.zeros([B*T, 778]).to(lh2obj_gt.device)
                new_batch['rh2obj'] = torch.zeros([B*T, 778]).to(rh2obj_gt.device)

                new_batch_p['lh2obj'] = torch.zeros([B*T, 778]).to(lh2obj_gt.device)
                new_batch_p['rh2obj'] = torch.zeros([B*T, 778]).to(rh2obj_gt.device)



            bps_type = 'dists'
            losses['rh2obj'] = 0
            losses['lh2obj'] = 0
            obj_verts_gt = batch['verts_obj'][:,cur_id:cur_id+T].reshape(B*T,-1, 3)
            if FN>0:
                # rh2obj_gt = self.bps_torch.encode(x = batch['verts_obj'][:,-1][females],
                #                                 feature_type=[bps_type],
                #                                 custom_basis=f_verts_gt[:,self.rhand_idx])[bps_type]

                s = time.time()
                rh2obj = self.bps_torch.encode(x = obj_verts_gt[females],
                                                feature_type=[bps_type],
                                                custom_basis=f_verts[:,self.rhand_idx])[bps_type]
                                                # custom_basis=f_verts[:,self.verts_ids[self.rh_ids_sampled]])[bps_type]
                
                lh2obj = self.bps_torch.encode(x = obj_verts_gt[females],
                                                feature_type=[bps_type],
                                                custom_basis=f_verts[:,self.lhand_idx])[bps_type]
                                                # custom_basis=f_verts[:,self.verts_ids[self.lh_ids_sampled]])[bps_type]
                e = time.time()
                t_f = e-s
                # print('time for female bps: ', t_f)
                
                losses['rh2obj'] += self.LossL1(torch.exp(-self.use_exp*rh2obj[:, self.rh_ids_sampled1]),torch.exp(-self.use_exp*rh2obj_gt[females][...,0]))
                # losses['rh2obj'] += self.LossL1(torch.exp(-self.use_exp*rh2obj),torch.exp(-self.use_exp*rh2obj_gt[females][...,0]))

                losses['lh2obj'] += self.LossL1(torch.exp(-self.use_exp*lh2obj[:, self.lh_ids_sampled1]),torch.exp(-self.use_exp*lh2obj_gt[females][...,0]))
                # losses['lh2obj'] += self.LossL1(torch.exp(-self.use_exp*lh2obj),torch.exp(-self.use_exp*lh2obj_gt[females][...,0]))

                if self.fit_ref:

                    new_batch['lh2obj'][females] = lh2obj.clone().detach()
                    new_batch['rh2obj'][females] = rh2obj.clone().detach()

                    rh2obj_p = self.bps_torch.encode(x = obj_verts_gt[females],
                                                    feature_type=[bps_type],
                                                    custom_basis=f_verts_p[:,self.rhand_idx])[bps_type]
                                                    # custom_basis=f_verts[:,self.verts_ids[self.rh_ids_sampled]])[bps_type]
                    
                    lh2obj_p = self.bps_torch.encode(x = obj_verts_gt[females],
                                                    feature_type=[bps_type],
                                                    custom_basis=f_verts_p[:,self.lhand_idx])[bps_type]
                    
                    new_batch_p['lh2obj'][females] = lh2obj_p.clone().detach()
                    new_batch_p['rh2obj'][females] = rh2obj_p.clone().detach()
            
            if MN>0:
                # rh2obj_gt = self.bps_torch.encode(x = batch['verts_obj'][:,-1][males],
                #                                 feature_type=[bps_type],
                #                                 custom_basis=m_verts_gt[:,self.rhand_idx])[bps_type]
                s = time.time()
                rh2obj = self.bps_torch.encode(x = obj_verts_gt[males],
                                feature_type=[bps_type],
                                custom_basis=m_verts[:,self.rhand_idx])[bps_type]
                                # custom_basis=m_verts[:,self.verts_ids[self.rh_ids_sampled]])[bps_type]
                
                lh2obj = self.bps_torch.encode(x = obj_verts_gt[males],
                                                feature_type=[bps_type],
                                                custom_basis=m_verts[:,self.lhand_idx])[bps_type]
                                                # custom_basis=m_verts[:,self.verts_ids[self.lh_ids_sampled]])[bps_type]
                e = time.time()
                t_m = e-s
                # print('time for male bps: ', t_m)
                
                losses['rh2obj'] += self.LossL1(torch.exp(-self.use_exp*rh2obj[:, self.rh_ids_sampled1]),torch.exp(-self.use_exp*rh2obj_gt[males][...,0]))
                # losses['rh2obj'] += self.LossL1(torch.exp(-self.use_exp*rh2obj),torch.exp(-self.use_exp*rh2obj_gt[males][...,0]))
                losses['lh2obj'] += self.LossL1(torch.exp(-self.use_exp*lh2obj[:, self.lh_ids_sampled1]),torch.exp(-self.use_exp*lh2obj_gt[males][...,0]))
                # losses['lh2obj'] += self.LossL1(torch.exp(-self.use_exp*lh2obj),torch.exp(-self.use_exp*lh2obj_gt[males][...,0]))

                if self.fit_ref:
                    new_batch['lh2obj'][males] = lh2obj.clone().detach()
                    new_batch['rh2obj'][males] = rh2obj.clone().detach()

                    rh2obj_p = self.bps_torch.encode(x = obj_verts_gt[males],
                                                    feature_type=[bps_type],
                                                    custom_basis=m_verts_p[:,self.rhand_idx])[bps_type]

                    lh2obj_p = self.bps_torch.encode(x = obj_verts_gt[males],
                                                    feature_type=[bps_type],
                                                    custom_basis=m_verts_p[:,self.lhand_idx])[bps_type]

                    new_batch_p['lh2obj'][males] = lh2obj_p.clone().detach()
                    new_batch_p['rh2obj'][males] = rh2obj_p.clone().detach()
                

            losses_w['rh2obj'] = self.rh_vertex_loss_weight*losses['rh2obj']
            losses_w['lh2obj'] = self.rh_vertex_loss_weight*losses['lh2obj']
        


        with torch.no_grad():
            loss_v2v = []

            if FN > 0:
                loss_v2v.append(v2v(f_verts_gt,
                                    f_verts,
                                     mean=False)
                                 )
            if MN > 0:
                loss_v2v.append(v2v(m_verts_gt,
                                    m_verts,
                                    mean=False)
                                 )

            loss_v2v = torch.cat(loss_v2v, dim=0).mean(dim=-1).sum()

        # loss_total = torch.stack(list(losses.values())).sum()
        loss_total = torch.stack(list(losses_w.values())).sum()
        losses['loss_total'] = loss_total
        losses['loss_v2v'] = losses['lh_vertices'] + losses['rh_vertices']
        # losses['loss_w'] = losses_w

        if eval_test:
            losses['loss_v2v'] = loss_v2v

            T = FN + MN
            losses['rh_vertices'] = 0
            if FN > 0:
                losses['rh_vertices'] += v2v(f_verts_gt[:, self.rhand_idx], f_verts[:, self.rhand_idx]).sum()
            if MN > 0:
                losses['rh_vertices'] += v2v(m_verts_gt[:, self.rhand_idx], m_verts[:, self.rhand_idx]).sum()

            losses['rh_vertices'] = losses['rh_vertices']/(T*778)

            losses['lh_vertices'] = 0
            if FN > 0:
                losses['lh_vertices'] += v2v(f_verts_gt[:, self.lhand_idx], f_verts[:, self.lhand_idx]).sum()
            if MN > 0:
                losses['lh_vertices'] += v2v(m_verts_gt[:, self.lhand_idx], m_verts[:, self.lhand_idx]).sum()
            
            losses['lh_vertices'] = losses['lh_vertices']/(T*778)

            losses['rh_vertices_p'] = 0
            if FN > 0:
                losses['rh_vertices_p'] += v2v(f_verts_gt[:, self.rhand_idx], f_verts_p[:, self.rhand_idx]).sum()
            if MN > 0:
                losses['rh_vertices_p'] += v2v(m_verts_gt[:, self.rhand_idx], m_verts_p[:, self.rhand_idx]).sum()
            
            losses['rh_vertices_p'] = losses['rh_vertices_p']/(T*778)

            losses['lh_vertices_p'] = 0
            if FN > 0:
                losses['lh_vertices_p'] += v2v(f_verts_gt[:, self.lhand_idx], f_verts_p[:, self.lhand_idx]).sum()
            if MN > 0:
                losses['lh_vertices_p'] += v2v(m_verts_gt[:, self.lhand_idx], m_verts_p[:, self.lhand_idx]).sum()
            
            losses['lh_vertices_p'] = losses['lh_vertices_p']/(T*778)

            losses['rh_joints'] = 0
            losses['rh_joints_p'] = 0
            if FN > 0:
                losses['rh_joints'] += v2v(f_joints_gt[:,40:55], f_joints[:,40:55]).sum()
                losses['rh_joints_p'] += v2v(f_joints_gt[:,40:55], f_joints_p[:,40:55]).sum()
            if MN > 0:
                losses['rh_joints'] += v2v(m_joints_gt[:,40:55], m_joints[:,40:55]).sum()
                losses['rh_joints_p'] += v2v(m_joints_gt[:,40:55], m_joints_p[:,40:55]).sum()
            
            losses['rh_joints'] = losses['rh_joints']/(T*15)
            losses['rh_joints_p'] = losses['rh_joints_p']/(T*15)


            losses['lh_joints'] = 0
            losses['lh_joints_p'] = 0
            if FN > 0:
                losses['lh_joints'] += v2v(f_joints_gt[:,25:40], f_joints[:,25:40]).sum()
                losses['lh_joints_p'] += v2v(f_joints_gt[:,25:40], f_joints_p[:,25:40]).sum()
            if MN > 0:
                losses['lh_joints'] += v2v(m_joints_gt[:,25:40], m_joints[:,25:40]).sum()
                losses['lh_joints_p'] += v2v(m_joints_gt[:,25:40], m_joints_p[:,25:40]).sum()

            losses['lh_joints'] = losses['lh_joints']/(T*15)
            losses['lh_joints_p'] = losses['lh_joints_p']/(T*15)


            ######## contact consistency metrics
            rh_contact_f = rh2obj_gt[females][...,0] < 0.002
            rh_contact_m = rh2obj_gt[males][...,0] < 0.002

            lh_contact_f = lh2obj_gt[females][...,0] < 0.002
            lh_contact_m = lh2obj_gt[males][...,0] < 0.002

            losses['rh_contact_const'] = 0
            if FN > 0:
                losses['rh_contact_const'] += v2v(f_verts_gt[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_f], f_verts[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_f]).sum()
            if MN > 0:
                losses['rh_contact_const'] += v2v(m_verts_gt[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_m], m_verts[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_m]).sum()

            losses['rh_contact_const'] = losses['rh_contact_const']/(rh_contact_m.sum() + rh_contact_f.sum()+1e-6)

            losses['lh_contact_const'] = 0
            if FN > 0:
                losses['lh_contact_const'] += v2v(f_verts_gt[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_f], f_verts[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_f]).sum()
            if MN > 0:
                losses['lh_contact_const'] += v2v(m_verts_gt[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_m], m_verts[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_m]).sum()

            losses['lh_contact_const'] = losses['lh_contact_const']/(lh_contact_m.sum() + lh_contact_f.sum()+1e-6)

            if losses['rh_contact_const'] != losses['rh_contact_const']:
                losses['rh_contact_const'] = 0
            if losses['lh_contact_const'] != losses['lh_contact_const']:
                losses['lh_contact_const'] = 0

        return loss_total, losses, new_batch, new_batch_p

    def get_loss_(self, batch, batch_idx, results):

        cur_id = 1

        bparams = results['body_params']

        new_batch = {k:v.clone().detach() for k,v in bparams.items()}
        new_batch['betas'] = batch['betas'][:,cur_id].clone()

        genders = batch['gender'][:,cur_id]
        males = genders == 1
        females = ~males

        B = batch['transl_obj'].shape[0]
        v_template = batch['sbj_vtemp'][:,cur_id]

        FN = sum(females)
        MN = sum(males)

        params_gt = parms_6D2full(batch['fullpose_rotmat'][:,cur_id],
                                  batch['transl'][:,cur_id],
                                  d62rot=False)

        losses = {}
        losses_w = {}

        rh2obj_gt = batch['rh2obj_gt'][:,cur_id]


        lh2obj_gt = batch['lh2obj_gt'][:,cur_id]


        rh2obj_gt = batch['rh2obj_gt'][:,cur_id]
        lh2obj_gt = batch['lh2obj_gt'][:,cur_id]

        new_batch['lh2obj'] = torch.zeros([B, 778]).to(lh2obj_gt.device)
        new_batch['rh2obj'] = torch.zeros([B, 778]).to(rh2obj_gt.device)

        bps_type = 'dists'

        if FN > 0:
            
            self.female_model.v_template = v_template[females].clone()

            f_params_gt = {k: v[females].clone().detach() for k, v in params_gt.items()}
            f_output_gt = self.female_model(**f_params_gt)
            f_verts_gt = f_output_gt.vertices

            f_params = {k: v[females].clone().detach() for k, v in bparams.items()}
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices

            new_batch['f_verts_gt'] = f_verts_gt
            new_batch['f_verts'] = f_verts.clone().detach()
            new_batch['f_params_gt'] = f_params_gt

            rh2obj = self.bps_torch.encode(x = batch['verts_obj'][:,cur_id][females],
                                            feature_type=[bps_type],
                                            custom_basis=f_verts[:,self.rhand_idx])[bps_type]
                                            # custom_basis=f_verts[:,self.verts_ids[self.rh_ids_sampled]])[bps_type]
            
            lh2obj = self.bps_torch.encode(x = batch['verts_obj'][:,cur_id][females],
                                            feature_type=[bps_type],
                                            custom_basis=f_verts[:,self.lhand_idx])[bps_type]
            
            new_batch['lh2obj'][females] = lh2obj.clone().detach()
            new_batch['rh2obj'][females] = rh2obj.clone().detach()

        if MN > 0:
            self.male_model.v_template = v_template[males].clone()

            m_params_gt = {k: v[males].clone() for k, v in params_gt.items()}
            m_output_gt = self.male_model(**m_params_gt)
            m_verts_gt = m_output_gt.vertices

            m_params = {k: v[males].clone() for k, v in bparams.items()}
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices

            new_batch['m_verts_gt'] = m_verts_gt
            new_batch['m_verts'] = m_verts.clone().detach()
            new_batch['m_params_gt'] = m_params_gt

            rh2obj = self.bps_torch.encode(x = batch['verts_obj'][:, cur_id][males],
                                            feature_type=[bps_type],
                                            custom_basis=m_verts[:,self.rhand_idx])[bps_type]

            lh2obj = self.bps_torch.encode(x = batch['verts_obj'][:, cur_id][males],
                                            feature_type=[bps_type],
                                            custom_basis=m_verts[:,self.lhand_idx])[bps_type]
            
            new_batch['lh2obj'][males] = lh2obj.clone().detach()
            new_batch['rh2obj'][males] = rh2obj.clone().detach()


        loss_total = 0
        losses['loss_total'] = 0
        losses['loss_v2v'] = 0

        return loss_total, losses, new_batch

    def get_loss_ref(self, batch, new_batch, batch_idx, results, eval_test=False):

        cur_id = 1

        B = batch['transl_obj'].shape[0]
        T = self.T
        v_template = batch['sbj_vtemp'][:,cur_id:cur_id+T].reshape(B*T,-1,3)

        bparams = results['body_params']

        # genders = batch['gender'][:,cur_id]
        genders = batch['gender'][:,cur_id:cur_id+T].reshape(-1)

        males = genders == 1
        females = ~males

        FN = sum(females)
        MN = sum(males)


        if FN > 0:
            
            self.female_model.v_template = v_template[females].clone()
            f_verts_gt = new_batch['f_verts_gt']
            f_joints_gt = new_batch['f_joints_gt']
            f_verts_hnet = new_batch['f_verts']
            f_joints_hnet = new_batch['f_joints']
            f_params_gt = new_batch['f_params_gt']

            f_params = {k: v[females].clone() for k, v in bparams.items()}
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices
            f_joints = f_output.joints

        if MN > 0:
            self.male_model.v_template = v_template[males].clone()

            m_verts_gt = new_batch['m_verts_gt']
            m_joints_gt = new_batch['m_joints_gt']
            m_verts_hnet = new_batch['m_verts']
            m_joints_hnet = new_batch['m_joints']
            m_params_gt = new_batch['m_params_gt']

            m_params = {k: v[males].clone() for k, v in bparams.items()}
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            m_joints = m_output.joints

        losses = {}
        losses_w = {}

        if self.pose_loss_weight > 0:
            losses['pose'] = 0
            if FN>0:
                losses['pose'] += self.LossL2(f_params_gt['fullpose_rotmat'], f_params['fullpose_rotmat'])
            if MN>0:
                losses['pose'] += self.LossL2(m_params_gt['fullpose_rotmat'], m_params['fullpose_rotmat'])

            losses_w['pose'] = self.pose_loss_weight*losses['pose']


        rh2obj = new_batch['rh2obj'].min(dim=-1)[0]
        # rh2obj_w = torch.exp(-self.use_exp*rh2obj)
        rh2obj_w = self.sig(2*self.use_exp*(-rh2obj + 0.1)).reshape(-1,1,1)


        lh2obj = new_batch['lh2obj'].min(dim=-1)[0]
        # lh2obj_w = torch.exp(-self.use_exp*lh2obj)
        lh2obj_w = self.sig(2*self.use_exp*(-lh2obj + 0.1)).reshape(-1,1,1)


        # right hand vertex loss
        if self.rh_vertex_loss_weight > 0:
            losses['rh_vertices'] = 0
            if FN > 0:
                rh2obj_f = rh2obj_w[females]*f_verts_gt[:, self.rhand_idx] + (1-rh2obj_w[females])*f_verts_hnet[:, self.rhand_idx]
                losses['rh_vertices'] += self.vertex_loss(rh2obj_f, f_verts[:, self.rhand_idx])
            if MN > 0:
                rh2obj_m = rh2obj_w[males]*m_verts_gt[:, self.rhand_idx] + (1-rh2obj_w[males])*m_verts_hnet[:, self.rhand_idx]
                losses['rh_vertices'] += self.vertex_loss(rh2obj_m, m_verts[:, self.rhand_idx])

            losses_w['rh_vertices'] = self.rh_vertex_loss_weight*losses['rh_vertices']


            losses['lh_vertices'] = 0
            if FN > 0:
                lh2obj_f = lh2obj_w[females]*f_verts_gt[:, self.lhand_idx] + (1-lh2obj_w[females])*f_verts_hnet[:, self.lhand_idx]
                losses['lh_vertices'] += self.vertex_loss(lh2obj_f, f_verts[:, self.lhand_idx])
            if MN > 0:
                lh2obj_m = lh2obj_w[males]*m_verts_gt[:, self.lhand_idx] + (1-lh2obj_w[males])*m_verts_gt[:, self.lhand_idx]
                losses['lh_vertices'] += self.vertex_loss(lh2obj_m, m_verts[:, self.lhand_idx])

            losses_w['lh_vertices'] = self.rh_vertex_loss_weight*losses['lh_vertices']


        with torch.no_grad():
            loss_v2v = []

            if FN > 0:
                loss_v2v.append(v2v(f_verts_gt,
                                     f_verts,
                                     mean=False)
                                 )
            if MN > 0:
                loss_v2v.append(v2v(m_verts_gt,
                                     m_verts,
                                     mean=False)
                                 )

            loss_v2v = torch.cat(loss_v2v, dim=0).mean(dim=-1).sum()

        # loss_total = torch.stack(list(losses.values())).sum()
        loss_total = torch.stack(list(losses_w.values())).sum()
        losses = {k+'_ref':v for k,v in losses.items()}

        losses['loss_total'] = loss_total
        losses['loss_v2v_ref'] = losses['lh_vertices_ref'] + losses['rh_vertices_ref']
        # losses['loss_w'] = losses_w

        if eval_test:
            losses['loss_v2v'] = loss_v2v
            T = FN+MN
            losses['rh_vertices'] = 0
            if FN > 0:
                losses['rh_vertices'] += v2v(f_verts_gt[:, self.rhand_idx], f_verts[:, self.rhand_idx]).sum()
            if MN > 0:
                losses['rh_vertices'] += v2v(m_verts_gt[:, self.rhand_idx], m_verts[:, self.rhand_idx]).sum()
            
            losses['rh_vertices'] = losses['rh_vertices'] / (T*778)


            losses['lh_vertices'] = 0
            if FN > 0:
                losses['lh_vertices'] += v2v(f_verts_gt[:, self.lhand_idx], f_verts[:, self.lhand_idx]).sum()
            if MN > 0:
                losses['lh_vertices'] += v2v(m_verts_gt[:, self.lhand_idx], m_verts[:, self.lhand_idx]).sum()

            losses['lh_vertices'] = losses['lh_vertices'] / (T*778)

            losses['rh_joints'] = 0
            if FN > 0:
                losses['rh_joints'] += v2v(f_joints_gt[:,40:55], f_joints[:,40:55]).sum()
            if MN > 0:
                losses['rh_joints'] += v2v(m_joints_gt[:,40:55], m_joints[:,40:55]).sum()
            
            losses['rh_joints'] = losses['rh_joints'] / (T*15)


            losses['lh_joints'] = 0
            if FN > 0:
                losses['lh_joints'] += v2v(f_joints_gt[:,25:40], f_joints[:,25:40]).sum()
            if MN > 0:
                losses['lh_joints'] += v2v(m_joints_gt[:,25:40], m_joints[:,25:40]).sum()
            
            losses['lh_joints'] = losses['lh_joints'] / (T*15)

            ###########
            rh2obj_gt = batch['rh2obj_gt'][:,cur_id:cur_id+self.T].norm(dim=-1).reshape(B*self.T, -1, 1)
            lh2obj_gt = batch['lh2obj_gt'][:,cur_id:cur_id+self.T].norm(dim=-1).reshape(B*self.T,-1, 1)

            rh_contact_f = rh2obj_gt[females][...,0] < 0.002
            rh_contact_m = rh2obj_gt[males][...,0] < 0.002

            lh_contact_f = lh2obj_gt[females][...,0] < 0.002
            lh_contact_m = lh2obj_gt[males][...,0] < 0.002

            losses['rh_contact_const'] = 0
            if FN > 0:
                losses['rh_contact_const'] += v2v(f_verts_gt[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_f], f_verts[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_f]).sum()
            if MN > 0:
                losses['rh_contact_const'] += v2v(m_verts_gt[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_m], m_verts[:, self.verts_ids[self.rh_ids_sampled]][rh_contact_m]).sum()

            losses['rh_contact_const'] = losses['rh_contact_const']/(rh_contact_m.sum() + rh_contact_f.sum()+1e-6)

            losses['lh_contact_const'] = 0
            if FN > 0:
                losses['lh_contact_const'] += v2v(f_verts_gt[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_f], f_verts[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_f]).sum()
            if MN > 0:
                losses['lh_contact_const'] += v2v(m_verts_gt[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_m], m_verts[:, self.verts_ids[self.lh_ids_sampled]][lh_contact_m]).sum()

            losses['lh_contact_const'] = losses['lh_contact_const']/(lh_contact_m.sum() + lh_contact_f.sum()+1e-6)

            if losses['rh_contact_const'] != losses['rh_contact_const']:
                losses['rh_contact_const'] = 0
            if losses['lh_contact_const'] != losses['lh_contact_const']:
                losses['lh_contact_const'] = 0


        return loss_total, losses


    def set_weight_annealing(self):

        self.vertex_loss_weight_ann = WeightAnneal(start_w=self.vertex_loss_weight/4, end_w=self.vertex_loss_weight, start_batch=0, end_batch=4)
        self.rh_vertex_loss_weight_ann = WeightAnneal(start_w=self.rh_vertex_loss_weight/5, end_w=self.rh_vertex_loss_weight, start_batch=0, end_batch=4)
        self.pose_loss_weight_ann   = WeightAnneal(start_w=self.pose_loss_weight, end_w=0, start_batch=0, end_batch=2)

        # self.contact_loss_weight_ann = WeightAnneal(start_w=self.contact_loss_weight/4, end_w=self.contact_loss_weight, start_batch=0, end_batch=4)

        # self.rh_vertex_loss_weight_ann = WeightAnneal(start_w=self.rh_vertex_loss_weight/10, end_w=self.rh_vertex_loss_weight, start_batch=4, end_batch=10)
        # self.feet_vertex_loss_weight_ann = WeightAnneal(start_w=self.feet_vertex_loss_weight/10, end_w=self.feet_vertex_loss_weight, start_batch=4, end_batch=10)

    def set_loss_weights(self):
        self.vertex_loss_weight = self.vertex_loss_weight_ann(self.epochs_completed)
        self.pose_loss_weight = self.pose_loss_weight_ann(self.epochs_completed)
        # self.contact_loss_weight = self.contact_loss_weight_ann(self.epochs_completed)

        self.rh_vertex_loss_weight = self.rh_vertex_loss_weight_ann(self.epochs_completed)
        # self.feet_vertex_loss_weight = self.feet_vertex_loss_weight_ann(self.epochs_completed)


    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf
        prev_lr_ref = np.inf
        self.fit_hnet = True
        self.fit_hnet_flag = False
        self.fit_ref = False

        self.set_weight_annealing()

        self.ref_epoch = np.inf # when to start training the refine network

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % (self.epochs_completed+1))

            if self.epochs_completed > self.ref_epoch:
                self.fit_ref = True

            self.set_loss_weights()

            train_loss_dict, train_loss_dict_ref = self.train()
            eval_loss_dict, eval_loss_dict_ref  = self.evaluate()


            if self.fit_hnet:
                self.lr_scheduler.step(eval_loss_dict['loss_v2v'])
                cur_lr = self.optimizer.param_groups[0]['lr']

                if cur_lr != prev_lr:
                    self.logger('--- learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                    prev_lr = cur_lr

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict, expr_ID=self.cfg.expr_ID,
                                                            epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                            model_name='hnet',
                                                            try_num=0, mode='evald')
                    if eval_loss_dict['loss_v2v'] < self.best_loss:

                        self.cfg.best_model = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model.pt' % (self.epochs_completed)), isfile=True)
                        self.save_network(self.network, self.cfg.best_model)
                        self.logger(eval_msg + ' ** ')
                        self.best_loss = eval_loss_dict['loss_v2v']

                    else:
                        self.logger(eval_msg)

                    for k in train_loss_dict.keys():
                        self.swriter.add_scalars('losses/%s/scalars' % k, {'train': train_loss_dict[k], 'val': eval_loss_dict[k]}, self.epochs_completed)


                if self.early_stopping(eval_loss_dict['loss_v2v']):
                    self.logger('Early stopping the hnet training!')
                    self.fit_hnet_flag = True

                    self._get_network().load_state_dict(torch.load(self.cfg.best_model, map_location=self.device), strict=False)
                    self.logger('Restored hnet trained model from %s' % self.cfg.best_model)

            if self.fit_ref:

                if not self.fit_hnet:
                    self.lr_scheduler_ref.step(eval_loss_dict_ref['loss_v2v_ref'])
                    cur_lr_ref = self.optimizer_ref.param_groups[0]['lr']

                    if cur_lr_ref != prev_lr_ref:
                        self.logger('--- ref learning rate changed from %.2e to %.2e ---' % (prev_lr_ref, cur_lr_ref))
                        prev_lr_ref = cur_lr_ref

                with torch.no_grad():
                    eval_msg_ref = Trainer.create_loss_message(eval_loss_dict_ref, expr_ID=self.cfg.expr_ID,
                                                            epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                            model_name='hnet_ref',
                                                            try_num=0, mode='evald')
                    if eval_loss_dict_ref['loss_v2v_ref'] < self.best_loss_ref:

                        self.cfg.best_model_ref = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model_ref.pt' % (self.epochs_completed)), isfile=True)
                        self.save_network(self.network_ref, self.cfg.best_model_ref)
                        self.logger(eval_msg_ref + ' ** ')
                        self.best_loss_ref = eval_loss_dict_ref['loss_v2v_ref']

                    else:
                        self.logger(eval_msg_ref)

                    for k in train_loss_dict_ref.keys():
                        self.swriter.add_scalars('losses/%s/scalars' % k, {'train': train_loss_dict_ref[k], 'val': eval_loss_dict_ref[k]}, self.epochs_completed)


                if not self.fit_hnet:
                    if self.early_stopping_ref(eval_loss_dict_ref['loss_v2v_ref']):
                        self.logger('Early stopping the hnet_ref training!')
                        self.fit_ref = False

            self.epochs_completed += 1
            if self.fit_hnet_flag:
                self.fit_hnet_flag = False
                self.fit_hnet = False
                self.fit_ref = True

            if not self.fit_hnet and not self.fit_ref:
                self.logger('Stopping the training!')
                break

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss))
        self.logger('Best hnet model path: %s\n' % self.cfg.best_model)
        self.logger('Best hnet_ref model path: %s\n' % self.cfg.best_model_ref)

    def configure_optimizers(self):

        self.optimizer = build_optimizer([self.network], self.cfg.optim)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.2, patience=8)
        self.early_stopping = EarlyStopping(**self.cfg.network.early_stopping, trace_func=self.logger)

        self.optimizer_ref = build_optimizer([self.network_ref], self.cfg.optim)
        self.lr_scheduler_ref = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_ref, 'min', factor=.2, patience=8)
        self.early_stopping_ref = EarlyStopping(**self.cfg.network.early_stopping, trace_func=self.logger)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)

    def inference_generate(self):

        # torch.set_grad_enabled(False)
        self.network.eval()
        self.network_ref.eval()
        device = self.device

        self.fit_hnet = True
        self.fit_ref = True

        ds_name = 'test'
        data = self.ds_test

        win_s = 500
        canvs = {
            'Input Body + Object': [win_s, win_s, ['1', '1']],
            'Generated Hands + Input Body': [win_s, win_s, ['1', '2']],
            'GT Body + Object': [win_s, win_s, [ '1', '3']],
            'Generated Right Hand': [win_s, win_s, ['2', '1']],
            'Generated Left Hand': [win_s, win_s, ['2','2']],
            'Input + Generated + GT': [win_s, win_s, ['2', '3']],
        }
        

        grid = [f'{3*win_s}px', f'{win_s}px {win_s}px {win_s}px', '{win_s}px {win_s}px' ]
        bg_color = np.array([186,186,186])/255.
        bg_color = np.array([238,238,239])/255.

        sp_anim = sp_animation(
            canvs=canvs,
            grid=grid,
            bg_color=bg_color,
        )

        canv_frames = 40
        past_seq_name = None
        past_sbj = None
        past_obj = None
        prev_batch = None


        curr_i = 1
        counter = 0
        T = self.T
        for batch_id, batch in enumerate(data):
            
            

            seq_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/s')[-1].replace('/', '_')
            seq_name_ = seq_name[:np.where([not i.isdigit() for i in seq_name])[0][-1]]
            curr_sbj = seq_name_.split('_')[0]
            curr_obj = seq_name_.split('_')[1]
            results_path = os.path.join(self.cfg.results_base_dir + '_grasp_transfer_02', seq_name_)
            results_path = os.path.join(self.cfg.results_base_dir + '_release_test', seq_name_)
            html_path = os.path.join(results_path, seq_name+'.html')
            
            if past_seq_name is None:
                past_seq_name = seq_name_

            if prev_batch is None or past_seq_name != seq_name_:
                # prev_batch = {k:v[:,curr_i-1].clone() for k , v in batch.items()}
                prev_batch = {'fullpose_rotmat':batch['fullpose_rotmat'].clone()}
                        
            if (past_seq_name != seq_name_) and ((past_sbj == curr_sbj) or (past_obj == curr_obj)):
                continue

            if past_seq_name != seq_name_:
                all_canvs = list(sp_anim.canvs.values())
                sp_anim.scene.link_canvas_events(*all_canvs[:3])
                sp_anim.scene.link_canvas_events(*all_canvs[3:])
                sp_anim.save_animation(html_path)
                sp_anim = sp_animation(canvs=canvs, grid=grid, bg_color=bg_color)
                print(f'finishing the {past_seq_name} sequence!')
                past_seq_name = seq_name_
                counter = 0


            past_sbj = curr_sbj
            past_obj = curr_obj

            batch = {k:v.to(self.device) for k,v in batch.items()}

            gender = batch['gender'][:,curr_i].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            
            ### object model

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/')[-1].split('_')[0]

            ##### FOR GRASP TRANSFER
            # obj_names = ['toothpaste', 'apple', 'mug', 'camera', 'binoculars']
            # # obj_names_replacement = ['cylindermedium', 'spheremedium', 'cylindermedium', 'cubemedium', 'cubesmall']
            # obj_names_replacement = ['flashlight', 'duck', 'stanfordbunny', 'elephant', 'cubemedium']


            # if obj_name in obj_names:
            #     obj_name = obj_names_replacement[obj_names.index(obj_name)]
            
            #######################

            obj_path = os.path.join(self.cfg.datasets.grab_path,'tools/object_meshes/contact_meshes', f'{obj_name}.ply')


            # obj_mesh = Mesh(filename=obj_path)
            obj_mesh_tri = trimesh.load(obj_path).simplify_quadratic_decimation(1000)
            obj_mesh = Mesh(v=obj_mesh_tri.vertices, f=obj_mesh_tri.faces)

            obj_verts = torch.from_numpy(obj_mesh.v)

            obj_m = ObjectModel(v_template=obj_verts).to(device)

            motion_obj = {
                'transl': batch['transl_obj'][:,curr_i:curr_i+T].reshape(-1,3),
                'global_orient': batch['global_orient_obj'][:,curr_i:curr_i+T].reshape(T,-1),
            }

            obj_verts = obj_m(**motion_obj).vertices.detach()


            makepath(html_path, isfile=True)

            grnd_mesh, cage, axis_l = get_ground()

            print(f'{seq_name} -- ')
            ##########################################
            ##########################################
            net_output = self.forward(batch)

            net_output = self.forward(batch)

            loss_total, losses_dict, new_batch, new_batch_p = self.get_loss(batch, 0, net_output)

            net_output_ref = self.forward_ref(new_batch)
        
            bparams = net_output['body_params']
            bparams_ref = net_output_ref['body_params']

            # prev_batch = {'fullpose_rotmat':bparams_ref['fullpose_rotmat'].reshape(1,T,-1,3,3).clone()}

            ##################################
            ##################################

            sbj_m.v_template = batch['sbj_vtemp'][:,curr_i:curr_i+T].clone().reshape(T,-1,3).to(sbj_m.v_template.device)
            output = sbj_m(**bparams)
            verts_init = output.vertices

            output_ref = sbj_m(**bparams_ref)
            verts_ref = output_ref.vertices                                                                            

            params_gt = parms_6D2full(batch['fullpose_rotmat'][:,curr_i:curr_i+T].clone().reshape(T,-1,3,3),
                                  batch['transl'][:,curr_i:curr_i+T].clone().reshape(T,3),
                                  d62rot=False)

            params_h = {k: v.clone() for k, v in params_gt.items() if not 'hand' in k }


            # sbj_m.v_template = batch['sbj_vtemp'][:,curr_i].clone().to(sbj_m.v_template.device)
            output_gt = sbj_m(**params_gt)
            verts_gt = output_gt.vertices

            output_h = sbj_m(**params_h)
            verts_h = output_h.vertices
            

            # for i in range(1,T):
            flat = np.array([199,214,245])/255.
            ref = np.array([245,210,210])/255.
            init = np.array([210,245,210])/255.
            obj_color = np.array([168, 229,0])/255.
            n_faces = None
            n_faces_hands = None
            

            for i in range(T-1):

                sbj_i_gt = simplify_mesh(v=to_cpu(verts_gt[i]), f=sbj_m.faces, vc=name_to_rgb['green'], n_faces=n_faces)
                sbj_i_h = simplify_mesh(v=to_cpu(verts_h[i]), f=sbj_m.faces, vc=flat, n_faces=n_faces)
                sbj_i_h_noh = simplify_mesh(v=to_cpu(verts_h[i]), f=sbj_m.faces, vc=flat, n_faces=n_faces, remove_verts=torch.cat([self.rhand_idx, self.lhand_idx]))
                sbj_i_h_rh = simplify_mesh(v=to_cpu(verts_h[i, self.rhand_idx]), f=self.rh_faces_ids, vc=flat, n_faces=n_faces_hands)
                sbj_i_h_lh = simplify_mesh(v=to_cpu(verts_h[i, self.lhand_idx]), f=self.lh_faces_ids, vc=flat, n_faces=n_faces_hands).flip_faces()

                sbj_i_init_rh = simplify_mesh(v=to_cpu(verts_init[i, self.rhand_idx]), f=self.rh_faces_ids, vc=init, n_faces=n_faces_hands)
                sbj_i_init_lh = simplify_mesh(v=to_cpu(verts_init[i, self.lhand_idx]), f=self.lh_faces_ids, vc=init, n_faces=n_faces_hands).flip_faces()
                sbj_i_init = simplify_mesh(v=to_cpu(verts_init[i]), f=sbj_m.faces, vc=init, n_faces=n_faces)

                sbj_i_ref = simplify_mesh(v=to_cpu(verts_ref[i]), f=sbj_m.faces, vc=ref, n_faces=n_faces)
                sbj_i_ref_rh = simplify_mesh(v=to_cpu(verts_ref[i, self.rhand_idx]), f=self.rh_faces_ids, vc=ref, n_faces=n_faces_hands)
                sbj_i_ref_lh = simplify_mesh(v=to_cpu(verts_ref[i, self.lhand_idx]), f=self.lh_faces_ids, vc=ref, n_faces=n_faces_hands).flip_faces()

                obj_i = simplify_mesh(v=to_cpu(obj_verts[i]), f = obj_mesh.f, vc=obj_color, n_faces=2000)


                meshes = {
                    'Input Body': sbj_i_h,
                    'Input Body no-hands': sbj_i_h_noh,
                    'Generated Right Hand': sbj_i_init_rh,
                    'Generated Left Hand': sbj_i_init_lh,
                    'Generated Body': sbj_i_init,
                    'Refined Body': sbj_i_ref,
                    'Refined Right Hand': sbj_i_ref_rh,
                    'Refined Left Hand': sbj_i_ref_lh,
                    'Input Right Hand': sbj_i_h_rh,
                    'Input Left Hand': sbj_i_h_lh,
                    'GT Body': sbj_i_gt,
                    'Object': obj_i,
                }

                canvs_meshes = {
                    'Input Body + Object': ['Input Body', 'Object'],
                    'Generated Hands + Input Body': ['Generated Right Hand', 'Generated Left Hand','Refined Right Hand', 'Refined Left Hand', 'Input Body no-hands', 'Object'],
                    'Generated Hands + Input Body': ['Generated Right Hand', 'Generated Left Hand','Refined Right Hand', 'Refined Left Hand', 'Input Right Hand', 'Input Left Hand', 'Object'],
                    'GT Body + Object': ['GT Body', 'Object'],
                    'Generated Right Hand': ['Generated Right Hand','Refined Right Hand', 'Object'],
                    'Generated Left Hand': ['Generated Left Hand', 'Refined Left Hand', 'Object'],
                    'Input + Generated + GT': ['Input Body', 'Generated Body', 'Refined Body', 'GT Body', 'Object'],
                }
                
                sp_anim.add_frame(meshes, canvs_meshes, focus='Object')
                counter += 1

                ############################
                
                if (counter)%canv_frames == 0 and canv_frames > 0:
                    all_canvs = list(sp_anim.canvs.values())
                    sp_anim.scene.link_canvas_events(*all_canvs[:3])
                    sp_anim.scene.link_canvas_events(*all_canvs[3:])
                    sp_anim.save_animation(html_path)
                    # sp_anim = sp_animation()
                    sp_anim = sp_animation(canvs=canvs,grid=grid, bg_color=bg_color)


        all_canvs = list(sp_anim.canvs.values())
        sp_anim.scene.link_canvas_events(*all_canvs[:3])
        sp_anim.scene.link_canvas_events(*all_canvs[3:])
        print(f'Saving the animation to: ', html_path)
        sp_anim.save_animation(html_path)         
            
    def inference_generate_params(self):

        # torch.set_grad_enabled(False)
        self.network.eval()
        self.network_ref.eval()
        device = self.device

        self.fit_hnet = True
        self.fit_ref = True


        ds_name = 'test'
        data = self.ds_test

        canv_frames = 40
        past_seq_name = None
        past_sbj = None
        past_obj = None
        prev_batch = None


        curr_i = 1
        counter = 0
        T = self.T
        save_data = {
            'bparams':[],
            'bparams_ref':[],
            'obj_params':[],
            'params_gt':[],
            'params_h':[],
            'params_p':[]
        }

        for batch_id, batch in enumerate(data):
            

            seq_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/s')[-1].replace('/', '_')
            seq_name_ = seq_name[:np.where([not i.isdigit() for i in seq_name])[0][-1]]
            curr_sbj = seq_name_.split('_')[0]
            curr_obj = seq_name_.split('_')[1]
            
            
            if past_seq_name is None:
                past_seq_name = seq_name_

            elif past_seq_name != seq_name_:
                makepath(results_path, isfile=True)
                torch.save(save_data,results_path)
                save_data = {
                        'bparams':[],
                        'bparams_ref':[],
                        'obj_params':[],
                        'params_gt':[],
                        'params_h':[],
                        'params_p':[]
                    }
                past_seq_name = seq_name_

            results_path = os.path.join(self.cfg.results_base_dir+'_saved_params_release', seq_name_+'.pt')
            params_path = os.path.join(results_path, seq_name+'.pt')

            if prev_batch is None or past_seq_name != seq_name_:
                # prev_batch = {k:v[:,curr_i-1].clone() for k , v in batch.items()}
                prev_batch = {'fullpose_rotmat':batch['fullpose_rotmat'].clone()}
            
            # batch['fullpose_rotmat'][:,curr_i-1] = prev_batch['fullpose_rotmat'][:,-1].clone()


            batch = {k:v.to(self.device) for k,v in batch.items()}
            # batch['verts'][:,0] = prev_batch['verts'].clone()
            # batch['fullpose_rotmat'][:,0] = prev_batch['fullpose_rotmat'].clone()

            gender = batch['gender'][:,curr_i].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            save_data['seq_name'] = seq_name_
            save_data['sbj_id'] = curr_sbj
            save_data['obj_id'] = curr_obj
            save_data['gender'] = gender
            # continue

            ### object model

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/')[-1].split('_')[0]
            obj_path = os.path.join(self.cfg.datasets.grab_path,'tools/object_meshes/contact_meshes', f'{obj_name}.ply')


            obj_mesh = Mesh(filename=obj_path)
            # obj_mesh_tri = trimesh.load(obj_path).simplify_quadratic_decimation(1000)
            # obj_mesh = Mesh(v=obj_mesh_tri.vertices, f=obj_mesh_tri.faces)

            obj_verts = torch.from_numpy(obj_mesh.v)
            save_data['obj_verts'] = obj_verts

            # obj_m = ObjectModel(v_template=obj_verts).to(device)

            motion_obj = {
                'transl': batch['transl_obj'][:,curr_i:curr_i+T].reshape(-1,3),
                'global_orient': batch['global_orient_obj'][:,curr_i:curr_i+T].reshape(T,-1),
            }

            save_data['obj_params'].append(motion_obj)

            # obj_verts = obj_m(**motion_obj).vertices.detach()

            # makepath(html_path, isfile=True)

            # grnd_mesh, cage, axis_l = get_ground()

            print(f'{seq_name} -- ')
            ##########################################
            ##########################################

            net_output = self.forward(batch)

            loss_total, losses_dict, new_batch, new_batch_p = self.get_loss(batch, 0, net_output)

            net_output_ref = self.forward_ref(new_batch)

            bparams = net_output['body_params']
            bparams_ref = net_output_ref['body_params']

            save_data['bparams'].append(bparams)
            save_data['bparams_ref'].append(bparams_ref)

            # prev_batch = {'fullpose_rotmat':bparams_ref['fullpose_rotmat'].reshape(1,T,-1,3,3).clone()}

            ##################################
            ##################################

            sbj_m.v_template = batch['sbj_vtemp'][:,curr_i:curr_i+T].clone().reshape(T,-1,3).to(sbj_m.v_template.device)
            save_data['sbj_vtemp'] = sbj_m.v_template.clone()
            # output = sbj_m(**bparams)
            # verts_init = output.vertices

            # output_ref = sbj_m(**bparams_ref)
            # verts_ref = output_ref.vertices                                                                            

            params_gt = parms_6D2full(batch['fullpose_rotmat'][:,curr_i:curr_i+T].clone().reshape(T,-1,3,3),
                                  batch['transl'][:,curr_i:curr_i+T].clone().reshape(T,3),
                                  d62rot=False)
            
            params_h = {k: v.clone() for k, v in params_gt.items() if not 'hand' in k }

            save_data['params_gt'].append(params_gt)
            save_data['params_h'].append(params_h)

            params_p = parms_6D2full(batch['fullpose_rotmat_p'][:,curr_i:curr_i+T].clone().reshape(T,-1,3,3),
                                  batch['transl_p'][:,curr_i:curr_i+T].clone().reshape(T,3),
                                  d62rot=False)
            save_data['params_p'].append(params_p)

        makepath(results_path, isfile=True)
        torch.save(save_data,results_path)
        
    def inference_generate_mesh(self):

        # torch.set_grad_enabled(False)
        self.network.eval()
        self.network_ref.eval()
        device = self.device

        ds_name = 'test'
        data = self.ds_test


        curr_i = 1
        counter = 0
        prev_obj = None

        for batch_id, batch in enumerate(data):
            
            counter += 1

            seq_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/s')[-1].replace('/', '_')
            # seq_name_ = seq_name[:np.where([not i.isdigit() for i in seq_name])[0][-1]]
            fid = int(seq_name.split('_')[-1])
            seq_name_ = '_'.join(seq_name.split('_')[:-1])
            curr_sbj = seq_name_.split('_')[0]
            curr_obj = seq_name_.split('_')[1]
            results_path = os.path.join(os.path.dirname(self.cfg.results_base_dir) + '/saved_meshes_release', seq_name_)

            batch = {k:v.to(self.device) for k,v in batch.items()}
            # batch['verts'][:,0] = prev_batch['verts'].clone()
            # batch['fullpose_rotmat'][:,0] = prev_batch['fullpose_rotmat'].clone()

            gender = batch['gender'][:,curr_i].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            
            ### object model

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'][:,curr_i].to(torch.long)].split('/')[-1].split('_')[0]
            obj_path = os.path.join(self.cfg.datasets.grab_path,'tools/object_meshes/contact_meshes', f'{obj_name}.ply')

            if not curr_obj == prev_obj:
                obj_mesh = Mesh(filename=obj_path)
                prev_obj = curr_obj

                # obj_mesh_tri = trimesh.load(obj_path).simplify_quadratic_decimation(1000)
                # obj_mesh = Mesh(v=obj_mesh_tri.vertices, f=obj_mesh_tri.faces)

                obj_verts = torch.from_numpy(obj_mesh.v)

                obj_m = ObjectModel(v_template=obj_verts).to(device)

            motion_obj = {
                'transl': batch['transl_obj'][:,curr_i],
                'global_orient': batch['global_orient_obj'][:,curr_i]
            }

            obj_verts = obj_m(**motion_obj).vertices.detach()


            makepath(results_path)

            grnd_mesh, cage, axis_l = get_ground()

            print(f'{seq_name} -- ')
            ##########################################
            ##########################################
            net_output = self.forward(batch)

            loss_total, losses_dict, new_batch, new_batch_p = self.get_loss(batch, 0, net_output)

            net_output_ref = self.forward_ref(new_batch)

            bparams = net_output['body_params']
            bparams_ref = net_output_ref['body_params']

            ##################################
            ##################################

            sbj_m.v_template = batch['sbj_vtemp'][:,curr_i].clone().to(sbj_m.v_template.device)
            output = sbj_m(**bparams)
            verts_init = output.vertices

            output_ref = sbj_m(**bparams_ref)
            verts_ref = output_ref.vertices                                                                            

            params_gt = parms_6D2full(batch['fullpose_rotmat'][:,curr_i].clone(),
                                  batch['transl'][:,curr_i].clone(),
                                  d62rot=False)

            params_h = {k: v.clone() for k, v in params_gt.items() if not 'hand' in k }


            sbj_m.v_template = batch['sbj_vtemp'][:,curr_i].clone().to(sbj_m.v_template.device)
            output_gt = sbj_m(**params_gt)
            verts_gt = output_gt.vertices

            output_h = sbj_m(**params_h)
            verts_h = output_h.vertices
            

            i=0

            sbj_i_gt = Mesh(v=to_cpu(verts_gt[i]), f=sbj_m.faces, vc=name_to_rgb['green'])
            sbj_i_gt.write_ply(os.path.join(results_path, f'{fid:04d}_body_gt.ply'))
            sbj_i_h = Mesh(v=to_cpu(verts_h[i]), f=sbj_m.faces, vc=name_to_rgb['orange'])
            sbj_i_h.write_ply(os.path.join(results_path, f'{fid:04d}_body_flat.ply'))
            sbj_i_h_rh = Mesh(v=to_cpu(verts_h[i, self.rhand_idx]), f=self.rh_faces_ids, vc=name_to_rgb['orange']) # right hand only
            sbj_i_h_rh.write_ply(os.path.join(results_path, f'{fid:04d}_rh_flat.ply')) # right hand only
            sbj_i_h_lh = Mesh(v=to_cpu(verts_h[i, self.lhand_idx]), f=self.lh_faces_ids, vc=name_to_rgb['orange']) # left hand only
            sbj_i_h_lh.write_ply(os.path.join(results_path, f'{fid:04d}_lh_flat.ply')) # left hand only
            sbj_i_h_noh = Mesh(v=to_cpu(verts_h[i]), f=sbj_m.faces, vc=name_to_rgb['orange'])
            sbj_i_h_noh.write_ply(os.path.join(results_path, f'{fid:04d}_body_flat_nohands.ply')) # no hands


            sbj_i_init_rh = Mesh(v=to_cpu(verts_init[i, self.rhand_idx]), f=self.rh_faces_ids, vc=name_to_rgb['pink'])
            sbj_i_init_rh.write_ply(os.path.join(results_path, f'{fid:04d}_rh_first.ply')) # right hand only
            sbj_i_init_lh = Mesh(v=to_cpu(verts_init[i, self.lhand_idx]), f=self.lh_faces_ids, vc=name_to_rgb['pink']).flip_faces()
            sbj_i_init_lh.write_ply(os.path.join(results_path, f'{fid:04d}_lh_first.ply')) # left hand only
            sbj_i_init = Mesh(v=to_cpu(verts_init[i]), f=sbj_m.faces, vc=name_to_rgb['pink'])
            sbj_i_init.write_ply(os.path.join(results_path, f'{fid:04d}_body_first.ply')) # no hands

            sbj_i_ref = Mesh(v=to_cpu(verts_ref[i]), f=sbj_m.faces, vc=name_to_rgb['blue'])
            sbj_i_ref.write_ply(os.path.join(results_path, f'{fid:04d}_body_ref.ply')) # no hands
            sbj_i_ref_rh = Mesh(v=to_cpu(verts_ref[i, self.rhand_idx]), f=self.rh_faces_ids, vc=name_to_rgb['blue'])
            sbj_i_ref_rh.write_ply(os.path.join(results_path, f'{fid:04d}_rh_ref.ply')) # right hand only
            sbj_i_ref_lh = Mesh(v=to_cpu(verts_ref[i, self.lhand_idx]), f=self.lh_faces_ids, vc=name_to_rgb['blue']).flip_faces()
            sbj_i_ref_lh.write_ply(os.path.join(results_path, f'{fid:04d}_lh_ref.ply')) # left hand only


            obj_i = Mesh(v=to_cpu(obj_verts[0]), f = obj_mesh.f, vc=name_to_rgb['yellow'])
            obj_i.write_ply(os.path.join(results_path, f'{fid:04d}_obj.ply')) # no hands


def simplify_mesh(mesh=None, v=None, f=None, n_faces=None, vc=name_to_rgb['pink'], remove_verts = None):

    if mesh is None:
        mesh_tri = trimesh.Trimesh(vertices=v, faces=f, process=False)
    else:
        mesh_tri = trimesh.Trimesh(vertices=mesh.v, faces=mesh.f, process=False)

    if remove_verts is not None:
        verts_mask = np.ones(mesh_tri.vertices.shape[0])
        verts_mask[to_np(remove_verts)] = 0
        mesh_tri.update_vertices(verts_mask.astype(np.bool_))
    if n_faces is not None:
        mesh_tri = mesh_tri.simplify_quadratic_decimation(n_faces)
    # mesh_tri = mesh_tri.simplify_quadratic_decimation(n_faces)
    return Mesh(v=mesh_tri.vertices, f=mesh_tri.faces, vc=vc)


def inference():

    parser = argparse.ArgumentParser(description='CNet-Inference')

    parser.add_argument('--work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--grab-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required=True,
                        type=str,
                        help='The path to the folder containing SMPL-X model downloaded from the website')

    parser.add_argument('--dataset-dir',
                        required=True,
                        type=str,
                        help='The path to the directory where the dataset is processed and stored')
    
    parser.add_argument('--out-type',
                        default='html',
                        type=str,
                        help='The type of output to generate. Options: mesh, params, html')

    parser.add_argument('--expr-id',
                        type=str,
                        help='Training ID')

    
    cmd_args = parser.parse_args()



    if cmd_args.expr_id is None:
        cfg_path = f'{cdir}/../configs/cnet.yaml'
        cfg = OmegaConf.load(cfg_path)
        cfg.best_model = f'{cdir}/../snapshots/cnet.pt'
        cfg.best_model_ref = f'{cdir}/../snapshots/rnet.pt'
        expr_ID = 'cnet_orig'
        cfg.expr_ID = expr_ID
    else:
        expr_ID = cmd_args.expr_id
        work_dir = cmd_args.work_dir
        cfg_path = os.path.join(work_dir,f'{expr_ID}/{expr_ID}.yaml')
        cfg = OmegaConf.load(cfg_path)

    cfg.datasets.grab_path = cmd_args.grab_path
    cfg.body_model.model_path = cmd_args.smplx_path


    cfg.output_folder = cmd_args.work_dir
    cfg.work_dir = os.path.join(cfg.output_folder, expr_ID)
    cfg.results_base_dir = os.path.join(cfg.work_dir, f'{cfg.expr_ID}_results')


    cfg.batch_size = 1
    cfg.num_gpus = 1
    cfg.cuda_id = 0
    cfg.datasets.dataset_dir = cmd_args.dataset_dir

    tester = Trainer(cfg=cfg, inference=True)

    if cmd_args.out_type == 'mesh':
        tester.inference_generate_mesh()
    elif cmd_args.out_type == 'params':
        tester.inference_generate_params()
    elif cmd_args.out_type == 'html':
        tester.inference_generate()

if __name__ == '__main__':

    inference()
