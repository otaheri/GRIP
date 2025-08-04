
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

import sys

import os
import glob
import numpy as np
import torch
from torch.utils import data
from tools.utils import np2torch, torch2np
from tools.utils import to_cpu, to_np, to_tensor
from tools.utils import loc2vel, loc2vel_rot

from torch.utils.data.dataloader import default_collate
from omegaconf import DictConfig

from smplx import SMPLXLayer

from psbody.mesh import Mesh, MeshViewers
import time
from tools.objectmodel import ObjectModel
from models.model_utils import parms_6D2full
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}

class LoadData(data.Dataset):
    def __init__(self,
                 cfg,
                 split_name='train'):

        super().__init__()

        self.split_name = split_name
        self.ds_dir = cfg.dataset_dir
        self.cfg = cfg
        dataset_dir = cfg.dataset_dir

        self.split_name = split_name
        self.ds_dir = dataset_dir

        self.ds = {}
        # dataset_dir = cfg.out_path
        self.ds_path = os.path.join(dataset_dir,split_name)
        datasets = glob.glob(self.ds_path + '/*.npy')

        self.load_ds(datasets)
        # self.normalize()
        self.frame_names = np.load(os.path.join(dataset_dir,split_name, 'frame_names.npz'))['frame_names']
        self.frame_sbjs = np.asarray([name.split('/')[-2] for name in self.frame_names])
        self.frame_st_end = np.asarray([int(name.split('_')[-1]) for name in self.frame_names])
        self.frame_objs = np.asarray([os.path.basename(name).split('_')[0] for name in self.frame_names])
        # self.frame_objs = np.asarray([name.split('Res/')[-1].split('/')[1] for name in self.frame_names]) ## for InterCap
        # self.frame_objs = np.asarray([re.search(r'\d+_(\D+?)_\d+', name.split('MoGaze/p')[-1]).group(1) for name in self.frame_names]) ## for MoGaze


        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        self.sbjs = np.unique(self.frame_sbjs)

        #######################################

        self.bps = torch.load(os.path.join(dataset_dir, 'bps.pt'))

        # v_templates
        base_path = os.path.join(self.cfg.grab_path,'tools/subject_meshes/male')
        file_list = []
        for sbj in self.sbjs:
            vt_path = os.path.join(base_path,sbj+'.ply')
            if os.path.exists(vt_path):
                file_list.append(vt_path)
            else:
                file_list.append(vt_path.replace('male','female'))
        self.sbj_vtemp = torch.from_numpy(np.asarray([Mesh(filename=file).v.astype(np.float32) for file in file_list]))
        self.sbj_betas = torch.from_numpy(np.asarray([np.load(file=f.replace('.ply','_betas.npy')).astype(np.float32) for f in file_list]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        self.ds['frame_sbj_ids'] = self.frame_sbjs

        self.genders = [self.sbj_info[sbj]['gender'] for sbj in self.sbjs]
        self.frame_genders = [self.genders[sbj] for sbj in self.frame_sbjs]
        self.ds['gender'] = torch.Tensor([1 if self.genders[sbj_id] == 'male' else 2 for sbj_id in self.ds['frame_sbj_ids']]).to(torch.long)



        self.objs = list(self.obj_info.keys())
        self.obj_verts = torch.from_numpy(np.asarray([self.obj_info[obj]['verts_sample'].astype(np.float32) for obj in self.objs]))
        for idx, name in enumerate(self.objs):
            self.frame_objs[(self.frame_objs == name)] = idx

        self.frame_objs = torch.from_numpy(self.frame_objs.astype(np.int8)).to(torch.long)

        # self.ds.pop('full_seq_id')
        self.L = self.ds['fullpose'].shape[0]

    def load_ds(self, dataset_names):
        self.ds = {}
        for name in dataset_names:
            self.ds.update(np.load(name, allow_pickle=True))
        self.ds = np2torch(self.ds)

    def normalize(self):

        norm_data_dir = os.path.join(self.ds_dir,'norm_data.pt')
        if os.path.exists(norm_data_dir):
            self.norm_data = torch.load(norm_data_dir)
        elif self.split_name =='train':
            in_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['in'].items() if v.dtype==torch.float}
            out_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['out'].items()}
            self.norm_data = {'in':in_p, 'out':out_p}
            torch.save(self.norm_data,norm_data_dir)
        else:
            raise('Please run the train split first to normalize the data')

        in_p = self.norm_data['in']
        out_p = self.norm_data['out']

        for k, v in in_p.items():
            self.ds['in'][k] = (self.ds['in'][k]-v[0])/v[1]

        # for k, v in out_p.items():
        #     self.ds['out'][k] = (self.ds['out'][k]-v[0])/v[1]

    def load_idx(self, idx, source=None):

        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v, dict):
                out[k] = self.load_idx(idx, v)
            else:
                out[k] = v[idx]
        out['betas'] = self.sbj_betas[self.frame_sbjs[idx]]
        out['sbj_vtemp'] =  self.sbj_vtemp[self.frame_sbjs[idx]]

        # compute object vertices on the fly
        motion_obj = {
            'transl': self.ds['transl_obj'][idx],
            'global_orient': self.ds['global_orient_obj'][idx]
        }
        
        bs = len(idx)
        idx_obj = self.obj_verts[self.frame_objs[idx][0]]
        obj_m = ObjectModel(v_template=idx_obj,
                            batch_size=bs)
        obj_out = obj_m(**motion_obj, pose2rot=True, intercap=False)
        # obj_out = obj_m(**motion_obj, pose2rot=True, intercap=True)
        out['verts_obj'] = obj_out.vertices.detach()

        # out['delta_trans_lh2obj'] = loc2vel(out['trans_lh2obj'],1)
        # out['delta_trans_rh2obj'] = loc2vel(out['trans_rh2obj'],1)

        # out['delta_rot_rh2obj'] = loc2vel_rot(out['rot_rh2obj'],1)
        # out['delta_rot_lh2obj'] = loc2vel_rot(out['rot_lh2obj'],1)


        return out

    def __len__(self):
        # return int(self.L/10)
        return self.L
        # return len(self.frame_names)

    def __getitem__(self, idx):

        end_id = self.ds['frame_ends'][idx].item()
        frame_id = self.ds['frame_ids'][idx].item()
        max_past = min(frame_id, 1)
        max_future = min(end_id - frame_id, 10)

        frames = torch.arange(-max_past,max_future+1).to(torch.long)
        # duplicate first and last frames to have past and furture frames for them as well
        frames = torch.cat([torch.zeros(1 - max_past), frames, max_future*torch.ones(10 - max_future)]).to(torch.long)

        p_ids = idx + frames
        seq_len = len(p_ids)

        data_past = self.load_idx(p_ids)
        # for n_idx in p_ids:

        #     data_out = self.load_idx(n_idx)
        #     data_out['idx'] = torch.from_numpy(np.array(n_idx, dtype=np.int32))

        #     data_past.append(data_out)
        data_past['idx'] = torch.from_numpy(np.array(p_ids, dtype=np.int32))
        # data_past['frame_ids'] = frames.to(torch.int32)
        # data_past['num_frames'] = torch.from_numpy(np.array([seq_len], dtype=np.int32))
        return data_past
        # return default_collate(data_past)


def build_dataloader(dataset: torch.utils.data.Dataset,
                     cfg: DictConfig,
                     split: str = 'train',
                     batch_size: int = 1,
                     ) -> torch.utils.data.DataLoader:

    dataset_cfg = cfg
    is_train    = 'train' in split
    is_test    = 'test' in split

    num_workers = dataset_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    shuffle     = dataset_cfg.get('shuffle', True)

    collate_fn  = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   batch_size,
        num_workers =   num_workers.get(split, 0),
        collate_fn  =   collate_fn,
        drop_last   =   True and (is_train or not is_test),
        pin_memory  =   dataset_cfg.get('pin_memory', False),
        shuffle     =   shuffle and is_train and not is_test,
    )
    return data_loader
