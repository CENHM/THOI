
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
from pathlib import Path
import shutil
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.meshviewer import Mesh, MeshViewer, points2sphere, colors
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import to_cpu
from tools.utils import euler
from tools.cfg_parser import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_sequences(cfg, subject, file):
    print(f'{subject} - {file}')
    grab_path = cfg.grab_dir

    mv = MeshViewer(offscreen=True)
    # set the camera pose
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
    camera_pose[:3, 3] = np.array([-.5, -1.5, 1.3])
    mv.update_camera_pose(camera_pose)

    vis_sequence(cfg, f'{grab_path}/{subject}/{file}', mv, '')
    # mv.close_viewer()


def vis_sequence(cfg, sequence, mv, k):
    seq_data = parse_npz(sequence)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']
    
    print(seq_data['motion_intent'])

    T = seq_data.n_frames

    rh_mesh = os.path.join(cfg.grab_dir, '..', seq_data.rhand.vtemp)
    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)
    rh_m = smplx.create(model_path=cfg.model_dir+"/MANO_RIGHT.pkl",
                        model_type='mano',
                        is_rhand=True,
                        gender=gender,
                        num_pca_comps=n_comps,
                        batch_size=T,
                        v_template=rh_vtemp,
                        flat_hand_mean=True)
    rh_parms = params2torch(seq_data.rhand.params)
    verts_rh = to_cpu(rh_m(**rh_parms).vertices)
    

    lh_mesh = os.path.join(cfg.grab_dir, '..', seq_data.lhand.vtemp)
    lh_vtemp = np.array(Mesh(filename=lh_mesh).vertices)
    lh_m = smplx.create(model_path=cfg.model_dir+"/MANO_LEFT.pkl",
                        model_type='mano',
                        is_rhand=False,
                        gender=gender,
                        num_pca_comps=n_comps,
                        batch_size=T,
                        v_template=lh_vtemp,
                        flat_hand_mean=True)
    lh_parms = params2torch(seq_data.lhand.params)
    verts_lh = to_cpu(lh_m(**lh_parms).vertices)


    if cfg.use_preprocessed_obj_mesh:
        paths = seq_data.object.object_mesh.split('/')
        obj_mesh = os.path.join(cfg.grab_dir, '..', paths[0], paths[1], paths[2] + "_preprocess", paths[3])
    else:
        obj_mesh = os.path.join(cfg.grab_dir, '..', seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_vtemp = np.array(obj_mesh.vertices)
    obj_m = ObjectModel(v_template=obj_vtemp,
                        batch_size=T)
    obj_parms = params2torch(seq_data.object.params)
    verts_obj = to_cpu(obj_m(**obj_parms).vertices)


    table_mesh = os.path.join(cfg.grab_dir, '..', seq_data.table.table_mesh)
    table_mesh = Mesh(filename=table_mesh)
    table_vtemp = np.array(table_mesh.vertices)
    table_m = ObjectModel(v_template=table_vtemp,
                        batch_size=T)
    table_parms = params2torch(seq_data.table.params)
    verts_table = to_cpu(table_m(**table_parms).vertices)

    skip_frame = 4

    p = f'save/{k}'
    if not os.path.exists(p):
        os.makedirs(p)

    clear_dir(p)

    if cfg.use_preprocessed_obj_mesh:
        paths = seq_data.object.object_mesh.split('/')
        obj_mesh = os.path.join(cfg.grab_dir, '..', paths[0], paths[1], paths[2] + "_preprocess", paths[3].split('.')[0] + '_idx.txt')
        contact_map_idx = np.array(read_txt_file(obj_mesh).split(',')[:-1], dtype=np.int)
        contact_map = seq_data['contact']['object']
        contact_map[contact_map == 21] = 26
        contact_map[contact_map == 22] = 26
        contact_map[contact_map < 26] = 0
        contact_map[contact_map >= 26] = 1
        contact_map = np.sum(contact_map, axis=0) / seq_data.n_frames
        contact_map[contact_map > 0] = 1
        contact_map = contact_map[contact_map_idx]
    else:
        contact_map = seq_data['contact']['object']

    for frame in tqdm(range(0, T, skip_frame)):
        o_mesh = Mesh(vertices=verts_obj[frame], faces=obj_mesh.faces, vc=colors['yellow'])
        o_mesh.set_vertex_colors(vc=colors['red'], vertex_ids=contact_map[frame] > 0)

        rh_mesh = Mesh(vertices=verts_rh[frame], faces=rh_m.faces, vc=colors['pink'], smooth=True)
        rh_mesh.set_vertex_colors(vc=colors['pink'])

        lh_mesh = Mesh(vertices=verts_lh[frame], faces=lh_m.faces, vc=colors['pink'], smooth=True)
        lh_mesh.set_vertex_colors(vc=colors['pink'])

        t_mesh = Mesh(vertices=verts_table[frame], faces=table_mesh.faces, vc=colors['white'])

        mv.set_static_meshes([o_mesh, rh_mesh, lh_mesh, t_mesh])
        
        mv.save_snapshot(p+f"/{frame}.png")


def read_txt_file(dir):
    with open(dir, "r") as f: 
        content = f.read()
    return content


def clear_dir(dir):
    for elm in Path(dir).glob('*'):
        elm.unlink() if elm.is_file() else shutil.rmtree(elm)


if __name__ == '__main__':
    # grab_dir = '$YOUR_GRAB_DIR/grab'
    # model_dir = '$YOUR_MANO_MODELS_DIR'

    use_preprocessed_obj_mesh = False

    cfg = {
        'grab_dir': grab_dir,
        'model_dir': model_dir,
        'use_preprocessed_obj_mesh': use_preprocessed_obj_mesh
    }

    cfg = Config(**cfg)
    visualize_sequences(cfg, 's10', 'cubesmall_pass_1.npz')
