
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

import numpy as np
import torch
import os, glob
import smplx
import argparse
from tqdm import tqdm

    
import numpy as np
import trimesh
import pyrender
from pyrender.light import DirectionalLight
from pyrender.node import Node
from PIL import Image
from .utils import euler


import imageio

class Visualizer:
    def __init__(self, save_dir) -> None:
        self.save_dir = save_dir
        self.viewer = MeshViewer(offscreen=False)

        # Set camera pose.
        self.__set_camera_pose()

    def __set_camera_pose(self) -> None:
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = euler([80, -15, 0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -1.5, 1.3])
        self.viewer.update_camera_pose(camera_pose)

    def __save_gif(self, frames: list, duration=0.1) -> None:
        save_dir = f"{self.save_dir}/output.gif"
        imageio.mimsave(save_dir, frames, duration=duration)

    def close_viewer(self):
        self.viewer.close_viewer()

    def visualize(self, mesh_params: list, t: int, meshes=[]):
        assert len(meshes) == 0, "\'meshes\' should be an empty list."
        for params in mesh_params:
            mesh = Mesh(v=params['verts'][t], f=params['faces'], vc=params['color'])
            if "contact_map" in params:
                mesh.set_vert_color(params['contact_map'][t] > 0, 'red')
            meshes.append(mesh)
        self.viewer.set_static_meshes(meshes)

    def visualize_seq(self, meshes: list, skip=0, frames=[]):
        assert len(frames) == 0, "\'frames\' should be an empty list."
        T = meshes[0]['motion'].shape[0]
        for t in range(0, T, skip):
            self.visualize(meshes, t)
            color, _ = self.viewer.viewer.render(self.viewer.scene)
            frames.append(color)
        self.__save_gif(frames)





class MeshViewer(object):

    def __init__(self,
                 width=1200,
                 height=800,
                 bg_color = [0.0, 0.0, 0.0, 1.0],
                 offscreen = False,
                 registered_keys=None):
        super(MeshViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        self.bg_color = bg_color
        self.offscreen = offscreen
        self.scene = pyrender.Scene(bg_color=bg_color,
                                    ambient_light=(0.3, 0.3, 0.3),
                                    name = 'scene')

        self.aspect_ratio = float(width) / height
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.aspect_ratio)
        camera_pose = np.eye(4)
        camera_pose[:3,:3] = euler([80,-15,0], 'xzx')
        camera_pose[:3, 3] = np.array([-.5, -2., 1.5])

        self.cam = pyrender.Node(name = 'camera', camera=pc, matrix=camera_pose)

        self.scene.add_node(self.cam)

        if self.offscreen:
            light = Node(light=DirectionalLight(color=np.ones(3), intensity=3.0),
                          matrix=camera_pose)
            self.scene.add_node(light)
            self.viewer = pyrender.OffscreenRenderer(width, height)
        else:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          viewport_size=(width, height),
                                          cull_faces=False,
                                          run_in_thread=True,
                                          registered_keys=registered_keys)

        for i, node in enumerate(self.scene.get_nodes()):
            if node.name is None:
                node.name = 'Req%d'%i


    def close_viewer(self):
        if self.viewer.is_active:
            self.viewer.close_external()

    def set_background_color(self, bg_color=[1., 1., 1.]):
        self.scene.bg_color = bg_color

    def update_camera_pose(self, pose):
        if self.offscreen:
            self.scene.set_pose(self.cam, pose=pose)
        else:
            self.viewer._default_camera_pose[:] = pose

    def set_meshes(self, meshes =[], set_type = 'static'):

        if not self.offscreen:
            self.viewer.render_lock.acquire()

        for node in self.scene.get_nodes():
            if node.name is None:
                continue
            if 'static' in set_type and 'mesh' in node.name:
                self.scene.remove_node(node)
            elif 'dynamic' in node.name:
                self.scene.remove_node(node)

        for i, mesh in enumerate(meshes):
            mesh = self.to_pymesh(mesh)
            self.scene.add(mesh, name='%s_mesh_%d'%(set_type,i))

        if not self.offscreen:
            self.viewer.render_lock.release()

    def set_static_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='static')

    def set_dynamic_meshes(self, meshes =[]):
        self.set_meshes(meshes=meshes, set_type='dynamic')

    def save_snapshot(self, save_path):
        if not self.offscreen:
            print('We do not support rendering in Interactive mode!')
            return
        color, depth = self.viewer.render(self.scene)
        img = Image.fromarray(color)
        img.save(save_path)
