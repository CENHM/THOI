# This part of code is borrowed from Junuk Cha 
# (https://github.com/JunukCha/Text2HOI/blob/main/preprocessing_grab_object.py)
# for the preprocess procedure of the GRAB dataset.
#
# Adjustment also have been made. We replace "simplification_quadric_edge_collap
# se_decimation" with "meshing_decimation_quadric_edge_collapse" based on 
# PyMeshLab official guidence (https://github.com/cnr-isti-vclab/PyMeshLab/blob/
# ef9d041fa227cc332e5a7cf01dd3fc6021cac86f/pymeshlab/keys.txt#L135)


import os
import os.path as osp

import numpy as np
import pymeshlab
import glob
from tqdm import tqdm


# def main(PATH_TO_GRAB):
#     original_folder = f"{PATH_TO_GRAB}/tools/object_meshes/contact_meshes"
#     save_folder = f"{PATH_TO_GRAB}/tools/object_meshes/contact_meshes_preprocess"
#     os.makedirs(save_folder, exist_ok=True)

#     n_target_vertices = 4000

#     original_meshes = glob.glob(os.path.join(original_folder, "*.ply"))
#     for mesh_path in tqdm.tqdm(original_meshes, desc="Processing object meshes"):
#         basename = os.path.basename(mesh_path)
#         ms = ml.MeshSet(verbose=0)
#         ms.load_new_mesh(mesh_path)
#         m = ms.current_mesh()
#         mo = ms.current_mesh()
#         print('mesh has', mo.vertex_number(), 'vertex and', mo.face_number(), 'faces')
#         TARGET = n_target_vertices
#         numFaces = 100 + 2 * TARGET
#         while (ms.current_mesh().vertex_number() > TARGET):
#             ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
#             numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
#         m = ms.current_mesh()
#         print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')
        
        # ms.save_current_mesh(osp.join(save_folder, basename))
        # pass


def main(PATH_TO_GRAB):
    original_folder = f"{PATH_TO_GRAB}/tools/object_meshes/contact_meshes"
    save_folder = f"{PATH_TO_GRAB}/tools/object_meshes/contact_meshes_preprocess"
    os.makedirs(save_folder, exist_ok=True)

    n_target_vertices = 4000

    original_meshes = glob.glob(os.path.join(original_folder, "*.ply"))
    for mesh_path in tqdm(original_meshes, desc="Processing object meshes"):
        basename = os.path.basename(mesh_path)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_path)

        # 获取原始顶点位置
        original_mesh = ms.current_mesh()
        original_vertex_positions = original_mesh.vertex_matrix()
        num_vertices = original_mesh.vertex_number()

        print(f'Original mesh has {num_vertices} vertices and {original_mesh.face_number()} faces')
        
        # 执行简化
        numFaces = 100 + 2 * n_target_vertices
        while (ms.current_mesh().vertex_number() > n_target_vertices):
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - n_target_vertices)

        # 获取简化后的顶点位置 (Mx3 矩阵)
        simplified_mesh = ms.current_mesh()
        simplified_vertex_positions = simplified_mesh.vertex_matrix()

        # 使用广播计算所有点之间的欧几里得距离
        original_vertex_positions = original_vertex_positions[:, np.newaxis, :]  # 调整形状为 (N, 1, 3)
        simplified_vertex_positions = simplified_vertex_positions[np.newaxis, :, :]  # 调整形状为 (1, M, 3)
        distances = np.linalg.norm(original_vertex_positions - simplified_vertex_positions, axis=2)  # 计算欧几里得距离，形状为 (N, M)

        # 找到每个简化顶点最近的原始顶点索引
        preserved_indices = np.argmin(distances, axis=0)  # 找到距离最小的原始顶点索引，结果为 (M,)

        print(f'Simplified mesh has {simplified_mesh.vertex_number()} vertices and {simplified_mesh.face_number()} faces')

        # 保存简化网格和保留的索引
        ms.save_current_mesh(os.path.join(save_folder, basename))
        with open(os.path.join(save_folder, f"{basename.split('.')[0]}_idx.txt"), 'w') as f:
            for idx in preserved_indices:
                f.write(f"{idx},")
        pass

if __name__ == "__main__":

    # PATH_TO_GRAB = "$YOUR_PATH_TO_GRAB"
    PATH_TO_GRAB = "E:/Datasets/HOI/GRAB/GRAB-dataset-extract"

    main(PATH_TO_GRAB)