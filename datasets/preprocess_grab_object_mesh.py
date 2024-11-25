# This part of code is borrowed from Junuk Cha 
# (https://github.com/JunukCha/Text2HOI/blob/main/preprocessing_grab_object.py)
# for the preprocess procedure of the GRAB dataset.
#
# Adjustment also have been made. We replace "simplification_quadric_edge_collap
# se_decimation" with "meshing_decimation_quadric_edge_collapse" based on 
# PyMeshLab official guidence (https://github.com/cnr-isti-vclab/PyMeshLab/blob/
# ef9d041fa227cc332e5a7cf01dd3fc6021cac86f/pymeshlab/keys.txt#L135).
# 
# Our code can record which vertices was selected by the filter, and store their
# correspond index in the original mesh in .txt files along with the filted .ply
# mesh files.


import os

import numpy as np
import pymeshlab
import glob
from tqdm import tqdm



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

        original_mesh = ms.current_mesh()
        original_vertex_positions = original_mesh.vertex_matrix()
        num_vertices = original_mesh.vertex_number()

        print(f'Original mesh has {num_vertices} vertices and {original_mesh.face_number()} faces')
        
        numFaces = 100 + 2 * n_target_vertices
        while (ms.current_mesh().vertex_number() > n_target_vertices):
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetfacenum=numFaces, preservenormal=True)
            numFaces = numFaces - (ms.current_mesh().vertex_number() - n_target_vertices)

        simplified_mesh = ms.current_mesh()
        simplified_vertex_positions = simplified_mesh.vertex_matrix()

        original_vertex_positions = original_vertex_positions[:, np.newaxis, :] 
        simplified_vertex_positions = simplified_vertex_positions[np.newaxis, :, :] 
        distances = np.linalg.norm(original_vertex_positions - simplified_vertex_positions, axis=2) 

        preserved_indices = np.argmin(distances, axis=0) 

        print(f'Simplified mesh has {simplified_mesh.vertex_number()} vertices and {simplified_mesh.face_number()} faces')

        ms.save_current_mesh(os.path.join(save_folder, basename))
        with open(os.path.join(save_folder, f"{basename.split('.')[0]}_idx.txt"), 'w') as f:
            for idx in preserved_indices:
                f.write(f"{idx},")
        pass

if __name__ == "__main__":

    # PATH_TO_GRAB = "$YOUR_PATH_TO_GRAB"
    PATH_TO_GRAB = "E:/Datasets/HOI/GRAB/GRAB-dataset-extract"

    main(PATH_TO_GRAB)