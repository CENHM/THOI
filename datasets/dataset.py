import torch
import numpy as np
from torch.utils import data

import struct
import re

from utils.rotation import rotvec_to_rot6d
from utils.utils import (
    align_param_size,
    params_to_torch,
)
from datasets.utils import (
    contact_map_avg,
    read_csv_file,
    read_npz_file,
    read_npy_file,
    read_txt_file,
    get_mesh_from_ply_file)


class Dataset(data.Dataset):
    def __init__(self, dir, dname, maxfn, train=True):
        
        assert dname in ['GRAB']

        self.train = train
        self.dname = dname
        self.maxfn = maxfn

        if dname == "GRAB":
            self.BASE_DIR = dir
            self.ANNOTATION_DIR = f"{dir}/tools/frame_prompt_annotation"
            self.PARAMS_DIR = f"{dir}/grab"
            self.OBJ_MESH_DIR = f"{dir}/tools/object_meshes/contact_meshes_preprocess"
            self.HAND_MESH_DIR = f"{dir}/tools/subject_meshes"

            self.len = len(read_csv_file(f"{self.ANNOTATION_DIR}/annotation_train.csv").index)


    def getitem_from_grab(self, idx):
        if self.train:
            [subject, file_name, start_frame, end_frame, prompt] = read_csv_file(
                f"{self.ANNOTATION_DIR}/annotation_train.csv", loc=idx)
            
            load_data = read_npz_file(f"{self.PARAMS_DIR}/{subject}/{file_name}.npz")

            nframe = end_frame - start_frame
            obj_name = load_data["obj_name"][()]
            obj_param = load_data["object"][()]["params"]
            lh_param = load_data["lhand"][()]["params"]
            rh_param = load_data["rhand"][()]["params"]
            contact_map = load_data["contact"][()]["object"]

            del load_data

            lh_param = params_to_torch(
                lh_param, dtype=torch.float32, start=start_frame, end=end_frame)
            lh_motion = torch.cat([
                lh_param["transl"], 
                rotvec_to_rot6d(torch.cat([lh_param["global_orient"], lh_param["fullpose"]], dim=1).reshape(-1, 3)).reshape(nframe, -1)], dim=1)
            
            rh_param = params_to_torch(
                rh_param, dtype=torch.float32, start=start_frame, end=end_frame)
            rh_motion = torch.cat([
                rh_param["transl"], 
                rotvec_to_rot6d(torch.cat([rh_param["global_orient"], rh_param["fullpose"]], dim=1).reshape(-1, 3)).reshape(nframe, -1)], dim=1)

            obj_param = params_to_torch(
                obj_param, dtype=torch.float32, start=start_frame, end=end_frame)
            obj_motion = torch.cat([
                obj_param["transl"], rotvec_to_rot6d(obj_param["global_orient"]), torch.zeros(nframe, 1)], dim=1)
            
            data_dict = {
                "lh_motion": lh_motion, 
                "rh_motion": rh_motion, 
                "obj_motion": obj_motion, 
            }
            data_dict = align_param_size(data_dict, self.maxfn)

            contact_map_idx = np.array(read_txt_file(f"{self.OBJ_MESH_DIR}/{obj_name}_idx.txt").split(',')[:-1], dtype=np.int)
            contact_map = contact_map_avg(contact_map, nframe)
            contact_map = torch.tensor(contact_map[contact_map_idx], dtype=torch.float).unsqueeze(-1)

            obj_verts, _ = get_mesh_from_ply_file(f"{self.OBJ_MESH_DIR}/{obj_name}.ply")
            obj_verts = torch.tensor(np.asarray(obj_verts))

            return {
                "lh_motion": data_dict["lh_motion"].float(), 
                "rh_motion": data_dict["rh_motion"].float(),
                "obj_motion": data_dict["obj_motion"].float(),
                "obj_name": obj_name, 
                "obj_verts": obj_verts.float(),
                "prompt": prompt,
                "contact_map": contact_map,
                "nframe": torch.tensor(nframe).reshape(1)
            }

        else:
            pass
        

    def __getitem__(self, idx):
        return self.getitem_from_grab(idx)


    def __len__(self):
        return self.len
    






