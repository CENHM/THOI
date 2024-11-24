import pandas as pd
import torch
import numpy as np
import re
import pywavefront
import os
import open3d


def read_csv_file(dir, **kwargs):
    content = pd.read_csv(dir, encoding="utf-8")
    if "loc" in kwargs:   
        return content.loc[kwargs['loc']]
    else:
        return content


def read_npz_file(dir):
    content = np.load(dir, allow_pickle=True)
    return content


def read_npy_file(dir):
    content = np.load(dir)
    return content


def get_mesh_from_ply_file(dir):
    content = read_ply_file(dir)
    return content.vertices, content.triangles


def read_ply_file(dir):
    content = open3d.io.read_triangle_mesh(dir)
    return content


def read_txt_file(dir):
    with open(dir, "r") as f: 
        content = f.read()
    return content


def contact_map_avg(contact_map, nframe):
    contact_map[contact_map == 21] = 26
    contact_map[contact_map == 22] = 26
    contact_map[contact_map < 26] = 0
    contact_map[contact_map >= 26] = 1
    contact_map = np.sum(contact_map, axis=0) / nframe
    # contact_map[contact_map > 0] = 1
    return contact_map

