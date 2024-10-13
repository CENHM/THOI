import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils.utils import visualize_obj_file

class Object:
    def __init__(self, path='E:/Datasets/H2O/object/book/book.obj'):
        self.vertices = self.__get_vertices(path)

    def __get_vertices(self, path):
        vertices = []
        with open(path) as f:
            for line in f.readlines():
                if line.startswith('v'):
                    coords = line.split()
                    vertices.append([float(coord) for coord in coords[1:4]])        
        return np.array(vertices)

    def compute_scale(self):
        centroid = np.mean(self.vertices, axis=1)
        s_obj = np.max(np.sqrt(np.sum((self.vertices - centroid)**2, axis=1))) 
        # `s_obj` is also the maximum distance from the center of object mesh to its vertices.
        return s_obj

