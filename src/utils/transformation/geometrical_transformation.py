import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class GeometricalTransformation:
    def __init__(self):
        self.K = []
        self.picking_points = None

        self.initialize()

    def initialize(self):
        pass

    def compute(self, origin_rotation_matrix, index: int = None):
        if index is not None and index >= len(self.picking_points):
            print("Index not in picking points range")
            return None
        if index is None:
            picking_points_rotation_matrix = self.all(origin_rotation_matrix)
            return picking_points_rotation_matrix
        else:
            picking_point_rotation_matrix = self.single(origin_rotation_matrix, index)
            return picking_point_rotation_matrix

    def all(self, origin_rotation_matrix):
        picking_points_rotation_matrix = []
        for pp in self.picking_points:
            picking_point_rotation_matrix = self.transform(origin_rotation_matrix, pp)
            picking_points_rotation_matrix.append(picking_point_rotation_matrix)

        return picking_points_rotation_matrix

    def single(self, origin_rotation_matrix, index):
        pp = self.picking_points[index]
        picking_point_rotation_matrix = self.transform(origin_rotation_matrix, pp)

        return picking_point_rotation_matrix

    @staticmethod
    def transform(origin_rotation_matrix, picking_point):
        rotation = Rotation.from_euler("xyz", picking_point["rotation"], degrees=True).as_matrix()
        translation = np.reshape(np.array(picking_point["translation"], dtype=np.float), (3,))

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        picking_point_rotation_matrix = np.dot(origin_rotation_matrix, transformation_matrix)

        return picking_point_rotation_matrix

    def center(self, picking_point_rotation_matrix):
        R = picking_point_rotation_matrix[:3, :3]
        t = picking_point_rotation_matrix[:3, 3]
        rotation_vector, _ = cv2.Rodrigues(R)
        points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
        axis_points, _ = cv2.projectPoints(points, rotation_vector, t, self.K, (0, 0, 0, 0))
        center = tuple(axis_points[3].ravel().astype(np.int32))
        return center
