import os
import yaml
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class PickingPoints:
    def __init__(self, config_path):
        if not os.path.exists(config_path):
            print("{} does not exist".format(config_path))
            return

        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.K = []
        self.picking_points = None

        self.initialize()

    def initialize(self):
        self.K = self.config["K"]
        self.picking_points = self.config["picking_points"]

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

        return [picking_point_rotation_matrix]

    @staticmethod
    def transform(origin_rotation_matrix, picking_point):
        rotation = Rotation.from_euler("xyz", picking_point["rotation"], degrees=True).as_matrix()
        translation = np.reshape(np.array(picking_point["translation"], dtype=np.float), (3,))

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        picking_point_rotation_matrix = np.dot(origin_rotation_matrix, transformation_matrix)

        return picking_point_rotation_matrix

    @staticmethod
    def draw_axis(image, picking_point_rotation_matrix, K, scale=1):
        R = picking_point_rotation_matrix[:3, :3]
        t = picking_point_rotation_matrix[:3, 3]

        rotation_vector, _ = cv2.Rodrigues(R)
        points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
        axis_points, _ = cv2.projectPoints(points, rotation_vector, t, K, (0, 0, 0, 0))

        axis = [tuple(ap.ravel().astype(np.int32)) for ap in axis_points]
        center = tuple(axis_points[3].ravel().astype(np.int32))
        image = cv2.line(image, center, axis[2], (0, 0, 255), 2)
        image = cv2.line(image, center, axis[1], (0, 255, 0), 2)
        image = cv2.line(image, center, axis[0], (255, 0, 0), 2)
        image = cv2.circle(image, center, 2, 1, -1)
        image = cv2.circle(image, axis[2], 2, 1, -1)
        image = cv2.circle(image, axis[1], 2, 1, -1)
        image = cv2.circle(image, axis[0], 2, 1, -1)
        return image
