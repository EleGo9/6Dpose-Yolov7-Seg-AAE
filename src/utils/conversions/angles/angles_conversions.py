from scipy.spatial.transform import Rotation


class AnglesConversions:
    @staticmethod
    def rotation_matrix_to_euler_angles(rotation_matrix, sequence="xyz", degrees=False):
        euler_angles = Rotation.from_matrix(rotation_matrix).as_euler(seq=sequence, degrees=degrees)
        return euler_angles

    @staticmethod
    def rotation_matrix_to_rotation_vector(rotation_matrix):
        rotation_vector = Rotation.from_matrix(rotation_matrix).as_rotvec()
        return rotation_vector

    @staticmethod
    def euler_angles_to_rotation_matrix(euler_angles):
        rotation_matrix = Rotation.from_euler("xyz", euler_angles, degrees=False).as_matrix()
        return rotation_matrix

    @staticmethod
    def euler_angles_to_rotation_vector(euler_angles):
        rotation_vector = Rotation.from_euler("xyz", euler_angles, degrees=False).as_rotvec()
        return rotation_vector

