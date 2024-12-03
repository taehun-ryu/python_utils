import numpy as np
import yaml
import cv2
from functools import singledispatchmethod

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import Quaternion as Q

class CalibrationYAML:
    data = None

    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = yaml.safe_load(file)

    def scale_intrinsics(self, camera_matrix, original_size, new_size):
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        scaled_matrix = camera_matrix.copy()
        scaled_matrix[0, 0] *= scale_x  # fx
        scaled_matrix[1, 1] *= scale_y  # fy
        scaled_matrix[0, 2] *= scale_x  # cx
        scaled_matrix[1, 2] *= scale_y  # cy
        return scaled_matrix

    def save_camera_info_yaml(self, file_path, camera_name, image_width, image_height, camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix):
        camera_info = {
            "camera_name": camera_name,
            "image_width": image_width,
            "image_height": image_height,
            "camera_matrix": {
                "rows": 3,
                "cols": 3,
                "data": camera_matrix.flatten().tolist()
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {
                "rows": 1,
                "cols": len(distortion_coeffs),
                "data": distortion_coeffs.flatten().tolist()
            },
            "rectification_matrix": {
                "rows": 3,
                "cols": 3,
                "data": rectification_matrix.flatten().tolist()
            },
            "projection_matrix": {
                "rows": 3,
                "cols": 4,
                "data": projection_matrix.flatten().tolist()
            }
        }

        # YAML 저장
        with open(file_path, 'w') as file:
            file.write("#%YAML:1.0\n")  # YAML 버전 표기
            yaml.dump(camera_info, file, default_flow_style=None, sort_keys=False)

    @singledispatchmethod
    def print_tf(self, arg):
        raise NotImplementedError("Unsupported type")

    @print_tf.register
    def _(self, right_pose: np.ndarray):
        if right_pose.shape == (4, 4):
            translation = right_pose[:3, 3]
            rotation = right_pose[:3, :3]
            q = Q.rotation_matrix_to_quaternion(rotation)
            print("tf: tx ty tz q1 q2 q3 q0")
            print(f"{translation[0]}, {translation[1]}, {translation[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

    @print_tf.register
    def _(self, args: tuple):
        if len(args) == 2:
            R, t = args
            q = Q.rotation_matrix_to_quaternion(R)
            print("tf: tx ty tz q1 q2 q3 q0")
            print(f"{t[0]}, {t[1]}, {t[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

    def calculate_rectification_matrices(self, left_intrinsic, left_distortion, right_intrinsic, right_distortion, 
                                        rotation, translation, image_size):
        """
        Calculate rectification matrices for stereo cameras.

        Parameters:
            left_intrinsic (np.array): Intrinsic matrix of the left camera.
            left_distortion (np.array): Distortion coefficients of the left camera.
            right_intrinsic (np.array): Intrinsic matrix of the right camera.
            right_distortion (np.array): Distortion coefficients of the right camera.
            rotation (np.array): Rotation matrix from left to right camera.
            translation (np.array): Translation vector from left to right camera.
            image_size (tuple): Image size as (width, height).

        Returns:
            R1, R2: Rectification matrices for the left and right cameras.
            P1, P2: Projection matrices after rectification for the left and right cameras.
            Q: Disparity-to-depth mapping matrix.
        """
        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_intrinsic, left_distortion,
            right_intrinsic, right_distortion,
            image_size, rotation, translation,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        return R1, R2, P1, P2, Q
