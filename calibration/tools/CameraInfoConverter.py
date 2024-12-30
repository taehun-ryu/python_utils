import numpy as np
import yaml
import cv2
from functools import singledispatchmethod
import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/
sys.path.append(project_root)
from pose import Quaternion as Q

class CameraInfoConverter:
  def __init__(self, calib, type):
    self.calib = calib
    self.type = type
    self.data = None

  def loadYaml(self, type):
    if type == 'mono':
      file_path = f"{project_root}/calibration/results/mono.yaml"
    elif type == 'stereo':
      file_path = f"{project_root}/calibration/results/stereo.yaml"
    elif type == 'non_overlap':
      file_path = f"{project_root}/calibration/results/non_overlap.yaml"
    else:
      raise ValueError("Invalid type. Choose from 'mono', 'stereo', 'non_overlap'.")

    with open(file_path, 'r') as file:
      self.data = yaml.safe_load(file)

  def saveCameraInfoYaml(self, file_name, camera_name, image_width, image_height, camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix):
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

        calibration_dir = os.path.join(f'{project_root}', 'calibration', 'results')
        os.makedirs(calibration_dir, exist_ok=True)
        # 파일 경로 완성
        full_file_path = os.path.join(calibration_dir, file_name)
        # YAML 저장
        with open(full_file_path, 'w') as file:
            file.write("#%YAML:1.0\n")  # YAML 버전 표기
            yaml.dump(camera_info, file, default_flow_style=None, sort_keys=False)

  @singledispatchmethod
  def printTF(self, arg):
      raise NotImplementedError("Unsupported type")

  @printTF.register
  def _(self, right_pose: np.ndarray):
      if right_pose.shape == (4, 4):
          translation = right_pose[:3, 3]
          rotation = right_pose[:3, :3]
          q = Q.rotation_matrix_to_quaternion(rotation)
          print("tf: tx ty tz q1 q2 q3 q0")
          print(f"{translation[0]}, {translation[1]}, {translation[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

  @printTF.register
  def _(self, args: tuple):
      if len(args) == 2:
          R, t = args
          q = Q.rotation_matrix_to_quaternion(R)
          print("tf: tx ty tz q1 q2 q3 q0")
          print(f"{t[0]}, {t[1]}, {t[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

  def run(self):
    self.loadYaml(self.type)

    if self.type == 'mono':
      width = self.data['image_width']
      height = self.data['image_height']
      proj = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
      ])
      intrinsic = np.array(self.data['K']['data']).reshape(3, 3)
      distortion = np.array(self.data['d']['data'])
      self.saveCameraInfoYaml('mono_camera_info.yaml', 'mono_camera', \
                              width, height, \
                              intrinsic, distortion, np.eye(3), proj)

    elif self.type == 'stereo':
      width = self.data['image_width']
      height = self.data['image_height']
      intrinsic1 = np.array(self.data['K_1']['data']).reshape(3, 3)
      distortion1 = np.array(self.data['d_1']['data'])
      intrinsic2 = np.array(self.data['K_2']['data']).reshape(3, 3)
      distortion2 = np.array(self.data['d_2']['data'])
      rotation = np.array(self.data['Rotation']['data']).reshape(3, 3)
      translation = np.array(self.data['Translation']['data']).reshape(3, 1).flatten()
      R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
          intrinsic1, distortion1,
          intrinsic2, distortion2,
          (width, height), rotation, translation,
          flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
      )
      self.saveCameraInfoYaml('stereo_camera_info_1.yaml', 'left_camera', \
                              width, height, \
                              intrinsic1, distortion1, R1, P1)
      self.saveCameraInfoYaml('stereo_camera_info_2.yaml', 'right_camera', \
                              width, height, \
                              intrinsic2, distortion2, R2, P2)
      self.printTF((rotation.T, -rotation.T @ translation))

    elif self.type == 'non_overlap':
      raise NotImplementedError("Non-overlap converter is not supported yet.")
    else:
      raise ValueError("Invalid type. Choose from 'mono', 'stereo', 'non_overlap'.")