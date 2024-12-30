import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/
sys.path.append(project_root)
from calibration.tools.CalibrationYAML import CalibrationYAML
from calibration.algorithms.Camera import Camera
from calibration.algorithms.NonOverlapCalib import NonOverlapCalib
from calibration.tools.CheckerBoards import *

# Our Lab's Charuco Board 1
board_1 = CharucoBoard_6_9_0_26()
# Our Lab's Charuco Board 2
board_2 = CharucoBoard_6_9_27_53()

# 이미지 디렉토리
img_dir_1 = "/home/user/calib_data/non_overlap/1/Cam_001"
img_dir_2 = "/home/user/calib_data/non_overlap/1/Cam_002"
# 카메라 객체 생성 및 초기화
camera_1 = Camera(img_dir_1, board_1.aruco_dict, board_1.board)
camera_2 = Camera(img_dir_2, board_2.aruco_dict, board_2.board)
camera_1.run(save=False)
camera_2.run(save=False)

non_overlap_calib = NonOverlapCalib(camera_1, camera_2)
non_overlap_calib.run(False)
