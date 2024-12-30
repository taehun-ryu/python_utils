import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/
sys.path.append(project_root)
from calibration.algorithms.Camera import Camera
from calibration.tools.CheckerBoards import *

# Our Lab's 5x5 Charuco Board
board = CharucoBoard_5_5_0_11()
# Our Lab's 6x9 Charuco Board 1
board_1 = CharucoBoard_6_9_0_26()
# Our Lab's 6x9 Charuco Board 2
board_2 = CharucoBoard_6_9_27_53()

# 이미지 디렉토리
img_dir = "/home/user/calib_data/stereo/Cam_001"

cam = Camera(img_dir, board.aruco_dict, board.board)
cam.run(save=True)