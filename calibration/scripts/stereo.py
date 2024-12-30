import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/
sys.path.append(project_root)
from calibration.tools.CameraInfoConverter import CameraInfoConverter
from calibration.algorithms.Camera import Camera
from calibration.algorithms.StereoCalib import StereoCalib
from calibration.tools.CheckerBoards import *
from calibration.tools.FramePlotter import FramePlotter

# Our Lab's 5x5 Charuco Board
board = CharucoBoard_5_5_0_11()

# 이미지 디렉토리
img_dir_1 = "/home/user/calib_data/stereo/Cam_001"
img_dir_2 = "/home/user/calib_data/stereo/Cam_002"

# 카메라 객체 생성 및 초기화
left_camera = Camera(img_dir_1, board.aruco_dict, board.board)
right_camera = Camera(img_dir_2, board.aruco_dict, board.board)
left_camera.run(save=False)
right_camera.run(save=False)

# 스테레오 캘리브레이션 실행
stereo_calib = StereoCalib(left_camera, right_camera)
stereo_calib.run(save=True)

# plot results
plotter = FramePlotter()
plotter.add_camera(stereo_calib.R.T, - stereo_calib.R.T@stereo_calib.T, label="Cam2")
plotter.plot_frames()

# Save camera info
converter = CameraInfoConverter(stereo_calib, 'stereo')
converter.run()