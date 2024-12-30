import os
import cv2
import numpy as np
from Camera import Camera
from CheckerBoards import CharucoBoard_5_5_0_11

class StereoCalib:
    def __init__(self, left_camera: Camera, right_camera: Camera):
        self.left_camera_ = left_camera
        self.right_camera_ = right_camera
        self.img_shape = self.left_camera_.img_shape
        self.charuco_board = self.left_camera_.charuco_board

    def checkConsistency(self):
        """
        두 카메라의 이미지 수, 코너 수, 아이디 수가 일치하는지 확인
        """
        if len(self.left_camera_.all_corners) != len(self.right_camera_.all_corners):
            raise ValueError("Number of frames captured by the two cameras do not match.")
        if len(self.left_camera_.all_ids) != len(self.right_camera_.all_ids):
            raise ValueError("Number of frames with detected corners do not match.")
        if len(self.left_camera_.all_corners) != len(self.left_camera_.all_ids):
            raise ValueError("Number of frames with detected corners and IDs do not match.")
        if len(self.right_camera_.all_corners) != len(self.right_camera_.all_ids):
            raise ValueError("Number of frames with detected corners and IDs do not match.")

    def performStereoCalibration(self, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2):
        flags = cv2.CALIB_FIX_INTRINSIC
        object_points = []
        image_points_1 = []
        image_points_2 = []

        # 각 프레임의 코너 ID를 기반으로 3D-2D 매칭 생성
        for corners_1, ids_1, corners_2, ids_2 in zip(self.left_camera_.all_corners, self.left_camera_.all_ids,
                                                    self.right_camera_.all_corners, self.right_camera_.all_ids):
            # 두 카메라에서 공통으로 검출된 코너 ID 찾기
            common_ids = np.intersect1d(ids_1.flatten(), ids_2.flatten())
            if len(common_ids) == 0:
                print("No common corners detected between the two cameras for this frame. Skipping frame.")
                continue

            # 공통 ID에 해당하는 3D 점, 2D 점 추출
            obj_pts = self.charuco_board.getChessboardCorners()[common_ids]
            img_pts_1 = [corners_1[ids_1.flatten() == id][0] for id in common_ids]
            img_pts_2 = [corners_2[ids_2.flatten() == id][0] for id in common_ids]

            object_points.append(obj_pts)
            image_points_1.append(np.array(img_pts_1))
            image_points_2.append(np.array(img_pts_2))

        # 3D-2D 매칭 데이터가 충분한지 확인
        if len(object_points) == 0 or len(image_points_1) == 0 or len(image_points_2) == 0:
            raise ValueError("Insufficient data for stereo calibration. Check Charuco corner detection.")

        # 스테레오 캘리브레이션 수행
        ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objectPoints=object_points,
            imagePoints1=image_points_1,
            imagePoints2=image_points_2,
            cameraMatrix1=camera_matrix_1,
            distCoeffs1=dist_coeffs_1,
            cameraMatrix2=camera_matrix_2,
            distCoeffs2=dist_coeffs_2,
            imageSize=self.img_shape,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 1e-6),
        )
        return ret, R, T

    def run(self):
        print("Performing Stereo Calibration...")
        ret, R, T = self.performStereoCalibration( \
            self.left_camera_.camera_matrix, self.left_camera_.distortion, \
            self.right_camera_.camera_matrix, self.right_camera_.distortion)

        print("Stereo Calibrations:")
        print("RMS Error:", ret)
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)

if __name__ == "__main__":
    # Our Lab's 5x5 Charuco Board
    board = CharucoBoard_5_5_0_11()

    # 이미지 디렉토리
    img_dir_1 = "/home/user/calib_data/stereo/Cam_001"
    img_dir_2 = "/home/user/calib_data/stereo/Cam_002"

    # 카메라 객체 생성 및 초기화
    left_camera = Camera(img_dir_1, board.aruco_dict, board.board)
    right_camera = Camera(img_dir_2, board.aruco_dict, board.board)
    left_camera.initFrame()
    right_camera.initFrame()
    left_camera.initCalibration()
    right_camera.initCalibration()

    # 스테레오 캘리브레이션 실행
    stereo_calib = StereoCalib(left_camera, right_camera)
    stereo_calib.checkConsistency()
    stereo_calib.run()