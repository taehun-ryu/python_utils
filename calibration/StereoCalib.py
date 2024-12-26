import cv2
import numpy as np
from Camera import Camera

class StereoCalib:
    def __init__(self, left_camera: Camera, right_camera: Camera, out_file):
        self.out_file = out_file
        self.left_camera_ = left_camera
        self.right_camera_ = right_camera
        self.img_shape = self.left_camera_.img_shape

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
        for corners_1, ids_1, corners_2, ids_2 in zip(self.all_corners_cam_1, self.all_ids_cam_1,
                                                    self.all_corners_cam_2, self.all_ids_cam_2):
            # 두 카메라에서 공통으로 검출된 코너 ID 찾기
            common_ids = np.intersect1d(ids_1.flatten(), ids_2.flatten())
            if len(common_ids) == 0:
                print("No common corners detected between the two cameras for this frame. Skipping frame.")
                continue

            # 공통 ID에 해당하는 3D 점, 2D 점 추출
            obj_pts = self.charuco_board.chessboardCorners[common_ids]
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

    def saves(self, filename, ret, R, T, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        fs.write("RMS_Error", ret)
        fs.write("Camera_Matrix_1", camera_matrix_1)
        fs.write("Distortion_Coefficients_1", dist_coeffs_1)
        fs.write("Camera_Matrix_2", camera_matrix_2)
        fs.write("Distortion_Coefficients_2", dist_coeffs_2)
        fs.write("Rotation_Matrix", R)
        fs.write("Translation_Vector", T)
        fs.release()
        print(f"Stereo calibrations saved to {filename}")

    def run(self):
        print("Performing Stereo Calibration...")
        ret, R, T = self.perform_stereo_calibration( \
            self.left_camera_.camera_matrix, self.left_camera_.distortion, \
            self.right_camera_.camera_matrix, self.right_camera_.distortion)

        print("Stereo Calibrations:")
        print("RMS Error:", ret)
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (T):\n", T)

        self.saves("stereo_calibrations.yaml", ret, R, T, \
            self.left_camera_.camera_matrix, self.left_camera_.distortion, \
            self.right_camera_.camera_matrix, self.right_camera_.distortion)


if __name__ == "__main__":
    # ArUco 및 Charuco 보드 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    '''
    6 x 9
    '''
    number_x_square = 6
    number_y_square = 9
    length_square = 0.083  # 사각형 크기 (미터 단위)
    length_marker = 0.062  # 마커 크기 (미터 단위)

    charuco_board = cv2.aruco.CharucoBoard(
        (number_x_square, number_y_square), length_square, length_marker, aruco_dict
    )
    # 이미지 디렉토리
    img_dir_1 = "/dev/ssd2/ocam/calib/calib_1204/Cam_001"
    img_dir_2 = "/dev/ssd2/ocam/calib/calib_1204/Cam_002"
    # 카메라 객체 생성 및 초기화
    left_camera = Camera(img_dir_1, aruco_dict, charuco_board)
    right_camera = Camera(img_dir_2, aruco_dict, charuco_board)
    left_camera.initFrame()
    right_camera.initFrame()
    left_camera.initCalibration()
    right_camera.initCalibration()

    # 스테레오 캘리브레이션 실행
    out_file = "/dev/ssd2/ocam/calib/calib_1204/stereo_result.yaml"
    stereo_calib = StereoCalib(left_camera, right_camera, out_file)
    stereo_calib.checkConsistency()
    stereo_calib.run()