import cv2
import numpy as np
import os

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/

class StereoCalib:
    def __init__(self, left_camera, right_camera):
        self.left_camera_ = left_camera
        self.right_camera_ = right_camera
        self.img_shape = self.left_camera_.img_shape
        self.charuco_board = self.left_camera_.charuco_board
        # results
        self.R = None
        self.T = None

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

    def save(self, filepath, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T, ):
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", self.img_shape[0])
        fs.write("image_height", self.img_shape[1])
        fs.write("K_1", camera_matrix_1)
        fs.write("d_1", dist_coeffs_1)
        fs.write("K_2", camera_matrix_2)
        fs.write("d_2", dist_coeffs_2)
        fs.write("Rotation", R)
        fs.write("Translation", T)
        fs.release()

        # FileStorage로 저장이 끝난 후, 문자열 치환 작업
        with open(filepath, "r") as f:
            data = f.read()

        data = data.replace(" !!opencv-matrix", "")
        data = data.replace("%YAML:1.0", "#%YAML:1.0")

        with open(filepath, "w") as f:
            f.write(data)

        print(f"Stereo calibrations saved to {filepath}")

    def run(self, save=False):
        self.checkConsistency()
        print("Performing Stereo Calibration...")
        ret, self.R, self.T = self.performStereoCalibration( \
            self.left_camera_.camera_matrix, self.left_camera_.distortion, \
            self.right_camera_.camera_matrix, self.right_camera_.distortion)
        print("Stereo Calibration complete.")

        if save:
            self.save(f"{project_root}/calibration/results/stereo.yaml", \
                self.left_camera_.camera_matrix, self.left_camera_.distortion, \
                self.right_camera_.camera_matrix, self.right_camera_.distortion, \
                self.R, self.T)
