import numpy as np
from CalibrationYAML import *

def main():
    file_path = 'calib_result.yaml'
    calib = CalibrationYAML(file_path)

    # image size
    size = (640, 480)

    # intrinsic parameter
    left_intrinsic = np.array(calib.data['mtx_left']['data']).reshape(3, 3)
    right_intrinsic = np.array(calib.data['mtx_right']['data']).reshape(3, 3)

    # distortion parameter
    left_distortion = np.array(calib.data['dist_left']['data'])
    right_distortion = np.array(calib.data['dist_right']['data'])

    # right camera’s pose in the left camera’s frame
    camera_rotation = np.array(calib.data['R']['data']).reshape(3, 3)
    camera_translation = np.array(calib.data['T']['data']).reshape(3, 1).flatten()

    # Calculate rectification matrices
    R1, R2, P1, P2, Q = calib.calculate_rectification_matrices(
        left_intrinsic, left_distortion,
        right_intrinsic, right_distortion,
        camera_rotation, camera_translation,
        size
    )

    # ros camera_info에 맞게 yaml 파일로 저장
    calib.save_camera_info_yaml("left.yaml", "left_camera", size[0], size[1], left_intrinsic, left_distortion, R1, P1)
    calib.save_camera_info_yaml("right.yaml", "right_camera", size[0], size[1], right_intrinsic, right_distortion, R2, P2)

    calib.print_tf((camera_rotation, camera_translation))

if __name__ == '__main__':
    main()
