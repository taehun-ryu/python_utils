import numpy as np
import yaml
import cv2
from CalibrationYAML import *

def main():
    file_path = 'calib_result.yaml'
    data = read_yaml(file_path)

    # 둘 중 어느 카메라가 왼쪽, 오른쪽인지 결정
    left_camera, right_camera = determine_cameras(data)

    # image size
    original_size = (data[left_camera]['img_width'], data[left_camera]['img_height'])
    new_size = (640, 480)

    # intrinsic parameter
    left_intrinsic = np.array(data[left_camera]['camera_matrix']['data']).reshape(3, 3)
    right_intrinsic = np.array(data[right_camera]['camera_matrix']['data']).reshape(3, 3)

    # distortion parameter
    left_distortion = np.array(data[left_camera]['distortion_vector']['data'])
    right_distortion = np.array(data[right_camera]['distortion_vector']['data'])

    # scale intrinsic parameter
    left_intrinsic_scaled = scale_intrinsics(left_intrinsic, original_size, new_size)
    right_intrinsic_scaled = scale_intrinsics(right_intrinsic, original_size, new_size)

    # right camera’s pose in the left camera’s frame
    camera_pose_matrix = np.array(data[right_camera]['camera_pose_matrix']['data']).reshape(4, 4)

    # from the coordinate system of the first camera to the second camera
    camera_rotation = camera_pose_matrix[:3, :3].T
    camera_translation = np.matmul(-camera_rotation, camera_pose_matrix[:3, 3] * 0.01)

    # Calculate rectification matrices
    R1, R2, P1, P2, Q = calculate_rectification_matrices(
        left_intrinsic_scaled, left_distortion,
        right_intrinsic_scaled, right_distortion,
        camera_rotation, camera_translation,
        new_size
    )

    # ros camera_info에 맞게 yaml 파일로 저장
    save_camera_info_yaml("left.yaml", "flir_left", new_size[0], new_size[1], left_intrinsic_scaled, left_distortion, R1, P1)
    save_camera_info_yaml("right.yaml", "flir_right", new_size[0], new_size[1], right_intrinsic_scaled, right_distortion, R2, P2)

    print_left_to_right_tf(camera_pose_matrix)

if __name__ == '__main__':
    main()
