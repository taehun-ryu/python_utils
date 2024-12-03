import numpy as np
from CalibrationYAML import *

def determine_cameras(data):
    cam_0_pose = np.array(data['camera_0']['camera_pose_matrix']['data']).reshape(4, 4)
    cam_1_pose = np.array(data['camera_1']['camera_pose_matrix']['data']).reshape(4, 4)

    if cam_1_pose[0, 3] > cam_0_pose[0, 3]:
        left_camera, right_camera = 'camera_0', 'camera_1'
        is_based_on_left = True
    else:
        left_camera, right_camera = 'camera_1', 'camera_0'
        is_based_on_left = False

    return left_camera, right_camera, is_based_on_left

def set_camera_pose_matrix(camera_pose_matrix, is_based_on_left):
    if is_based_on_left:
        # from the coordinate system of the first camera to the second camera
        camera_rotation = camera_pose_matrix[:3, :3]
        camera_translation = np.matmul(camera_rotation, camera_pose_matrix[:3, 3] * 0.01)
    else:
        raise NotImplementedError("Not implemented yet") #TODO: how can i get the correct R ant t?

    camera_rotation = camera_pose_matrix[:3, :3]
    camera_translation = camera_pose_matrix[:3, 3]

    return camera_rotation, camera_translation

def main():
    file_path = 'calib_result.yaml'
    calib = CalibrationYAML(file_path)

    # 둘 중 어느 카메라가 왼쪽, 오른쪽인지 결정
    left_camera, right_camera, is_based_on_left = determine_cameras(calib.data)

    # image size
    original_size = (calib.data[left_camera]['img_width'], calib.data[left_camera]['img_height'])
    new_size = (640, 480)

    # intrinsic parameter
    left_intrinsic = np.array(calib.data[left_camera]['camera_matrix']['data']).reshape(3, 3)
    right_intrinsic = np.array(calib.data[right_camera]['camera_matrix']['data']).reshape(3, 3)

    # distortion parameter
    left_distortion = np.array(calib.data[left_camera]['distortion_vector']['data'])
    right_distortion = np.array(calib.data[right_camera]['distortion_vector']['data'])

    # scale intrinsic parameter
    left_intrinsic_scaled = calib.scale_intrinsics(left_intrinsic, original_size, new_size)
    right_intrinsic_scaled = calib.scale_intrinsics(right_intrinsic, original_size, new_size)

    # right camera’s pose in the left camera’s frame
    camera_pose_matrix = np.array(calib.data[right_camera]['camera_pose_matrix']['data']).reshape(4, 4)
    R, t = set_camera_pose_matrix(camera_pose_matrix, is_based_on_left)

    # Calculate rectification matrices
    R1, R2, P1, P2, Q = calib.calculate_rectification_matrices(
        left_intrinsic_scaled, left_distortion,
        right_intrinsic_scaled, right_distortion,
        R, t, new_size
    )

    # ros camera_info에 맞게 yaml 파일로 저장
    calib.save_camera_info_yaml("left.yaml", "left_camera", new_size[0], new_size[1], left_intrinsic_scaled, left_distortion, R1, P1)
    calib.save_camera_info_yaml("right.yaml", "right_camera", new_size[0], new_size[1], right_intrinsic_scaled, right_distortion, R2, P2)

    calib.print_tf(camera_pose_matrix)

if __name__ == '__main__':
    main()
