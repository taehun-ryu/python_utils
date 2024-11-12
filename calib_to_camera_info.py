import math
import numpy as np
import yaml
import Quaternion as Q
import cv2

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def scale_intrinsics(camera_matrix, original_size, new_size):
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    scaled_matrix = camera_matrix.copy()
    scaled_matrix[0, 0] *= scale_x  # fx
    scaled_matrix[1, 1] *= scale_y  # fy
    scaled_matrix[0, 2] *= scale_x  # cx
    scaled_matrix[1, 2] *= scale_y  # cy
    return scaled_matrix

def determine_cameras(data):
    cam_0_pose = np.array(data['camera_0']['camera_pose_matrix']['data']).reshape(4, 4)
    cam_1_pose = np.array(data['camera_1']['camera_pose_matrix']['data']).reshape(4, 4)

    if cam_1_pose[0, 3] > cam_0_pose[0, 3]:
        left_camera, right_camera = 'camera_0', 'camera_1'
    else:
        left_camera, right_camera = 'camera_1', 'camera_0'

    return left_camera, right_camera

def save_camera_info_yaml(file_path, camera_name, image_width, image_height, camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix):
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

    # YAML 저장
    with open(file_path, 'w') as file:
        file.write("#%YAML:1.0\n")  # YAML 버전 표기
        yaml.dump(camera_info, file, default_flow_style=None, sort_keys=False)

def print_left_to_right_tf(right_pose): # right_pose: 4x4
    translation = right_pose[:3, 3]
    rotation = right_pose[:3, :3]
    q = Q.rotation_matrix_to_quaternion(rotation)
    print("tf: tx ty tz q1 q2 q3 q0")
    print(f"{translation[0]}, {translation[1]}, {translation[2]}, {q[1]}, {q[1]}, {q[3]}, {q[0]}")

def calculate_rectification_matrices(left_intrinsic, left_distortion, right_intrinsic, right_distortion, 
                                     rotation, translation, image_size):
    """
    Calculate rectification matrices for stereo cameras.
    
    Parameters:
        left_intrinsic (np.array): Intrinsic matrix of the left camera.
        left_distortion (np.array): Distortion coefficients of the left camera.
        right_intrinsic (np.array): Intrinsic matrix of the right camera.
        right_distortion (np.array): Distortion coefficients of the right camera.
        rotation (np.array): Rotation matrix from left to right camera.
        translation (np.array): Translation vector from left to right camera.
        image_size (tuple): Image size as (width, height).
        
    Returns:
        R1, R2: Rectification matrices for the left and right cameras.
        P1, P2: Projection matrices after rectification for the left and right cameras.
        Q: Disparity-to-depth mapping matrix.
    """
    # Compute rectification transforms
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_intrinsic, left_distortion,
        right_intrinsic, right_distortion,
        image_size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    return R1, R2, P1, P2, Q

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