import cv2
import numpy as np
import yaml
import Quaternion as Q

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

def calculate_right_projection_matrix(right_intrinsic, right_pose, scale="m"):
    if scale == 'm':
        scale_factor = 1
    elif scale == 'cm':
        scale_factor = 0.01
    else:
        raise Exception("Type should be [m] or [cm].")
    # Extract rotation and translation
    rotation_from_right_to_left = right_pose[:3, :3].T  # Transpose for inverse rotation
    translation_from_right_to_left = np.matmul(-rotation_from_right_to_left, right_pose[:3, 3] * scale_factor)

    Rt = np.hstack((rotation_from_right_to_left, translation_from_right_to_left.reshape(-1, 1)))

    projection_matrix = np.matmul(right_intrinsic, Rt)


    return projection_matrix

def save_camera_info_yaml(file_path, camera_name, image_width, image_height, camera_matrix, distortion_coeffs, projection_matrix):
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
            "data": np.eye(3).flatten().tolist()  # Identity matrix for rectification
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

def print_left_to_right_tf(right_pose):
    translation = right_pose[:3, 3]
    rotation = right_pose[:3, :3]
    q = Q.rotation_matrix_to_quaternion(rotation)
    print("tf: tx ty tz q1 q2 q3 q0") 
    print(f"{translation[0]}, {translation[1]}, {translation[2]}, {q[1]}, {q[1]}, {q[3]}, {q[0]}")

def main():
    file_path = 'calib_result.yaml'
    data = read_yaml(file_path)

    # 둘 중 어느 카메라가 왼쪽, 오른쪽인지 결정
    left_camera, right_camera = determine_cameras(data)

    # image size 결정
    original_size = (data[left_camera]['img_width'], data[left_camera]['img_height'])
    new_size = (640, 480)

    # intrinsic parameter 로드
    left_intrinsic = np.array(data[left_camera]['camera_matrix']['data']).reshape(3, 3)
    right_intrinsic = np.array(data[right_camera]['camera_matrix']['data']).reshape(3, 3)

    # distortion parameter 로드
    left_distortion = np.array(data[left_camera]['distortion_vector']['data'])
    right_distortion = np.array(data[right_camera]['distortion_vector']['data'])

    # scale intrinsic parameter 계산
    left_intrinsic_scaled = scale_intrinsics(left_intrinsic, original_size, new_size)
    right_intrinsic_scaled = scale_intrinsics(right_intrinsic, original_size, new_size)

    # left camrea projection matrix
    left_projection_matrix = np.hstack((left_intrinsic_scaled, np.zeros((3, 1))))

    # right camera projection matrix
    right_pose_matrix = np.array(data[right_camera]['camera_pose_matrix']['data']).reshape(4, 4)
    right_projection_matrix = calculate_right_projection_matrix(right_intrinsic_scaled, right_pose_matrix, 'cm')

    # ros camera_info에 맞게 yaml 파일로 저장
    save_camera_info_yaml("left.yaml", "flir_left", new_size[0], new_size[1], left_intrinsic_scaled, left_distortion, left_projection_matrix)
    save_camera_info_yaml("right.yaml", "flir_right", new_size[0], new_size[1], right_intrinsic_scaled, right_distortion, right_projection_matrix)
    
    print_left_to_right_tf(right_pose_matrix)

if __name__ == '__main__':
    main()
