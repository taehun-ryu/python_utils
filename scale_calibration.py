import cv2
import numpy as np
import yaml

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

def calculate_right_projection_matrix(right_intrinsic, right_pose):
    rotation = right_pose[:3, :3]
    translation = right_pose[:3, 3].reshape(-1, 1)
    projection_matrix = np.dot(right_intrinsic, np.hstack((rotation, translation)))
    return projection_matrix

def save_camera_info_yaml(file_path, image_width, image_height, camera_matrix, distortion_coeffs, projection_matrix):
    camera_info = {
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
            "cols": 5,
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
    
    with open(file_path, 'w') as file:
        yaml.dump(camera_info, file)

def main():
    file_path = 'calib_result.yaml'
    data = read_yaml(file_path)
    
    # Determine which is left and right camera
    left_camera, right_camera = determine_cameras(data)
    
    # Original and new image sizes
    original_size = (data[left_camera]['img_width'], data[left_camera]['img_height'])
    new_size = (640, 480)
    
    # Load intrinsic matrices
    left_intrinsic = np.array(data[left_camera]['camera_matrix']['data']).reshape(3, 3)
    right_intrinsic = np.array(data[right_camera]['camera_matrix']['data']).reshape(3, 3)
    
    # Load distortion coefficients
    left_distortion = np.array(data[left_camera]['distortion_vector']['data'])
    right_distortion = np.array(data[right_camera]['distortion_vector']['data'])
    
    # Scale intrinsic matrices
    left_intrinsic_scaled = scale_intrinsics(left_intrinsic, original_size, new_size)
    right_intrinsic_scaled = scale_intrinsics(right_intrinsic, original_size, new_size)
    
    # Create left camera projection matrix (3x4) by extending 3x3 intrinsic matrix
    left_projection_matrix = np.hstack((left_intrinsic_scaled, np.zeros((3, 1))))
    
    # Calculate right camera projection matrix
    right_pose_matrix = np.array(data[right_camera]['camera_pose_matrix']['data']).reshape(4, 4)
    
    # Save to left.yaml and right.yaml
    save_camera_info_yaml("left.yaml", new_size[0], new_size[1], left_intrinsic_scaled, left_distortion, left_projection_matrix)
    save_camera_info_yaml("right.yaml", new_size[0], new_size[1], right_intrinsic_scaled, right_distortion, right_projection_matrix)

if __name__ == '__main__':
    main()
