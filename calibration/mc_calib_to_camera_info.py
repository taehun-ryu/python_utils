import numpy as np
from CalibrationYAML import *

class McCalib:
    cam_0_pose = None
    cam_1_pose = None
    left_camera = None
    right_camera = None
    camera_rotation = None
    camera_translation = None
    camera_pose_matrix = None

    def __init__(self, data):
        self.cam_0_pose = np.array(data['camera_0']['camera_pose_matrix']['data']).reshape(4, 4)
        self.cam_1_pose = np.array(data['camera_1']['camera_pose_matrix']['data']).reshape(4, 4)
        self.camera_pose_matrix = self.cam_1_pose

        if self.cam_1_pose[0, 3] > self.cam_0_pose[0, 3]:
            self.left_camera, self.right_camera = 'camera_0', 'camera_1'
            self.camera_rotation = self.camera_pose_matrix[:3, :3].T
            self.camera_translation = np.matmul(-self.camera_rotation, self.camera_pose_matrix[:3, 3]) * 0.01 
        else:
            self.left_camera, self.right_camera = 'camera_1', 'camera_0'
            self.camera_rotation = self.camera_pose_matrix[:3, :3]
            self.camera_translation = self.camera_pose_matrix[:3, 3] * 0.01 

        
            

def main():
    file_path = 'calib_result.yaml'
    calib = CalibrationYAML(file_path)
    mccalib = McCalib(calib.data)

    # image size
    original_size = (calib.data[mccalib.left_camera]['img_width'], calib.data[mccalib.left_camera]['img_height'])
    new_size = (640, 480)

    # intrinsic parameter
    left_intrinsic = np.array(calib.data[mccalib.left_camera]['camera_matrix']['data']).reshape(3, 3)
    right_intrinsic = np.array(calib.data[mccalib.right_camera]['camera_matrix']['data']).reshape(3, 3)

    # distortion parameter
    left_distortion = np.array(calib.data[mccalib.left_camera]['distortion_vector']['data'])
    right_distortion = np.array(calib.data[mccalib.right_camera]['distortion_vector']['data'])

    # scale intrinsic parameter
    left_intrinsic_scaled = calib.scale_intrinsics(left_intrinsic, original_size, new_size)
    right_intrinsic_scaled = calib.scale_intrinsics(right_intrinsic, original_size, new_size)

    # Calculate rectification matrices
    R1, R2, P1, P2, Q = calib.calculate_rectification_matrices(
        left_intrinsic_scaled, left_distortion,
        right_intrinsic_scaled, right_distortion,
        mccalib.camera_rotation, mccalib.camera_translation, new_size
    )

    # ros camera_info에 맞게 yaml 파일로 저장
    calib.save_camera_info_yaml("left.yaml", "left_camera", new_size[0], new_size[1], left_intrinsic_scaled, left_distortion, R1, P1)
    calib.save_camera_info_yaml("right.yaml", "right_camera", new_size[0], new_size[1], right_intrinsic_scaled, right_distortion, R2, P2)

    calib.print_tf(mccalib.camera_pose_matrix)

if __name__ == '__main__':
    main()
