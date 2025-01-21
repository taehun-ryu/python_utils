import numpy as np
import yaml
import cv2
from functools import singledispatchmethod

def axisAngleToQuaternion(axis:np.ndarray, angle:float) -> np.ndarray:

    if isinstance(axis, list) and len(axis)==3:
        axis = np.array(axis)
    elif isinstance(axis, np.ndarray) and axis.size==3:
        pass
    else:
        raise TypeError("The axis of rotation must be given as [3x1] np.ndarray vector or a python list of 3 elements")

    if angle == 0:
        q = np.array([1,0,0,0])  #  identity quaternion
    else:
        if np.linalg.norm(axis) == 0:
            raise Exception("A valid rotation 'axis' parameter must be provided to describe a meaningful rotation.")
        else:
            u = axis/np.linalg.norm(axis)
            q = np.append(np.cos(angle/2), np.sin(angle/2)*u)

    return q


def quaternion_to_axis_angle(q:np.ndarray) -> list:
    if isinstance(q, list) and len(q)==4:
        e0 = np.array(q[0])
        e = np.array(q[1:])
    elif isinstance(q, np.ndarray) and q.size==4:
        e0 = q[0]
        e = q[1:]
    else:
        raise TypeError("The quaternion \"q\" must be given as [4x1] np.ndarray quaternion or a python list of 4 elements")

    if np.linalg.norm(e) == 0:
        axis = np.array([1,0,0]) #To be checked again
        angle = 0
    elif np.linalg.norm(e) != 0:
        axis = e/np.linalg.norm(e) 
        if e0 == 0:
            angle = np.pi
        else:
            angle = 2*np.arctan(np.linalg.norm(e)/e0) 

    return axis, angle


def quaternion_to_rotation_matrix(q:np.ndarray) -> np.ndarray:
    if isinstance(q, list) and len(q)==4:
        q = np.array(q) 
    elif isinstance(q, np.ndarray) and q.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    e0 = q[0]
    e1 = q[1]
    e2 = q[2]
    e3 = q[3]

    rotation_matrix = np.array([[e0*e0 + e1*e1 - e2*e2 - e3*e3, 2*e1*e2 - 2*e0*e3, 2*e0*e2 + 2*e1*e3],
                                [2*e0*e3 + 2*e1*e2, e0*e0 - e1*e1 + e2*e2 - e3*e3, 2*e2*e3 - 2*e0*e1],
                                [2*e1*e3 - 2*e0*e2, 2*e0*e1 + 2*e2*e3, e0*e0 - e1*e1 - e2*e2 + e3*e3]])        

    return rotation_matrix

def rotation_matrix_to_quaternion(R:np.ndarray) -> np.ndarray:
    u_q0 = np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])/4) # the prefix u_ means unsigned
    u_q1 = np.sqrt((1 + R[0,0] - R[1,1] - R[2,2])/4)
    u_q2 = np.sqrt((1 - R[0,0] + R[1,1] - R[2,2])/4)
    u_q3 = np.sqrt((1 - R[0,0] - R[1,1] + R[2,2])/4)

    q = np.array([u_q0, u_q1, u_q2, u_q3])

    if u_q0 == max(q):
        q0 = u_q0
        q1 = (R[2,1] - R[1,2])/(4*q0)
        q2 = (R[0,2] - R[2,0])/(4*q0)
        q3 = (R[1,0] - R[0,1])/(4*q0)

    if u_q1 == max(q):
        q1 = u_q1
        q0 = (R[2,1] - R[1,2])/(4*q1)
        q2 = (R[0,1] + R[1,0])/(4*q1)
        q3 = (R[0,2] + R[2,0])/(4*q1)

    if u_q2 == max(q):
        q2 = u_q2
        q0 = (R[0,2] - R[2,0])/(4*q2)
        q1 = (R[0,1] + R[1,0])/(4*q2)
        q3 = (R[1,2] + R[2,1])/(4*q2)

    if u_q3 == max(q):
        q3 = u_q3
        q0 = (R[1,0] - R[0,1])/(4*q3)
        q1 = (R[0,2] + R[2,0])/(4*q3)
        q2 = (R[1,2] + R[2,1])/(4*q3)

    q = np.array([q0, q1, q2, q3])
    return q

def euler_to_quaternion(euler_angles:np.ndarray|list)->np.ndarray:
    if isinstance(euler_angles, list) and len(euler_angles)==3:
        euler_angles = np.array(euler_angles) 
    elif isinstance(euler_angles, np.ndarray) and euler_angles.size==3:
        pass
    else:
        raise TypeError("The euler_angles must be given as [3x1] np.ndarray vector or a python list of 3 elements")

    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    q0 = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q1 = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q2 = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    q3 = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

    q = np.r_[q0, q1, q2, q3]

    return q

def quaternion_to_euler(q:np.ndarray|list)->np.ndarray:
    if isinstance(q, list) and len(q)==4:
        q = np.array(q)
    elif isinstance(q, np.ndarray) and q.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    t2 = 2.0*(q0*q2 - q1*q3)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    if t2 == 1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = -np.arctan2(q0, q1)
    elif t2 == -1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = +np.arctan2(q0, q1)
    else:
        pitch = np.arcsin(t2)
        roll = np.arctan2(2.0*(q0*q1 + q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        yaw = np.arctan2(2.0*(q0*q3 + q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

    euler_angles = np.r_[roll, pitch, yaw]

    return  euler_angles

class CalibrationYAML:
    data = None

    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = yaml.safe_load(file)

    def scale_intrinsics(self, camera_matrix, original_size, new_size):
        scale_x = new_size[0] / original_size[0]
        scale_y = new_size[1] / original_size[1]
        scaled_matrix = camera_matrix.copy()
        scaled_matrix[0, 0] *= scale_x  # fx
        scaled_matrix[1, 1] *= scale_y  # fy
        scaled_matrix[0, 2] *= scale_x  # cx
        scaled_matrix[1, 2] *= scale_y  # cy
        return scaled_matrix

    def save_camera_info_yaml(self, file_path, camera_name, image_width, image_height, camera_matrix, distortion_coeffs, rectification_matrix, projection_matrix):
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

    @singledispatchmethod
    def print_tf(self, arg):
        raise NotImplementedError("Unsupported type")

    @print_tf.register
    def _(self, right_pose: np.ndarray):
        if right_pose.shape == (4, 4):
            translation = right_pose[:3, 3]
            rotation = right_pose[:3, :3]
            q = rotation_matrix_to_quaternion(rotation)
            print("tf: tx ty tz q1 q2 q3 q0")
            print(f"{translation[0]}, {translation[1]}, {translation[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

    @print_tf.register
    def _(self, args: tuple):
        if len(args) == 2:
            R, t = args
            q = rotation_matrix_to_quaternion(R)
            print("tf: tx ty tz q1 q2 q3 q0")
            print(f"{t[0]}, {t[1]}, {t[2]}, {q[1]}, {q[2]}, {q[3]}, {q[0]}")

    def calculate_rectification_matrices(self, left_intrinsic, left_distortion, right_intrinsic, right_distortion, 
                                        rotation, translation, image_size):
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
    calib = CalibrationYAML(file_path)

    # image size
    size = (1280, 800)
    new_size = (640, 480)

    # intrinsic parameter
    left_intrinsic = np.array(calib.data['mtx_left']['data']).reshape(3, 3)
    right_intrinsic = np.array(calib.data['mtx_right']['data']).reshape(3, 3)

    # distortion parameter
    left_distortion = np.array(calib.data['dist_left']['data'])
    right_distortion = np.array(calib.data['dist_right']['data'])

    scaled_left_intrinsic = calib.scale_intrinsics(left_intrinsic,size,new_size)
    scaled_right_intrinsic = calib.scale_intrinsics(right_intrinsic,size,new_size)

    # right camera’s pose in the left camera’s frame
    camera_rotation = np.array(calib.data['R']['data']).reshape(3, 3)
    camera_translation = np.array(calib.data['T']['data']).reshape(3, 1).flatten()

    # Calculate rectification matrices
    R1, R2, P1, P2, Q = calib.calculate_rectification_matrices(
        scaled_left_intrinsic, left_distortion,
        scaled_right_intrinsic, right_distortion,
        camera_rotation, camera_translation,
        new_size
    )

    # ros camera_info에 맞게 yaml 파일로 저장
    calib.save_camera_info_yaml("left.yaml", "left_camera", new_size[0], new_size[1], scaled_left_intrinsic, left_distortion, R1, P1)
    calib.save_camera_info_yaml("right.yaml", "right_camera", new_size[0], new_size[1], scaled_right_intrinsic, right_distortion, R2, P2)

    calib.print_tf((camera_rotation.T, -camera_rotation.T @ camera_translation))

if __name__ == '__main__':
    main()