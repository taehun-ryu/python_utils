import numpy as np
import Quaternion as Q

def calculate_projection_matrix(T_01, K, tf_type="m"):
    if tf_type == 'm':
        scale_factor = 1
    elif tf_type == 'cm':
        scale_factor = 0.01
    else:
        raise Exception("Type should be [m] or [cm].")

    # Extract rotation and translation
    R_10 = T_01[:3, :3].T  # Transpose for inverse rotation
    t_10 = np.matmul(-R_10, T_01[:3, 3] * scale_factor)

    # Concatenate R_10 and t_10 to form [R | t] (3x4 matrix)
    Rt = np.hstack((R_10, t_10.reshape(-1, 1)))

    print("[R | t] is \n", Rt)

    # Compute the projection matrix
    P = np.matmul(K, Rt)

    # Output the projection matrix P
    print("Projection matrix P: \n", P)


def main():
    # Transformation matrix from camera 0 to 1
    T_01 = np.array([[ 9.9834479620493777e-01, 1.8600926830503905e-02,
          -5.4421258820130693e-02, 1.4808515298595543e+01],
          [-2.0229118157711567e-02, 9.9935941244112991e-01,
          -2.9521984077646144e-02, -4.8991292005601955e-01],
          [5.3837260973072891e-02, 3.0574013252526861e-02,
          9.9808154929572346e-01, 4.2572247079801379e-01], [0., 0., 0., 1. ]])

    # Camera intrinsic matrix
    K = np.array([[366.88398515,   0.,         314.22955833],
 [  0. ,        438.65752185 ,232.42332784],
 [  0. ,          0.   ,        1.        ]])

    # Calculate matrix P
    calculate_projection_matrix(T_01, K, 'cm')
    
if __name__ == '__main__':
    main()