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
    T_01 = np.array([[9.9990284967847165e-01, -1.3864324806504688e-02,1.4393410060100941e-03, -1.5839181914932663e+01],
                    [1.3891351905335098e-02, 9.9968575565847340e-01, -2.0866726523086448e-02, 1.2441969879841257e+00],
                    [-1.1495856270788637e-03, 2.0884693706321750e-02, 9.9978123008070074e-01, 1.6373402041540047e+01],
                    [0., 0., 0., 1. ]])

    # Camera intrinsic matrix
    K = np.array([[1.2983790035712402e+03, 0    , 2.3594789676956711e+02],
                [0, 1.2862918340343197e+03, 2.0875950583665767e+02], 
                [0, 0, 1]])

    # Calculate matrix P
    calculate_projection_matrix(T_01, K, 'cm')
    
if __name__ == '__main__':
    main()