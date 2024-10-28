import numpy as np

def scale_calibration(K, original_resolution, target_resolution=(640, 480)):

    # Extract original and target dimensions
    orig_width, orig_height = original_resolution
    target_width, target_height = target_resolution

    # Calculate scaling factors for width and height
    scale_x = target_width / orig_width
    scale_y = target_height / orig_height

    # Create a new intrinsic matrix with scaled parameters
    K_new = np.copy(K)

    # Scale focal lengths (fx, fy) and principal point (cx, cy) independently
    K_new[0, 0] *= scale_x  # fx
    K_new[1, 1] *= scale_y  # fy
    K_new[0, 2] *= scale_x  # cx
    K_new[1, 2] *= scale_y  # cy

    return K_new

if __name__ == "__main__":
    # Intrinsic matrix (original image)
    K = np.array([[ 1.1270147556998475e+03, 0., 9.3518845506810419e+02], 
              [0., 1.1205384002217845e+03, 5.7657969861199456e+02], 
              [0., 0., 1. ]])

    # Original resolution
    original_resolution = (1920, 1200)

    # Target resolution (640x480)
    target_resolution = (640, 480)

    # Get the scaled intrinsic matrix
    K_new = scale_calibration(K, original_resolution, target_resolution)

    print("Original Intrinsic Matrix:\n", K)
    print("Scaled Intrinsic Matrix:\n", K_new)
