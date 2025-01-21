import math
import matplotlib.pyplot as plt
from itertools import combinations
import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # python_utils/
sys.path.append(project_root)

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to load poses from result.txt
def load_poses(file_path):
    poses = []
    try:
        with open(file_path, "r") as file:
            for line in file:
                x, y, yaw = map(float, line.strip().split())
                poses.append((x, y, yaw))
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    return poses

# Main function
def main(file_path, threshold=0.15):
    """
    Validate distances between poses and visualize the results.

    Args:
        file_path (str): Path to the result file.
        threshold (float): Maximum allowable distance in meters.

    Returns:
        bool: True if all distances are within the threshold, False otherwise.
    """
    # Load poses from file
    pose_points = load_poses(file_path)
    if not pose_points:
        print("No pose points to visualize.")
        return False

    # Extract (x, y) for distance calculation
    xy_points = [(x, y) for x, y, _ in pose_points]

    # Calculate distances between all pairs
    pose_distances = {}
    for i, j in combinations(range(len(xy_points)), 2):
        distance = calculate_distance(xy_points[i], xy_points[j])
        pose_distances[f"{i+1}-{j+1}"] = distance

    # Check if all distances are within the threshold
    all_within_threshold = all(distance <= threshold for distance in pose_distances.values())

    # Visualization
    plt.figure(figsize=(6, 6))
    for i, (x, y, yaw) in enumerate(pose_points):
        plt.scatter(x, y, label=f"Pose {i+1} (Yaw: {yaw:.1f}°)", s=100)

    # Draw connections and annotate distances
    for (i, j), distance in zip(combinations(range(len(xy_points)), 2), pose_distances.values()):
        x_values = [xy_points[i][0], xy_points[j][0]]
        y_values = [xy_points[i][1], xy_points[j][1]]
        plt.plot(x_values, y_values, 'k--', alpha=0.6)

        # Calculate label position with slight offset
        mid_x = (x_values[0] + x_values[1]) / 2
        mid_y = (y_values[0] + y_values[1]) / 2
        offset_x = (x_values[1] - x_values[0]) * 0.1
        offset_y = (y_values[1] - y_values[0]) * 0.1
        plt.text(mid_x + offset_x, mid_y + offset_y, f"{distance:.3f}m", fontsize=10, color='blue')

    # Add result status
    status_text = "SUCCESS" if all_within_threshold else "FAIL"
    plt.title(f"Pose Distance Validation: {status_text}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

    return all_within_threshold

# Entry point for the script
if __name__ == "__main__":
    # Path to the result file
    file_path = f"{project_root}/pose/results/pose_result.txt"
    threshold = 0.15  # 15 cm
    result = main(file_path, threshold)
