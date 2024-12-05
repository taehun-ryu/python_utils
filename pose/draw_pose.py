import math
import matplotlib.pyplot as plt
from itertools import combinations

# Function to calculate Euclidean distance
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Main function
def main(pose_points, threshold=0.15):
    """
    Validate distances between poses and visualize the results.

    Args:
        pose_points (list of tuple): List of (x, y) coordinates.
        threshold (float): Maximum allowable distance in meters.

    Returns:
        bool: True if all distances are within the threshold, False otherwise.
    """
    # Calculate distances between all pairs
    pose_distances = {}
    for i, j in combinations(range(len(pose_points)), 2):
        distance = calculate_distance(pose_points[i], pose_points[j])
        pose_distances[f"{i+1}-{j+1}"] = distance

    # Check if all distances are within the threshold
    all_within_threshold = all(distance <= threshold for distance in pose_distances.values())

    # Visualization
    plt.figure(figsize=(6, 6))
    for i, point in enumerate(pose_points):
        plt.scatter(point[0], point[1], label=f"Pose {i+1}", s=100)

    # Draw connections and annotate distances
    for (i, j), distance in zip(combinations(range(len(pose_points)), 2), pose_distances.values()):
        x_values = [pose_points[i][0], pose_points[j][0]]
        y_values = [pose_points[i][1], pose_points[j][1]]
        plt.plot(x_values, y_values, 'k--', alpha=0.6)
        mid_x = (x_values[0] + x_values[1]) / 2
        mid_y = (y_values[0] + y_values[1]) / 2
        plt.text(mid_x, mid_y, f"{distance:.3f}m", fontsize=10, color='blue')

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
    # Example input
    pose_points = [
        (-0.126, 0.026),
        (-0.105, 0.016),
        (-0.005, 0.007),
        (0.005, 0.007)
    ]
    threshold = 0.15  # 15 cm
    result = main(pose_points, threshold)
