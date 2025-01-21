import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import threading
import os
import sys

# Import custom modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # python_utils/
sys.path.append(project_root)

class PoseRecorder(Node):
  def __init__(self):
    super().__init__('pose_recorder')
    self.result_file = f"{project_root}/pose/results/pose_result.txt"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(self.result_file), exist_ok=True)

    # Clear the result file at the start
    open(self.result_file, "w").close()

    self.subscription = self.create_subscription(
      PoseWithCovarianceStamped,
      '/CLOBOT/localization_pose',
      self.pose_callback,
      10
    )
    self.current_pose = None
    self.lock = threading.Lock()

    print("Press 's' to save pose, 'q' to quit.")

  def pose_callback(self, msg):
    with self.lock:
      self.current_pose = msg.pose.pose

  def save_pose(self):
    with self.lock:
      if self.current_pose is None:
        print("No pose received yet!")
        return

      x = self.current_pose.position.x
      y = self.current_pose.position.y
      qx = self.current_pose.orientation.x
      qy = self.current_pose.orientation.y
      qz = self.current_pose.orientation.z
      qw = self.current_pose.orientation.w

      # Convert quaternion to yaw (in degrees)
      yaw = math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

      # Save to file
      with open(self.result_file, "a") as file:
        file.write(f"{x:.3f} {y:.3f} {yaw:.3f}\n")

      print(f"Saved pose: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}°")

  def quit_node(self):
    self.destroy_node()
    rclpy.shutdown()


def main():
  rclpy.init()
  node = PoseRecorder()

  def key_listener():
    while rclpy.ok():
      key = input()
      if key == 's':
        node.save_pose()
      elif key == 'q':
        node.quit_node()
        break

  listener_thread = threading.Thread(target=key_listener, daemon=True)
  listener_thread.start()

  while rclpy.ok():
    rclpy.spin_once(node)

if __name__ == '__main__':
  main()
