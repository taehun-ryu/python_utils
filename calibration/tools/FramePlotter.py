import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FramePlotter:
    def __init__(self):
        self.axis_size = 0.1
        self.figure_title = "Camera Frames (5cm Grid)"
        self.xlim = (-0.2, 0.2)
        self.ylim = (-0.2, 0.2)
        self.zlim = (-0.2, 0.2)

        # Camera1은 항상 원점(Identity)으로 고정
        self.camera1 = (np.eye(3), np.zeros((3,1)), "Cam1")
        self.cameras = []  # Camera2 이상을 저장

    def add_camera(self, R, t, label="Cam"):
        t = np.array(t).reshape(3, 1)
        self.cameras.append((R, t, label))

    def _draw_frame(self, ax, R, t, label_prefix=""):
        origin_local = np.array([0, 0, 0]).reshape(3, 1)
        x_axis_local = np.array([self.axis_size, 0, 0]).reshape(3, 1)
        y_axis_local = np.array([0, self.axis_size, 0]).reshape(3, 1)
        z_axis_local = np.array([0, 0, self.axis_size]).reshape(3, 1)

        origin_world = R @ origin_local + t
        x_axis_world = R @ x_axis_local + t
        y_axis_world = R @ y_axis_local + t
        z_axis_world = R @ z_axis_local + t

        ox, oy, oz = origin_world.flatten()
        xx, xy, xz = x_axis_world.flatten()
        yx, yy, yz = y_axis_world.flatten()
        zx, zy, zz = z_axis_world.flatten()

        ax.plot([ox, xx], [oy, xy], [oz, xz], c='r')
        ax.plot([ox, yx], [oy, yy], [oz, yz], c='g')
        ax.plot([ox, zx], [oy, zy], [oz, zz], c='b')

        if label_prefix:
            ax.text(ox, oy, oz, f"{label_prefix} O", color='k')
        ax.text(xx, xy, xz, f"{label_prefix}X", color='r')
        ax.text(yx, yy, yz, f"{label_prefix}Y", color='g')
        ax.text(zx, zy, zz, f"{label_prefix}Z", color='b')

    def plot_frames(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.figure_title)

        # Camera1(항상 원점 고정) 먼저 그리기
        R1, t1, label1 = self.camera1
        self._draw_frame(ax, R1, t1, label_prefix=label1)

        # 이후 Camera2 이상만 추가로 그리기
        for (R, t, label) in self.cameras:
            self._draw_frame(ax, R, t, label_prefix=label)

        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        ax.set_zlim(self.zlim)

        # 5cm(=0.05m) 간격으로 tick 설정
        x_tick_min, x_tick_max = self.xlim
        y_tick_min, y_tick_max = self.ylim
        z_tick_min, z_tick_max = self.zlim
        ax.set_xticks(np.arange(x_tick_min, x_tick_max + 0.0001, 0.05))
        ax.set_yticks(np.arange(y_tick_min, y_tick_max + 0.0001, 0.05))
        ax.set_zticks(np.arange(z_tick_min, z_tick_max + 0.0001, 0.05))

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True)
        plt.show()
