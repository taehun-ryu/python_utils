import cv2
from Camera import Camera
import geometrytools as gts
import numpy as np
import itertools

class NonOverlapCalib:
  def __init__(self, camera_1: Camera, camera_2: Camera):
    self.camera_1 = camera_1
    self.camera_2 = camera_2

  def checkConsistency(self):
    """
    두 카메라의 이미지 수, 코너 수, 아이디 수가 일치하는지 확인
    """
    if len(self.camera_1.all_corners) != len(self.camera_2.all_corners):
      raise ValueError("Number of frames captured by the two cameras do not match.")
    if len(self.camera_1.all_ids) != len(self.camera_2.all_ids):
      raise ValueError("Number of frames with detected corners do not match.")
    if len(self.camera_1.all_corners) != len(self.camera_1.all_ids):
      raise ValueError("Number of frames with detected corners and IDs do not match.")
    if len(self.camera_2.all_corners) != len(self.camera_2.all_ids):
      raise ValueError("Number of frames with detected corners and IDs do not match.")

  def getTranslationForClustering(self, pose_abs_1, pose_abs_2):
    data = []
    num_poses = min(len(pose_abs_1), len(pose_abs_2))
    for i in range(num_poses):
      rot_1, trans_1 = gts.Proj2RT(pose_abs_1[i])
      rot_2, trans_2 = gts.Proj2RT(pose_abs_2[i])
      concat_trans_1_2 = []
      concat_trans_1_2 = cv2.hconcat([np.transpose(trans_1), np.transpose(trans_2)])
      data.append(concat_trans_1_2)
      position_1_2 = np.array([p.flatten() for p in data], dtype=np.float32)
    return position_1_2

  def clusterTranslation(self, position_1_2, num_clusters):
    """k-means clustering을 통해 translation을 클러스터링

    Args:
        position_1_2 (vector<t>): translation between two frames
        num_clusters (int): number of clusters

    Returns:
        matrix: cluster labels
    """
    data = np.array([p.flatten() for p in position_1_2], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    nb_kmean_iteration = 5
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, _ = cv2.kmeans(data, num_clusters, None, criteria, nb_kmean_iteration, flags)
    return labels

  def selectClusters(self, num_clusters, nb_clust_pick):
    #TODO: implement this function
    pass

  def selectPoses(self, selected_cluster_idxs, cluster_labels):
    #TODO: implement this function
    pass

  def preparePosesForHandEyeCalibration(self, pose_abs_1, pose_abs_2, selected_poses_idxs):
    #TODO: implement this function
    pass

  def getPosesForHandEyeCalibration(self, pose_abs_1, pose_abs_2, cluster_labels, num_clusters, nb_clust_pick):
    selected_cluster_idxs = self.selectClusters(num_clusters, nb_clust_pick)
    selected_poses_idxs = self.selectPoses(selected_cluster_idxs, cluster_labels)
    r_cam_1, t_cam_1, r_cam_2, t_cam_2 = \
      self.preparePosesForHandEyeCalibration(pose_abs_1, pose_abs_2, selected_poses_idxs)

  def handeyeBootstraptTranslationCalibration(self, nb_cluster, nb_it, pose_abs_1, pose_abs_2):
    """hand-eye calibration을 통해 두 카메라의 상대 위치를 추정

    Args:
        nb_cluster (unsigned int): cluster 수
        nb_it (unsigned int): hand-eye calibration 반복 횟수
        pose_abs_1 (vector<T>): 첫 번째 카메라의 frame간 포즈
        pose_abs_2 (vector<T>): 두 번째 카메라의 frame간 포즈
    """
    print("Bootstrapt handeye calibration 실행(MC-Calib)")

    position_1_2 = self.getTranslationForClustering(pose_abs_1, pose_abs_2)
    num_poses = len(position_1_2)

    num_clusters = min(nb_cluster, num_poses)
    cluster_labels = self.clusterTranslation(position_1_2, num_clusters)

    nb_cluster_pick = 6
    nb_success = 0
    for i in range(nb_it):
      selected_poses_idxs, r_cam_1, t_cam_1, r_cam_2, t_cam_2 = self.getPosesForHandEyeCalibration(
        pose_abs_1, pose_abs_2, cluster_labels, num_clusters, nb_cluster_pick
      )

      # Hand-eye calibration
      # THINK: 이 부분은 어떻게 구현해야 할까..?

  def getPoseBetweenFrames(self, camera: Camera, i, j):
    """
    board2frame1과 board2frame2를 통해 frame1과 frame2 사이의 변환 행렬을 계산
    """
    frame_1 = camera.getFrame(i)
    frame_2 = camera.getFrame(j)
    b2f_1_r = frame_1["board2cam_r"]
    b2f_1_t = frame_1["board2cam_t"]
    b2f_2_r = frame_2["board2cam_r"]
    b2f_2_t = frame_2["board2cam_t"]
    proj_1 = gts.RVecT2Proj(b2f_1_r, b2f_1_t) # board to frame1
    proj_2 = gts.RVecT2Proj(b2f_2_r, b2f_2_t) # board to frame2
    proj_1_inv = np.linalg.inv(proj_1) # frame1 to board
    f2f = np.dot(proj_1_inv, proj_2) # frame1 to frame2

    return f2f

  def initNonOverlapPair(self, camera_1: Camera, camera_2: Camera):
    """
    두 카메라의 이미지 쌍을 초기화
    """
    pose_abs_1 = []
    pose_abs_2 = []
    if camera_1.size() != camera_2.size():
      raise ValueError("Number of frames captured by the two cameras do not match.")
    N = camera_1.size() # == frame 수

    # THINK frame 사이의 변환이 적으면 무시하는게 필요할거 같은데 전체 iter 안하고 어떻게 하지?
    combinations = list(itertools.combinations(range(N), 2))
    for i, j in combinations:
      pose_1 = self.getPoseBetweenFrames(camera_1, i, j)
      pose_2 = self.getPoseBetweenFrames(camera_2, i, j)
      pose_abs_1.append(pose_1)
      pose_abs_2.append(pose_2)

    return pose_abs_1, pose_abs_2

  def run(self):
    self.checkConsistency()
    pose_abs_1, pose_abs_2 = self.initNonOverlapPair(self.camera_1, self.camera_2)
    nb_cluster = 10
    nb_it = 200
    self.handeyeBootstraptTranslationCalibration(nb_cluster, nb_it, pose_abs_1, pose_abs_2)

if __name__ == "__main__":
  # ArUco 및 Charuco 보드 설정
  aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
  number_x_square = 5
  number_y_square = 5
  length_square = 0.098  # 사각형 크기 (미터 단위)
  length_marker = 0.073  # 마커 크기 (미터 단위)

  charuco_board = cv2.aruco.CharucoBoard(
      (number_x_square, number_y_square), length_square, length_marker, aruco_dict
  )
  # 이미지 디렉토리
  img_dir_1 = "/home/user/calib_data/Cam_001"
  img_dir_2 = "/home/user/calib_data/Cam_002"
  # 카메라 객체 생성 및 초기화
  camera_1 = Camera(img_dir_1, aruco_dict, charuco_board)
  camera_2 = Camera(img_dir_2, aruco_dict, charuco_board)
  camera_1.initFrame()
  camera_2.initFrame()
  camera_1.initCalibration()
  camera_2.initCalibration()

  # 스테레오 캘리브레이션 실행
  non_overlap_calib = NonOverlapCalib(camera_1, camera_2)
  non_overlap_calib.run()
