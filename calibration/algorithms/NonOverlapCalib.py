# Author: Taehun Ryu
# This code is based on th following paper:
# "MC-Calib: A generic and robust calibration toolbox for multi-camera systems"
# Only the non-overlapping calibration part is implemented and it is specially focused on the stereo camera.
# If you want to use a different setup, such as having 4 non-overlapping cameras, you must implement new code, leveraging this code.

import cv2
import numpy as np
import itertools
import random
from typing import List

# Set project root path
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/
sys.path.append(project_root)
# Import custom modules in other directories
from calibration.tools import geometrytools as gts

class NonOverlapCalib:
  def __init__(self, camera_1, camera_2):
    self.camera_1 = camera_1
    self.camera_2 = camera_2
    # results
    self.R = None
    self.T = None

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
    """비슷한 translation을 가진 pose들을 클러스터링 => num_clusters만큼의 클러스터 생성

    Args:
        position_1_2 (vector<t>): translation between two frames
        num_clusters (int): number of clusters

    Returns:
        clusters labels(np.ndarray): cluster index for each translation
    """
    data = np.array([p.flatten() for p in position_1_2], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    nb_kmean_iteration = 5
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, _ = cv2.kmeans(data, num_clusters, None, criteria, nb_kmean_iteration, flags)

    return labels

  def selectClusters(self, num_clusters: int, nb_clust_pick: int) -> List[int]:
    """
    num_clusters 중에서 nb_clust_pick만큼의 클러스터를 랜덤하게 선택
    """
    # pick from n clusters randomly
    shuffled_ind = list(range(num_clusters))
    random.shuffle(shuffled_ind)

    cluster_select = shuffled_ind[:nb_clust_pick]
    assert len(cluster_select) == nb_clust_pick

    return cluster_select

  def selectPoses(self, selected_cluster_idxs: List[int], clusters_labels: np.ndarray) -> List[int]:
    """
    선택된 cluster index에 해당하는 pose 중 하나를 선택
    """
    # Select one pose index from each cluster
    selected_poses_idxs = []

    for cluster_idx in selected_cluster_idxs:
        pose_idxs = [pose_idx for pose_idx in range(clusters_labels.shape[0])
                      if clusters_labels[pose_idx] == cluster_idx]

        if not pose_idxs:
          raise ValueError(f"No poses found for cluster index {cluster_idx}")

        random_idx = random.choice(pose_idxs)
        selected_poses_idxs.append(random_idx)

    assert len(selected_poses_idxs) == len(selected_cluster_idxs)

    return selected_poses_idxs

  def preparePosesForHandEyeCalibration(self, pose_abs_1, pose_abs_2, selected_poses_idxs):
    """선택된 index에 해당하는 pose들을 hand-eye calibration에 사용할 수 있도록 변환

    Args:
        pose_abs_1: camera 1 pose (inverse)
        pose_abs_2: camera 2 pose
        selected_poses_idxs (list): selected poses indices

    Returns:
        rotation is represented as a 3x1 matrix
        translation is represented as a 3x1 matrix
    """
    r_cam_1_select = []
    t_cam_1_select = []
    r_cam_2_select = []
    t_cam_2_select = []
    for i in selected_poses_idxs:
      pose_cam_1 = np.linalg.inv(pose_abs_1[i])
      pose_cam_2 = pose_abs_2[i]
      r_cam_1, t_cam_1 = gts.Proj2RT(pose_cam_1)
      r_cam_2, t_cam_2 = gts.Proj2RT(pose_cam_2)
      r_1_vec = cv2.Rodrigues(r_cam_1)[0]
      r_2_vec = cv2.Rodrigues(r_cam_2)[0]
      r_cam_1_select.append(r_1_vec)
      t_cam_1_select.append(t_cam_1)
      r_cam_2_select.append(r_2_vec)
      t_cam_2_select.append(t_cam_2)

    return r_cam_1_select, t_cam_1_select, r_cam_2_select, t_cam_2_select

  def getPosesForHandEyeCalibration(self, pose_abs_1, pose_abs_2, cluster_labels, num_clusters, nb_clust_pick):
    selected_cluster_idxs = self.selectClusters(num_clusters, nb_clust_pick)

    selected_poses_idxs = self.selectPoses(selected_cluster_idxs, cluster_labels)

    r_cam_1, t_cam_1, r_cam_2, t_cam_2 = \
      self.preparePosesForHandEyeCalibration(pose_abs_1, pose_abs_2, selected_poses_idxs)

    return selected_poses_idxs, r_cam_1, t_cam_1, r_cam_2, t_cam_2

  def checkSetConsistency(self, pose_abs_1, pose_abs_2, selected_pose_idxs, pose_c1_c2):
    max_rotation_error = 0
    for i in range(len(selected_pose_idxs)):
      pose_cam_1_1 = pose_abs_1[selected_pose_idxs[i]]
      pose_cam_2_1 = pose_abs_2[selected_pose_idxs[i]]

      for j in range(len(selected_pose_idxs)):
        if i == j:
          continue

        pose_cam_1_2 = pose_abs_1[selected_pose_idxs[i]]
        pose_cam_2_2 = pose_abs_2[selected_pose_idxs[i]]

        # 각 카메라에서 두 프레임 사이의 상대 변환 계산
        PP1 = np.linalg.inv(pose_cam_1_2) @ pose_cam_1_1
        PP2 = np.linalg.inv(pose_cam_2_1) @ pose_cam_2_2

        # 오차행렬 계산
        ErrMat = np.linalg.inv(PP2) @ pose_c1_c2 @ PP1 @ pose_c1_c2
        ErrRot, ErrTrans = gts.Proj2RT(ErrMat)
        ErrRotMat = cv2.Rodrigues(ErrRot)[0]

        # tr(R_err) = 1 + 2cos(theta_err) 이므로, theta_err를 구하기 위해 trace 계산
        traceRot = np.trace(ErrRotMat) - np.finfo(float).eps
        err_degree = np.arccos(0.5 * (traceRot - 1.0)) * 180.0 / np.pi
        max_rotation_error = max(max_rotation_error, err_degree)

    return max_rotation_error

  def handeyeBootstraptTranslationCalibration(self, nb_cluster, nb_it, pose_abs_1, pose_abs_2):
    """hand-eye calibration을 통해 두 카메라의 상대 위치를 추정

    Args:
        nb_cluster (unsigned int): cluster 수
        nb_it (unsigned int): hand-eye calibration 반복 횟수
        pose_abs_1 (vector<T>): 첫 번째 카메라의 frame간 포즈
        pose_abs_2 (vector<T>): 두 번째 카메라의 frame간 포즈
    """
    print("Bootstrapt handeye calibration 실행...")

    position_1_2 = self.getTranslationForClustering(pose_abs_1, pose_abs_2)
    num_poses = len(position_1_2)

    num_clusters = min(nb_cluster, num_poses)
    cluster_labels = self.clusterTranslation(position_1_2, num_clusters)

    r1_he, r2_he, r3_he = [], [], []
    t1_he, t2_he, t3_he = [], [], []
    nb_cluster_pick = 6
    nb_success = 0
    for i in range(nb_it):
      selected_poses_idxs, r_cam_1, t_cam_1, r_cam_2, t_cam_2 = self.getPosesForHandEyeCalibration(
        pose_abs_1, pose_abs_2, cluster_labels, num_clusters, nb_cluster_pick
      )

      # Hand-eye calibration
      R_c1_c2, t_c1_c2 = cv2.calibrateHandEye(r_cam_1, t_cam_1, r_cam_2, t_cam_2, method=cv2.CALIB_HAND_EYE_TSAI)
      pose_c1_c2 = gts.RT2Proj(R_c1_c2, t_c1_c2)

      max_rotation_error = self.checkSetConsistency(pose_abs_1, pose_abs_2, selected_poses_idxs, pose_c1_c2)

      if max_rotation_error < 15:
        nb_success += 1
        rot_temp, trans_temp = gts.Proj2RT(pose_c1_c2)
        r1_he.append(rot_temp[0])
        r2_he.append(rot_temp[1])
        r3_he.append(rot_temp[2])
        t1_he.append(trans_temp[0])
        t2_he.append(trans_temp[1])
        t3_he.append(trans_temp[2])

    if nb_success > 3:
      r_he, t_he = np.zeros((3,1)), np.zeros((3,1))
      r_he[0] = np.median(r1_he)
      r_he[1] = np.median(r2_he)
      r_he[2] = np.median(r3_he)
      t_he[0] = np.median(t1_he)
      t_he[1] = np.median(t2_he)
      t_he[2] = np.median(t3_he)
      pose_c1_c2 = gts.RT2Proj(r_he, t_he)

      return pose_c1_c2
    else:
      print("[WARN] Run the normal handeye calibration on all the samples")
      pose_c1_c2 = self.handeyeNormalCalibration(pose_abs_1, pose_abs_2)
      return pose_c1_c2

  def handeyeNormalCalibration(self, pose_abs_1, pose_abs_2):
    print("Normal handeye calibration 실행...")
    num_poses = min(len(pose_abs_1), len(pose_abs_2)) # It should be the same
    r_cam_1, r_cam_2, t_cam_1, t_cam_2 = [], [], [], []
    for i in range(num_poses):
      pose_cam_1 = np.linalg.inv(pose_abs_1[i])
      pose_cam_2 = pose_abs_2[i]
      r_1, t_1 = gts.Proj2RT(pose_cam_1)
      r_2, t_2 = gts.Proj2RT(pose_cam_2)
      r_1_vec = cv2.Rodrigues(r_1)[0]
      r_2_vec = cv2.Rodrigues(r_2)[0]
      r_cam_1.append(r_1_vec)
      t_cam_1.append(t_1)
      r_cam_2.append(r_2_vec)
      t_cam_2.append(t_2)

    # Hand-eye calibration
    R_c1_c2, t_c1_c2 = cv2.calibrateHandEye(r_cam_1, t_cam_1, r_cam_2, t_cam_2, method=cv2.CALIB_HAND_EYE_TSAI)
    pose_c1_c2 = gts.RT2Proj(R_c1_c2, t_c1_c2)

    return pose_c1_c2

  def getPoseBetweenFrames(self, camera, i, j):
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

  def initNonOverlapPair(self, camera_1, camera_2):
    """
    두 카메라의 이미지 쌍을 초기화
    """
    pose_abs_1 = []
    pose_abs_2 = []
    if camera_1.size() != camera_2.size():
      raise ValueError("Number of frames captured by the two cameras do not match.")
    N = camera_1.size() # == frame 수

    combinations = list(itertools.combinations(range(N), 2))
    for i, j in combinations:
      pose_1 = self.getPoseBetweenFrames(camera_1, i, j)
      pose_2 = self.getPoseBetweenFrames(camera_2, i, j)
      pose_abs_1.append(pose_1)
      pose_abs_2.append(pose_2)

    return pose_abs_1, pose_abs_2

  def save(self, filepath, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, R, T, ):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write("K_1", camera_matrix_1)
    fs.write("d_1", dist_coeffs_1)
    fs.write("K_2", camera_matrix_2)
    fs.write("d_2", dist_coeffs_2)
    fs.write("Rotation", R)
    fs.write("Translation", T)
    fs.release()
    print(f"Non-overlapping calibrations saved to {filepath}")

  def run(self):
    self.checkConsistency()
    pose_abs_1, pose_abs_2 = self.initNonOverlapPair(self.camera_1, self.camera_2)
    nb_cluster = 10
    nb_it = 200
    pose_c1_c2 = self.handeyeBootstraptTranslationCalibration(nb_cluster, nb_it, pose_abs_1, pose_abs_2)
    self.R, self.T = gts.Proj2RT(pose_c1_c2)
    print("Non-overlapping calibration complete")
    self.save(f"{project_root}/calibration/results/non_overlap.yaml", \
        self.camera_1.camera_matrix, self.camera_1.distortion, \
        self.camera_2.camera_matrix, self.camera_2.distortion, \
        self.R, self.T)
