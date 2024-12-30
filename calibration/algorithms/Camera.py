import cv2
import glob
import os

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # python_utils/

class Camera:
  def __init__(self, img_dir, aruco_dict, charuco_board):
    # Image
    self.img_dir = img_dir
    self.img_shape = None
    # Board
    self.aruco_dict = aruco_dict
    self.charuco_board = charuco_board
    self.all_corners = []
    self.all_ids = []
    # Frames
    self.frames_ = None
    # Results(Camera information)
    self.camera_matrix_ = None  # 내부 파라미터 행렬
    self.distortion_ = None  # 왜곡 계수

  @property
  def camera_matrix(self):
    return self.camera_matrix_
  @property
  def distortion(self):
    return self.distortion_

  @camera_matrix.setter
  def camera_matrix(self, intrinsic):
    self.camera_matrix_ = intrinsic
  @distortion.setter
  def distortion(self, distortion):
    self.distortion_ = distortion

  def detectCharucoCornersAndIds(self, image):
    """
    이미지에서 Charuco 보드를 감지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict)
    state : str = ""

    if ids is not None and len(ids) > 0:
      _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
          corners, ids, gray, self.charuco_board
      )
      if charuco_ids is not None and len(charuco_ids) > 0:
        return "true", charuco_corners, charuco_ids
      else:
        return "non_corner", None, None
    else:
      return "non_marker", None, None

  def initFrame(self):
    """
    1) 해당 프레임에서의 이미지, 코너, 아이디를 저장
    2) 각 프레임 별로 정보를 관리하기 위해 frames_에 딕셔너리로 저장
    """
    image_paths = sorted(glob.glob(self.img_dir + "/*.png"))
    frames = []
    for img_path in image_paths:
      img = cv2.imread(img_path)
      if self.img_shape is None:
        self.img_shape = img.shape[:2][::-1]
      str, corners, ids = self.detectCharucoCornersAndIds(img)
      frame_info = {
        "image": img,
        "corners": corners,
        "ids": ids
      }
      if str == "non_corner":
        print(f"No charuco corners detected in {img_path}")
        continue
      elif str == "non_marker":
        print(f"No markers detected in {img_path}")
        continue
      elif str == "true":
        self.all_corners.append(corners)
        self.all_ids.append(ids)
        frames.append(frame_info)
      else:
        raise ValueError("Invalid state")
    self.frames_ = frames

  def getFrame(self, idx: int):
    """
    frames_에서 특정 인덱스에 해당하는 딕셔너리를 반환
    """
    if idx < 0 or idx >= len(self.frames_):
        raise IndexError("Invalid frame index")
    return self.frames_[idx]

  def size(self):
    return len(self.frames_)

  def initCalibration(self):
    """
    1) 모든 frame에서 감지된 코너와 아이디를 이용하여 calibration 수행
    2) 각 frames_[i]에서 Board2Cam 계산 -> non-overlapping calibration에 사용
    """
    print("Calibrating camera...")
    ret, camera_matrix, distortion, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
      charucoCorners=self.all_corners,
      charucoIds=self.all_ids,
      board=self.charuco_board,
      imageSize=self.img_shape,
      cameraMatrix=None,
      distCoeffs=None,
    )
    self.camera_matrix = camera_matrix
    self.distortion = distortion
    for i in range(len(rvecs)):
        self.frames_[i]["board2cam_r"] = rvecs[i]  # Rotation vectors(Rodrigues)
        self.frames_[i]["board2cam_t"] = tvecs[i]

  def save(self, filepath, camera_matrix, dist_coeffs):
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write("K", camera_matrix)
    fs.write("d", dist_coeffs)
    fs.release()
    print(f"Calibration saved to {filepath}")

  def run(self):
    """
    1) 이미지 디렉토리에서 이미지를 읽어들여 calibration 수행
    2) calibration 결과를 파일로 저장
    """
    self.initFrame()
    self.initCalibration()
    print("Calibration completed")
    self.save(f"{project_root}/calibration/results/mono.yaml", self.camera_matrix, self.distortion)
