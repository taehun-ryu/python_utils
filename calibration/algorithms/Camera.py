import cv2
import glob
from CheckerBoards import CharucoBoard_6_9_0_26, CharucoBoard_6_9_27_53, CharucoBoard_5_5_0_11

class Camera:
  def __init__(self, img_dir, aruco_dict, charuco_board):
    # Image
    self.img_dir = img_dir
    self.img_shape = None
    # Camera information
    self.camera_matrix_ = None  # 내부 파라미터 행렬
    self.distortion_ = None  # 왜곡 계수
    # Board
    self.aruco_dict = aruco_dict
    self.charuco_board = charuco_board
    self.all_corners = []
    self.all_ids = []
    # Frames
    self.frames_ = None

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


if __name__ == "__main__":
  # Our Lab's 5x5 Charuco Board
  board = CharucoBoard_5_5_0_11()
  # Our Lab's 6x9 Charuco Board 1
  board_1 = CharucoBoard_6_9_0_26()
  # Our Lab's 6x9 Charuco Board 2
  board_2 = CharucoBoard_6_9_27_53()

  # 이미지 디렉토리
  img_dir = "/home/user/calib_data/non_overlap/1/Cam_002"

  cam = Camera(img_dir, board_2.aruco_dict, board_2.board)
  cam.initFrame()
  cam.initCalibration()
  print(cam.camera_matrix)
  print(cam.distortion)
