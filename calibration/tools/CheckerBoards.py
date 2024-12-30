# Author: Taehun Ryu
# Last Updated: 2024-12-30
# Purpose: CharucoBoard list in 3D Vision Robotics Lab

import numpy as np
import cv2

class CharucoBoard_6_9_0_26:
  def __init__(self):
    board_marker_ids = np.array(range(0, 27))
    self.board_name = "CharucoBoard_6_9_0_26"
    self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    self.board = cv2.aruco.CharucoBoard(
        (6, 9), 0.083, 0.062, \
          self.aruco_dict, board_marker_ids
    )
  def showCharucoBoardImage(self):
    """
    Charuco 보드 이미지 생성
    """
    board_image = self.board.generateImage((1000, 1000), None, 0, 1)
    cv2.imshow(self.board_name, board_image)

    return board_image

class CharucoBoard_6_9_27_53:
  def __init__(self):
    board_marker_ids = np.array(range(27, 54))
    self.board_name = "CharucoBoard_6_9_27_53"
    self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    self.board = cv2.aruco.CharucoBoard(
        (6, 9), 0.083, 0.062, \
          self.aruco_dict, board_marker_ids
    )
  def showCharucoBoardImage(self):
    """
    Charuco 보드 이미지 생성
    """
    board_image = self.board.generateImage((1000, 1000), None, 0, 1)
    cv2.imshow(self.board_name, board_image)

    return board_image

class CharucoBoard_5_5_0_11:
  def __init__(self):
    board_marker_ids = np.array(range(0, 12))
    self.board_name = "CharucoBoard_5_5_0_11"
    self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    self.board = cv2.aruco.CharucoBoard(
        (5, 5), 0.098, 0.073, \
          self.aruco_dict, board_marker_ids
    )
  def showCharucoBoardImage(self):
    """
    Charuco 보드 이미지 생성
    """
    board_image = self.board.generateImage((1000, 1000), None, 0, 1)
    cv2.imshow(self.board_name, board_image)

    return board_image

if __name__ == "__main__":
  board1 = CharucoBoard_6_9_0_26()
  board2 = CharucoBoard_6_9_27_53()
  board3 = CharucoBoard_5_5_0_11()
  board1.showCharucoBoardImage()
  board2.showCharucoBoardImage()
  board3.showCharucoBoardImage()
  cv2.waitKey(0)
  cv2.destroyAllWindows()
