import cv2
import numpy as np

def RVecT2Proj(RVec, t):
    """
    3x1 RVec와 3x1 tVec를 4x4 투영 행렬로 변환
    """
    R, _ = cv2.Rodrigues(RVec)
    proj = np.hstack((R, t))
    proj = np.vstack((proj, np.array([0, 0, 0, 1])))
    return proj

def RT2Proj(R, t):
    """
    3x3 R과 3x1 t를 4x4 투영 행렬로 변환
    """
    proj = np.hstack((R, t))
    proj = np.vstack((proj, np.array([0, 0, 0, 1])))
    return proj

def Proj2RVecT(proj):
    """
    4x4 투영 행렬을 3x1 RVec와 3x1 tVec로 변환
    """
    R = proj[:3, :3]
    t = proj[:3, 3:]
    RVec, _ = cv2.Rodrigues(R)
    return RVec, t

def Proj2RT(proj):
    """
    4x4 투영 행렬을 3x3 R과 3x1 t로 변환
    """
    R = proj[:3, :3]
    t = proj[:3, 3:]
    return R, t
