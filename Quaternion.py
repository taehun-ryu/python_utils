import numpy as np

def axisAngleToQuaternion(axis:np.ndarray, angle:float) -> np.ndarray:
    """Creates a quaternion from the given rotation parameters (axis and angle in radians).

    Parameters
    ----------
    axis : [3x1] np.ndarray
            rotation axis
    angle : float
            rotation angle in radians

    Returns
    -------
    q : [4x1] np.ndarra
        quaternion defining the orientation
    """
    if isinstance(axis, list) and len(axis)==3:
        axis = np.array(axis)
    elif isinstance(axis, np.ndarray) and axis.size==3:
        pass
    else:
        raise TypeError("The axis of rotation must be given as [3x1] np.ndarray vector or a python list of 3 elements")

    if angle == 0:
        q = np.array([1,0,0,0])  #  identity quaternion
    else:
        if np.linalg.norm(axis) == 0:
            raise Exception("A valid rotation 'axis' parameter must be provided to describe a meaningful rotation.")
        else:
            u = axis/np.linalg.norm(axis)
            q = np.append(np.cos(angle/2), np.sin(angle/2)*u)

    return q


def quaternion_to_axis_angle(q:np.ndarray) -> list:
    """Compute rotation parameters (axis and angle in radians) from a quaternion q defining a given orientation.

    Parameters
    ----------
    q : [4x1] np.ndarray
        quaternion defining a given orientation

    Returns
    -------
    axis : [3x1] np.ndarray, when undefined=[0. 0. 0.]
    angle : float
    """
    if isinstance(q, list) and len(q)==4:
        e0 = np.array(q[0])
        e = np.array(q[1:])
    elif isinstance(q, np.ndarray) and q.size==4:
        e0 = q[0]
        e = q[1:]
    else:
        raise TypeError("The quaternion \"q\" must be given as [4x1] np.ndarray quaternion or a python list of 4 elements")

    if np.linalg.norm(e) == 0:
        axis = np.array([1,0,0]) #To be checked again
        angle = 0
    elif np.linalg.norm(e) != 0:
        axis = e/np.linalg.norm(e) 
        if e0 == 0:
            angle = np.pi
        else:
            angle = 2*np.arctan(np.linalg.norm(e)/e0) 

    return axis, angle


def quaternion_to_rotation_matrix(q:np.ndarray) -> np.ndarray:
    """Computes the rotation matrix given a quaternion q defining an orientation.

    Parameters
    ----------
    q : [4x1] np.ndarray
        quaternion defining a given orientation.

    Returns
    -------
    rotation_matrix : [3x3] np.ndarray
    """
    if isinstance(q, list) and len(q)==4:
        q = np.array(q) 
    elif isinstance(q, np.ndarray) and q.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    e0 = q[0]
    e1 = q[1]
    e2 = q[2]
    e3 = q[3]

    rotation_matrix = np.array([[e0*e0 + e1*e1 - e2*e2 - e3*e3, 2*e1*e2 - 2*e0*e3, 2*e0*e2 + 2*e1*e3],
                                [2*e0*e3 + 2*e1*e2, e0*e0 - e1*e1 + e2*e2 - e3*e3, 2*e2*e3 - 2*e0*e1],
                                [2*e1*e3 - 2*e0*e2, 2*e0*e1 + 2*e2*e3, e0*e0 - e1*e1 - e2*e2 + e3*e3]])        

    return rotation_matrix

def rotation_matrix_to_quaternion(R:np.ndarray) -> np.ndarray:
    """Creates a quaternion from a rotation matrix defining a given orientation.

    Parameters
    ----------
    R : [3x3] np.ndarray
        Rotation matrix

    Returns
    -------
    q : [4x1] np.ndarray
        quaternion defining the orientation
    """
    u_q0 = np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])/4) # the prefix u_ means unsigned
    u_q1 = np.sqrt((1 + R[0,0] - R[1,1] - R[2,2])/4)
    u_q2 = np.sqrt((1 - R[0,0] + R[1,1] - R[2,2])/4)
    u_q3 = np.sqrt((1 - R[0,0] - R[1,1] + R[2,2])/4)

    q = np.array([u_q0, u_q1, u_q2, u_q3])

    if u_q0 == max(q):
        q0 = u_q0
        q1 = (R[2,1] - R[1,2])/(4*q0)
        q2 = (R[0,2] - R[2,0])/(4*q0)
        q3 = (R[1,0] - R[0,1])/(4*q0)

    if u_q1 == max(q):
        q1 = u_q1
        q0 = (R[2,1] - R[1,2])/(4*q1)
        q2 = (R[0,1] + R[1,0])/(4*q1)
        q3 = (R[0,2] + R[2,0])/(4*q1)

    if u_q2 == max(q):
        q2 = u_q2
        q0 = (R[0,2] - R[2,0])/(4*q2)
        q1 = (R[0,1] + R[1,0])/(4*q2)
        q3 = (R[1,2] + R[2,1])/(4*q2)

    if u_q3 == max(q):
        q3 = u_q3
        q0 = (R[1,0] - R[0,1])/(4*q3)
        q1 = (R[0,2] + R[2,0])/(4*q3)
        q2 = (R[1,2] + R[2,1])/(4*q3)

    q = np.array([q0, q1, q2, q3])
    return q

def euler_to_quaternion(euler_angles:np.ndarray|list)->np.ndarray:
    """
    Convert an Euler angle to a quaternion.

    We have used the following definition of Euler angles.

    - Tait-Bryan variant of Euler Angles
    - Yaw-pitch-roll rotation order (ZYX convention), rotating around the z, y and x axes respectively
    - Intrinsic rotation (the axes move with each rotation)
    - Active (otherwise known as alibi) rotation (the point is rotated, not the coordinate system)
    - Right-handed coordinate system with right-handed rotations

    Parameters
    ----------
    euler_angles :
        [3x1] np.ndarray
        [roll, pitch, yaw] angles in radians

    Returns
    -------
    q : [4x1] np.ndarray
        quaternion defining a given orientation
  """
    if isinstance(euler_angles, list) and len(euler_angles)==3:
        euler_angles = np.array(euler_angles) 
    elif isinstance(euler_angles, np.ndarray) and euler_angles.size==3:
        pass
    else:
        raise TypeError("The euler_angles must be given as [3x1] np.ndarray vector or a python list of 3 elements")

    roll = euler_angles[0]
    pitch = euler_angles[1]
    yaw = euler_angles[2]

    q0 = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q1 = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    q2 = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    q3 = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)

    q = np.r_[q0, q1, q2, q3]

    return q

def quaternion_to_euler(q:np.ndarray|list)->np.ndarray:
    """
    Convert a quaternion into euler angles [roll, pitch, yaw]
    - roll is rotation around x in radians (CCW)
    - pitch is rotation around y in radians (CCW)
    - yaw is rotation around z in radians (CCW)

    Parameters
    ----------
    q : [4x1] np.ndarray
        quaternion defining a given orientation

    Returns
    -------
    euler_angles :
        [3x1] np.ndarray
        [roll, pitch, yaw] angles in radians
    """
    if isinstance(q, list) and len(q)==4:
        q = np.array(q)
    elif isinstance(q, np.ndarray) and q.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    t2 = 2.0*(q0*q2 - q1*q3)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    if t2 == 1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = -np.arctan2(q0, q1)
    elif t2 == -1:
        pitch = np.arcsin(t2)
        roll = 0
        yaw = +np.arctan2(q0, q1)
    else:
        pitch = np.arcsin(t2)
        roll = np.arctan2(2.0*(q0*q1 + q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3)
        yaw = np.arctan2(2.0*(q0*q3 + q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3)

    euler_angles = np.r_[roll, pitch, yaw]

    return  euler_angles
