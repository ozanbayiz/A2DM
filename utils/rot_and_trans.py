import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

#------------------------------------------------------------------------------------------------
# Conversion Functions - Between different rotation representations
#------------------------------------------------------------------------------------------------

def axis_angle_to_quaternion(a: torch.Tensor, eps=1e-7) -> torch.Tensor:
    """
    Convert axis-angle vectors to quaternions (w, x, y, z) in PyTorch.
    
    Args:
        a: Tensor of shape (..., 3) axis-angle. 
           The norm of a is the rotation angle theta, direction is the axis.
        eps: Small value to avoid division by zero.
           
    Returns: 
        Tensor of shape (..., 4) containing quaternions.
    """
    # Rotation angle is the norm of 'a'
    angles = a.norm(dim=-1, keepdim=True).clamp(min=eps)  # shape (..., 1)
    # Unit axis
    axis = a / angles
    half_angles = 0.5 * angles

    # w = cos(theta/2),  (x, y, z) = axis * sin(theta/2)
    w = half_angles.cos()
    xyz = axis * half_angles.sin()
    return torch.cat([w, xyz], dim=-1)  # shape (..., 4)

def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: np.ndarray of shape (3,) containing axis-angle representation
        
    Returns:
        rotation_matrix: np.ndarray of shape (3, 3) containing rotation matrix
    """
    # Use scipy's Rotation class for stable conversions
    return R.from_rotvec(axis_angle).as_matrix()

def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        matrix: np.ndarray of shape (3, 3) containing rotation matrix
        
    Returns:
        axis_angle: np.ndarray of shape (3,) containing axis-angle representation
    """
    # Use scipy's Rotation class for stable conversions
    return R.from_matrix(matrix).as_rotvec()

#------------------------------------------------------------------------------------------------
# Quaternion Operations - Basic quaternion math
#------------------------------------------------------------------------------------------------

def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a quaternion.
    
    Args:
        q: Quaternion of shape (..., 4)
        
    Returns:
        Inverse quaternion of shape (..., 4)
    """
    q_inv = q.clone()
    # Negate the vector part (x, y, z)
    q_inv[..., 1:] = -q_inv[..., 1:]
    return q_inv

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    
    Args:
        q1: First quaternion tensor [..., 4]
        q2: Second quaternion tensor [..., 4]
        
    Returns:
        Result of quaternion multiplication
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def compute_relative_quaternion(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute relative quaternion q_rel = q2 * q1^-1
    
    Args:
        q1: First quaternion tensor [..., 4]
        q2: Second quaternion tensor [..., 4]
        
    Returns:
        Relative quaternion
    """
    q1_inv = quaternion_inverse(q1)
    return quaternion_multiply(q2, q1_inv)

#------------------------------------------------------------------------------------------------
# Distance and Velocity Calculations - Computing geodesic distances and velocities
#------------------------------------------------------------------------------------------------

def quaternion_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between two quaternions.
    
    Args:
        q1: First quaternion tensor [..., 4]
        q2: Second quaternion tensor [..., 4]
        
    Returns:
        Geodesic distance (angle in radians)
    """
    # Compute dot product (or use relative quaternion's w)
    dot_product = torch.sum(q1 * q2, dim=-1)
    dot_product = torch.clamp(torch.abs(dot_product), -1.0+1e-6, 1.0-1e-6)
    
    # Compute geodesic distance (angle)
    return 2 * torch.acos(dot_product)

def geodesic_distance(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    Compute the geodesic distance (in radians) between two batches of 
    axis-angle vectors.
    
    Args:
        a1: Tensor of shape (..., 3) containing axis-angle rotations
        a2: Tensor of shape (..., 3) containing axis-angle rotations
        
    Returns:
        Tensor of shape (...) containing geodesic angles in [0, pi]
    """
    # Convert both sets of axis-angles to quaternions
    q1 = axis_angle_to_quaternion(a1)  # (..., 4)
    q2 = axis_angle_to_quaternion(a2)  # (..., 4)

    # Dot product along the quaternion dimension
    dot = (q1 * q2).sum(dim=-1).abs()  # (...)
    # Clamp to avoid floating-point issues outside [-1, 1]
    dot = dot.clamp(-1.0+1e-6, 1.0-1e-6)

    # Geodesic angle = 2 * arccos(|q1·q2|)
    angles = 2.0 * torch.acos(dot)  # (...)
    return angles

def compute_angular_velocity(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute angular velocity between two quaternions as geodesic distance.
    
    Args:
        q1: First quaternion tensor [..., 4]
        q2: Second quaternion tensor [..., 4]
        
    Returns:
        Angular velocity (angle in radians)
    """
    # Compute relative quaternion
    q_rel = compute_relative_quaternion(q1, q2)
    
    # Extract w component and compute angle
    rel_w = q_rel[..., 0]
    return 2 * torch.acos(torch.clamp(torch.abs(rel_w), -1.0+1e-6, 1.0-1e-6))

#------------------------------------------------------------------------------------------------
# Coordinate Transformations - Functions for relative/global transformations
#------------------------------------------------------------------------------------------------

def transform_relative_orientation(root_orient):
    """
    Transform absolute orientations to relative orientations.
    
    Args:
        root_orient: np.ndarray of shape (seq_len, 3) containing absolute orientations in axis-angle format
        
    Returns:
        np.ndarray of shape (seq_len, 3) containing relative orientations in axis-angle format
    """
    # Convert the entire sequence to rotation objects
    rots = R.from_rotvec(root_orient)
    
    # Initialize relative rotations
    rel_rots = np.zeros_like(root_orient)
    
    # First frame is kept as identity rotation (zeros in axis-angle)
    for i in range(1, len(root_orient)):
        # Calculate relative rotation: R_rel = R_curr * R_prev^-1
        # In scipy, this is: R_rel = R_curr * R_prev.inv()
        r_rel = rots[i] * rots[i-1].inv()
        rel_rots[i] = r_rel.as_rotvec()
    
    return rel_rots

def inverse_transform_global_orientation(rel_orient, init_root_orient):
    """
    Invert the relative orientation transformation to recover absolute orientations in SO(3).
    
    Args:
        rel_orient: Relative orientations in axis-angle format (T, 3)
        init_root_orient: Initial root orientation in axis-angle format (3,) or (1, 3)
        
    Returns:
        root_orient: Recovered absolute orientations in axis-angle format (T, 3)
    """
    # Ensure init_root_orient has the right shape
    if len(init_root_orient.shape) == 1:
        init_root_orient = init_root_orient.reshape(1, 3)
    
    # Convert to rotation objects
    rel_rots = R.from_rotvec(rel_orient)
    init_rot = R.from_rotvec(init_root_orient[0])
    
    # Initialize absolute rotations
    abs_rots = np.zeros_like(rel_orient)
    abs_rots[0] = init_root_orient[0]
    
    # Current absolute rotation
    r_abs = init_rot
    
    # Accumulate rotations
    for i in range(1, len(rel_orient)):
        # Apply relative rotation: R_abs_new = R_rel * R_abs
        r_abs = rel_rots[i] * r_abs
        abs_rots[i] = r_abs.as_rotvec()
    
    return abs_rots

def transform_relative_translation(trans):
    """
    Transform absolute translations to relative translations.
    
    Args:
        trans: np.ndarray of shape (seq_len, 3) containing absolute translations
        
    Returns:
        np.ndarray of shape (seq_len, 3) containing relative translations
    """
    # Compute frame-to-frame differences
    rel_trans = np.zeros_like(trans)
    rel_trans[1:] = np.diff(trans, axis=0)
    # First frame has zero relative translation
    return rel_trans

def inverse_transform_translation(rel_trans, init_trans):
    """
    Invert the relative translation transformation to recover absolute translations.
    
    Args:
        rel_trans: Relative translations (T, 3)
        init_trans: Initial translation (3,) or (1, 3)
        
    Returns:
        trans: Recovered absolute translations (T, 3)
    """
    # Ensure init_trans has the right shape
    if len(init_trans.shape) == 1:
        init_trans = init_trans.reshape(1, 3)
    
    # Initialize absolute translations
    trans = np.zeros_like(rel_trans)
    trans[0] = init_trans[0]
    
    # Accumulate relative translations
    for i in range(1, len(rel_trans)):
        trans[i] = trans[i-1] + rel_trans[i]
    
    return trans