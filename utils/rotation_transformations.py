import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

#------------------------------------------------------------------------------------------------
# Conversion Functions - Between different rotation representations
#------------------------------------------------------------------------------------------------

def axis_angle_to_matrix(axis_angle):
    """
    Converts axis-angle rotations to rotation matrices.
    
    Args:
        axis_angle: Tensor of shape (B, 3) where each row is an axis-angle rotation.
                   The axis is expected to be non-zero; the norm encodes the rotation angle.
    
    Returns:
        Rotation matrices of shape (B, 3, 3).
    """
    if isinstance(axis_angle, torch.Tensor):
        # PyTorch implementation
        # Compute the rotation angle (theta) and normalize the axis.
        theta = torch.norm(axis_angle, dim=-1, keepdim=True)  # (B, 1)
        axis = axis_angle / (theta + 1e-8)                    # (B, 3)
        
        # Expand theta dimensions for trigonometric functions.
        theta = theta.unsqueeze(-1)                           # (B, 1, 1)
        cos_theta = torch.cos(theta)                          # (B, 1, 1)
        sin_theta = torch.sin(theta)                          # (B, 1, 1)
        
        # Create the skew-symmetric matrix K for each axis.
        x = axis[:, 0:1]  # (B, 1)
        y = axis[:, 1:2]  # (B, 1)
        z = axis[:, 2:3]  # (B, 1)
        zeros = torch.zeros_like(x)
        K = torch.stack([zeros, -z, y,
                         z, zeros, -x,
                         -y, x, zeros], dim=-1)  # (B, 9)
        K = K.view(-1, 3, 3)                     # (B, 3, 3)
        
        # Identity matrix expanded to batch size.
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(axis_angle.size(0), 3, 3)
        
        # Rodrigues' rotation formula.
        return I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    else:
        # NumPy implementation
        return R.from_rotvec(axis_angle).as_matrix()

def matrix_to_6d(R):
    """
    Converts rotation matrices to a 6D representation by taking the 
    first two columns and flattening them.
    
    Args:
        R: Rotation matrices of shape (B, 3, 3).
    
    Returns:
        6D representations of shape (B, 6).
    """
    return R[:, :, :2].reshape(R.size(0), 6)

def sixd_to_matrix(sixd):
    """
    Converts a 6D continuous representation back to a rotation matrix.
    This uses a Gram-Schmidt-like orthogonalization process.
    
    Args:
        sixd: 6D representations of shape (B, 6).
    
    Returns:
        Rotation matrices of shape (B, 3, 3).
    """
    # Split the 6D vector into two 3D vectors.
    a1 = sixd[:, 0:3]
    a2 = sixd[:, 3:6]
    
    # Normalize the first vector.
    b1 = F.normalize(a1, dim=-1)
    
    # Make the second vector orthogonal to the first.
    proj = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = a2 - proj * b1
    b2 = F.normalize(b2, dim=-1)
    
    # The third vector is the cross product of b1 and b2.
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack the three vectors as columns to form the rotation matrix.
    R = torch.stack([b1, b2, b3], dim=-1)
    return R

def matrix_to_axis_angle(matrix):
    """
    Converts rotation matrices to axis-angle representations.
    
    Args:
        matrix: Rotation matrices. PyTorch tensor of shape (B, 3, 3) or 
                NumPy array of shape (3, 3).
    
    Returns:
        Axis-angle representations, where the norm of each vector 
        represents the rotation angle (in radians) and the direction is the rotation axis.
    """
    if isinstance(matrix, torch.Tensor):
        # PyTorch implementation
        # Compute the trace of each rotation matrix.
        trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]  # (B,)
        
        # Clamp the value inside the valid range for arccos.
        cos_theta = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos_theta)  # (B,)
        
        # Compute the sine of theta and prepare for division.
        sin_theta = torch.sin(theta).unsqueeze(-1)  # (B, 1)
        
        # Compute the rotation axis components.
        axis_x = matrix[:, 2, 1] - matrix[:, 1, 2]
        axis_y = matrix[:, 0, 2] - matrix[:, 2, 0]
        axis_z = matrix[:, 1, 0] - matrix[:, 0, 1]
        axis = torch.stack([axis_x, axis_y, axis_z], dim=-1)  # (B, 3)
        
        # Avoid division by zero; for very small rotations, the axis can be arbitrary.
        axis = axis / (2 * sin_theta + 1e-8)
        
        # Multiply by the angle to obtain the axis-angle representation.
        return axis * theta.unsqueeze(-1)
    else:
        # NumPy implementation
        return R.from_matrix(matrix).as_rotvec()

def axis_angle_to_quaternion(a, eps=1e-7):
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

#------------------------------------------------------------------------------------------------
# Quaternion Operations - Basic quaternion math
#------------------------------------------------------------------------------------------------

def quaternion_inverse(q):
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

def quaternion_multiply(q1, q2):
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

def compute_relative_quaternion(q1, q2):
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

def quaternion_geodesic_distance(q1, q2):
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

def geodesic_distance(a1, a2):
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

def compute_angular_velocity(q1, q2):
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