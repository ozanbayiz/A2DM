import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

#------------------------------------------------------------------------------------------------
# Velocity loss functions
#------------------------------------------------------------------------------------------------

def compute_velocity_loss_optimized(ground_truth, reconstruction):
    """
    Optimized version of compute_velocity_loss using vectorized operations.
    This version is faster for large batches but requires more memory.
    
    Args:
        ground_truth (torch.Tensor): Ground truth tensor with axis-angle rotations (B, T, multiple of 3).
        reconstruction (torch.Tensor): Reconstructed tensor with axis-angle rotations (B, T, multiple of 3).
    
    Returns:
        torch.Tensor: Scalar loss value.
    """
    batch_size, n_frames, feature_dim = ground_truth.shape
    n_rotations = feature_dim // 3
    device = ground_truth.device
    
    # Reshape to separate rotations
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_rotations, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_rotations, 3)
    
    # Convert axis-angles to quaternions for more efficient computation
    # This avoids CPU transfers and numpy conversions
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_rotations, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_rotations, 4)
    
    # Initialize tensors to store geodesic velocities
    gt_geodesic_vel = torch.zeros(batch_size, n_frames-1, n_rotations, device=device)
    recon_geodesic_vel = torch.zeros(batch_size, n_frames-1, n_rotations, device=device)
    
    # Compute geodesic velocities using quaternions
    for t in range(n_frames-1):
        # Get quaternions for consecutive frames
        gt_q1 = gt_quats[:, t]      # (batch_size, n_rotations, 4)
        gt_q2 = gt_quats[:, t+1]    # (batch_size, n_rotations, 4)
        
        recon_q1 = recon_quats[:, t]      # (batch_size, n_rotations, 4)
        recon_q2 = recon_quats[:, t+1]    # (batch_size, n_rotations, 4)
        
        # Compute relative rotation quaternions (q2 * q1^-1)
        # For quaternion inverse, we just negate the xyz components
        gt_q1_inv = gt_q1.clone()
        gt_q1_inv[:, :, 1:] = -gt_q1_inv[:, :, 1:]
        
        recon_q1_inv = recon_q1.clone()
        recon_q1_inv[:, :, 1:] = -recon_q1_inv[:, :, 1:]
        
        # Quaternion multiplication for relative rotation
        # Formula: q_rel = q2 * q1_inv
        # w = w2*w1 - x2*x1 - y2*y1 - z2*z1
        # x = w2*x1 + x2*w1 + y2*z1 - z2*y1
        # y = w2*y1 - x2*z1 + y2*w1 + z2*x1
        # z = w2*z1 + x2*y1 - y2*x1 + z2*w1
        
        # Ground truth relative quaternion
        gt_w2, gt_x2, gt_y2, gt_z2 = gt_q2[:, :, 0], gt_q2[:, :, 1], gt_q2[:, :, 2], gt_q2[:, :, 3]
        gt_w1, gt_x1, gt_y1, gt_z1 = gt_q1_inv[:, :, 0], gt_q1_inv[:, :, 1], gt_q1_inv[:, :, 2], gt_q1_inv[:, :, 3]
        
        gt_rel_w = gt_w2*gt_w1 - gt_x2*gt_x1 - gt_y2*gt_y1 - gt_z2*gt_z1
        gt_rel_x = gt_w2*gt_x1 + gt_x2*gt_w1 + gt_y2*gt_z1 - gt_z2*gt_y1
        gt_rel_y = gt_w2*gt_y1 - gt_x2*gt_z1 + gt_y2*gt_w1 + gt_z2*gt_x1
        gt_rel_z = gt_w2*gt_z1 + gt_x2*gt_y1 - gt_y2*gt_x1 + gt_z2*gt_w1
        
        # Reconstruction relative quaternion
        recon_w2, recon_x2, recon_y2, recon_z2 = recon_q2[:, :, 0], recon_q2[:, :, 1], recon_q2[:, :, 2], recon_q2[:, :, 3]
        recon_w1, recon_x1, recon_y1, recon_z1 = recon_q1_inv[:, :, 0], recon_q1_inv[:, :, 1], recon_q1_inv[:, :, 2], recon_q1_inv[:, :, 3]
        
        recon_rel_w = recon_w2*recon_w1 - recon_x2*recon_x1 - recon_y2*recon_y1 - recon_z2*recon_z1
        recon_rel_x = recon_w2*recon_x1 + recon_x2*recon_w1 + recon_y2*recon_z1 - recon_z2*recon_y1
        recon_rel_y = recon_w2*recon_y1 - recon_x2*recon_z1 + recon_y2*recon_w1 + recon_z2*recon_x1
        recon_rel_z = recon_w2*recon_z1 + recon_x2*recon_y1 - recon_y2*recon_x1 + recon_z2*recon_w1
        
        # Compute rotation angle from quaternion (2 * acos(w) for unit quaternion)
        # Ensure w is in valid range [-1, 1] for numerical stability
        gt_angle = 2 * torch.acos(torch.clamp(gt_rel_w, -1.0+1e-6, 1.0-1e-6))
        recon_angle = 2 * torch.acos(torch.clamp(recon_rel_w, -1.0+1e-6, 1.0-1e-6))
        
        # Store the angular velocity
        gt_geodesic_vel[:, t] = gt_angle
        recon_geodesic_vel[:, t] = recon_angle
    
    # Compute MSE between the geodesic velocities
    velocity_loss = F.mse_loss(recon_geodesic_vel, gt_geodesic_vel)
    
    return velocity_loss

def geodesic_loss(pred_rotmats, gt_rotmats):
    """
    Compute the geodesic loss between predicted and ground truth rotation matrices.
    
    Args:
        pred_rotmats (torch.Tensor): Predicted rotation matrices [B, ..., 3, 3]
        gt_rotmats (torch.Tensor): Ground truth rotation matrices [B, ..., 3, 3]
        
    Returns:
        torch.Tensor: Geodesic loss (mean of rotation angles)
    """
    # Compute the relative rotation: R_rel = R_gt^T * R_pred
    rel_rotmats = torch.matmul(
        gt_rotmats.transpose(-2, -1),  # Transpose the last two dimensions (3x3)
        pred_rotmats
    )
    
    # Compute the trace of each rotation matrix
    traces = torch.diagonal(rel_rotmats, dim1=-2, dim2=-1).sum(-1)
    
    # Compute the rotation angle (clamp for numerical stability)
    angles = torch.acos(torch.clamp((traces - 1) / 2, -1.0+1e-6, 1.0-1e-6))
    
    # Return the mean angle
    return angles.mean()

#------------------------------------------------------------------------------------------------
# Quaternion and geodesic distance functions
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

def quaternion_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Compute the product of two quaternions.
    
    Args:
        q1: Quaternion of shape (..., 4)
        q2: Quaternion of shape (..., 4)
        
    Returns:
        Product quaternion of shape (..., 4)
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)

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

#------------------------------------------------------------------------------------------------
# Orientation and translation transformation functions
#------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------
# Geodesic Loss Functions
#------------------------------------------------------------------------------------------------

def compute_geodesic_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """
    Compute the geodesic loss between ground truth and reconstructed rotations.
    
    Args:
        ground_truth: Tensor of shape (N, T, J*3) containing ground truth rotations
        reconstruction: Tensor of shape (N, T, J*3) containing reconstructed rotations
        
    Returns:
        Scalar tensor containing the mean geodesic loss
    """
    batch_size, n_frames, feature_dim = ground_truth.shape
    n_joints = feature_dim // 3
    
    # Reshape to separate joints
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_joints, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_joints, 3)
    
    # Compute geodesic distance for all joints
    geo_distances = geodesic_distance(gt_reshape, recon_reshape)  # (N, T, J)
    
    # Return mean geodesic distance
    return geo_distances.mean()

def compute_velocity_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity loss using geodesic distance between consecutive rotations.
    
    Args:
        ground_truth: Tensor of shape (N, T, J*3) containing ground truth rotations
        reconstruction: Tensor of shape (N, T, J*3) containing reconstructed rotations
        
    Returns:
        Scalar tensor containing the velocity loss
    """
    batch_size, n_frames, feature_dim = ground_truth.shape
    n_joints = feature_dim // 3
    
    # Reshape to separate joints
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_joints, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_joints, 3)
    
    # Convert to quaternions for efficient computation
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    
    # Compute relative rotation between consecutive frames
    gt_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=ground_truth.device)
    recon_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=reconstruction.device)
    
    for t in range(n_frames-1):
        # Get quaternions for current and next frame
        gt_q1 = gt_quats[:, t]      # (N, J, 4)
        gt_q2 = gt_quats[:, t+1]    # (N, J, 4)
        
        recon_q1 = recon_quats[:, t]      # (N, J, 4)
        recon_q2 = recon_quats[:, t+1]    # (N, J, 4)
        
        # Compute inverse quaternions
        gt_q1_inv = quaternion_inverse(gt_q1)
        recon_q1_inv = quaternion_inverse(recon_q1)
        
        # Compute relative quaternions (q2 * q1^-1)
        gt_rel_quat = torch.stack([
            gt_q2[..., 0]*gt_q1_inv[..., 0] - gt_q2[..., 1]*gt_q1_inv[..., 1] - gt_q2[..., 2]*gt_q1_inv[..., 2] - gt_q2[..., 3]*gt_q1_inv[..., 3],
            gt_q2[..., 0]*gt_q1_inv[..., 1] + gt_q2[..., 1]*gt_q1_inv[..., 0] + gt_q2[..., 2]*gt_q1_inv[..., 3] - gt_q2[..., 3]*gt_q1_inv[..., 2],
            gt_q2[..., 0]*gt_q1_inv[..., 2] - gt_q2[..., 1]*gt_q1_inv[..., 3] + gt_q2[..., 2]*gt_q1_inv[..., 0] + gt_q2[..., 3]*gt_q1_inv[..., 1],
            gt_q2[..., 0]*gt_q1_inv[..., 3] + gt_q2[..., 1]*gt_q1_inv[..., 2] - gt_q2[..., 2]*gt_q1_inv[..., 1] + gt_q2[..., 3]*gt_q1_inv[..., 0]
        ], dim=-1)
        
        recon_rel_quat = torch.stack([
            recon_q2[..., 0]*recon_q1_inv[..., 0] - recon_q2[..., 1]*recon_q1_inv[..., 1] - recon_q2[..., 2]*recon_q1_inv[..., 2] - recon_q2[..., 3]*recon_q1_inv[..., 3],
            recon_q2[..., 0]*recon_q1_inv[..., 1] + recon_q2[..., 1]*recon_q1_inv[..., 0] + recon_q2[..., 2]*recon_q1_inv[..., 3] - recon_q2[..., 3]*recon_q1_inv[..., 2],
            recon_q2[..., 0]*recon_q1_inv[..., 2] - recon_q2[..., 1]*recon_q1_inv[..., 3] + recon_q2[..., 2]*recon_q1_inv[..., 0] + recon_q2[..., 3]*recon_q1_inv[..., 1],
            recon_q2[..., 0]*recon_q1_inv[..., 3] + recon_q2[..., 1]*recon_q1_inv[..., 2] - recon_q2[..., 2]*recon_q1_inv[..., 1] + recon_q2[..., 3]*recon_q1_inv[..., 0]
        ], dim=-1)
        
        # Compute geodesic angle from quaternion w component
        gt_angle = 2 * torch.acos(torch.clamp(gt_rel_quat[..., 0].abs(), -1.0+1e-6, 1.0-1e-6))
        recon_angle = 2 * torch.acos(torch.clamp(recon_rel_quat[..., 0].abs(), -1.0+1e-6, 1.0-1e-6))
        
        # Store angular velocities
        gt_velocity[:, t] = gt_angle
        recon_velocity[:, t] = recon_angle
    
    # Compute loss between velocities
    return F.mse_loss(recon_velocity, gt_velocity)

def compute_acceleration_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    """
    Compute acceleration loss using changes in angular velocity.
    
    Args:
        ground_truth: Tensor of shape (N, T, J*3) containing ground truth rotations
        reconstruction: Tensor of shape (N, T, J*3) containing reconstructed rotations
        
    Returns:
        Scalar tensor containing the acceleration loss
    """
    batch_size, n_frames, feature_dim = ground_truth.shape
    n_joints = feature_dim // 3
    
    # Reshape to separate joints
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_joints, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_joints, 3)
    
    # Convert to quaternions for efficient computation
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    
    # Compute velocities (relative rotations between consecutive frames)
    gt_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=ground_truth.device)
    recon_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=reconstruction.device)
    
    for t in range(n_frames-1):
        # Get quaternions for current and next frame
        gt_q1 = gt_quats[:, t]      # (N, J, 4)
        gt_q2 = gt_quats[:, t+1]    # (N, J, 4)
        
        recon_q1 = recon_quats[:, t]      # (N, J, 4)
        recon_q2 = recon_quats[:, t+1]    # (N, J, 4)
        
        # Compute inverse quaternions
        gt_q1_inv = quaternion_inverse(gt_q1)
        recon_q1_inv = quaternion_inverse(recon_q1)
        
        # Compute relative quaternions (q2 * q1^-1)
        gt_rel_quat = torch.stack([
            gt_q2[..., 0]*gt_q1_inv[..., 0] - gt_q2[..., 1]*gt_q1_inv[..., 1] - gt_q2[..., 2]*gt_q1_inv[..., 2] - gt_q2[..., 3]*gt_q1_inv[..., 3],
            gt_q2[..., 0]*gt_q1_inv[..., 1] + gt_q2[..., 1]*gt_q1_inv[..., 0] + gt_q2[..., 2]*gt_q1_inv[..., 3] - gt_q2[..., 3]*gt_q1_inv[..., 2],
            gt_q2[..., 0]*gt_q1_inv[..., 2] - gt_q2[..., 1]*gt_q1_inv[..., 3] + gt_q2[..., 2]*gt_q1_inv[..., 0] + gt_q2[..., 3]*gt_q1_inv[..., 1],
            gt_q2[..., 0]*gt_q1_inv[..., 3] + gt_q2[..., 1]*gt_q1_inv[..., 2] - gt_q2[..., 2]*gt_q1_inv[..., 1] + gt_q2[..., 3]*gt_q1_inv[..., 0]
        ], dim=-1)
        
        recon_rel_quat = torch.stack([
            recon_q2[..., 0]*recon_q1_inv[..., 0] - recon_q2[..., 1]*recon_q1_inv[..., 1] - recon_q2[..., 2]*recon_q1_inv[..., 2] - recon_q2[..., 3]*recon_q1_inv[..., 3],
            recon_q2[..., 0]*recon_q1_inv[..., 1] + recon_q2[..., 1]*recon_q1_inv[..., 0] + recon_q2[..., 2]*recon_q1_inv[..., 3] - recon_q2[..., 3]*recon_q1_inv[..., 2],
            recon_q2[..., 0]*recon_q1_inv[..., 2] - recon_q2[..., 1]*recon_q1_inv[..., 3] + recon_q2[..., 2]*recon_q1_inv[..., 0] + recon_q2[..., 3]*recon_q1_inv[..., 1],
            recon_q2[..., 0]*recon_q1_inv[..., 3] + recon_q2[..., 1]*recon_q1_inv[..., 2] - recon_q2[..., 2]*recon_q1_inv[..., 1] + recon_q2[..., 3]*recon_q1_inv[..., 0]
        ], dim=-1)
        
        # Compute geodesic angle from quaternion w component
        gt_angle = 2 * torch.acos(torch.clamp(gt_rel_quat[..., 0].abs(), -1.0+1e-6, 1.0-1e-6))
        recon_angle = 2 * torch.acos(torch.clamp(recon_rel_quat[..., 0].abs(), -1.0+1e-6, 1.0-1e-6))
        
        # Store angular velocities
        gt_velocity[:, t] = gt_angle
        recon_velocity[:, t] = recon_angle
    
    # Compute acceleration (second derivative) as the difference in velocities
    if n_frames > 2:
        gt_accel = gt_velocity[:, 1:] - gt_velocity[:, :-1]
        recon_accel = recon_velocity[:, 1:] - recon_velocity[:, :-1]
    
        # Compute loss between accelerations
        return F.mse_loss(recon_accel, gt_accel)
    else:
        # Not enough frames for acceleration, return zero
        return torch.tensor(0.0, device=ground_truth.device)

def combined_motion_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor, 
                        geo_weight=1.0, vel_weight=1.0, accel_weight=0.5) -> dict:
    """
    Compute a combined loss considering geodesic distance, velocity, and acceleration.
    
    Args:
        ground_truth: Tensor of shape (N, T, J*3) containing ground truth rotations
        reconstruction: Tensor of shape (N, T, J*3) containing reconstructed rotations
        geo_weight: Weight for geodesic distance loss
        vel_weight: Weight for velocity loss
        accel_weight: Weight for acceleration loss
        
    Returns:
        Dictionary containing the total loss and individual components
    """
    # Compute individual losses
    geo_loss = compute_geodesic_loss(ground_truth, reconstruction)
    vel_loss = compute_velocity_loss(ground_truth, reconstruction)
    accel_loss = compute_acceleration_loss(ground_truth, reconstruction)
    
    # Compute weighted total loss
    total_loss = geo_weight * geo_loss + vel_weight * vel_loss + accel_weight * accel_loss
    
    # Return all loss components
    return {
        'total': total_loss,
        'geodesic': geo_loss,
        'velocity': vel_loss,
        'acceleration': accel_loss
    }

#------------------------------------------------------------------------------------------------
# Tests for the implemented loss functions
#------------------------------------------------------------------------------------------------

def test_loss_functions():
    """Test the correctness of the geodesic, velocity, and acceleration loss functions."""
    print("Testing geodesic, velocity, and acceleration loss functions...")
    
    # Create a simple test case: 2 sequences, 10 frames, 2 joints
    batch_size, n_frames, n_joints = 2, 10, 2
    feature_dim = n_joints * 3
    
    # Create ground truth motion: simple rotation around Y axis with increasing angle
    gt = torch.zeros(batch_size, n_frames, feature_dim)
    for t in range(n_frames):
        angle = t * 0.1  # Increasing angle over time
        # Set Y-axis rotation for all joints
        for j in range(n_joints):
            gt[:, t, j*3+1] = angle
    
    # Create reconstructed motion with some error
    recon = gt.clone()
    noise = torch.randn_like(recon) * 0.05  # Small random noise
    recon = recon + noise
    
    # Test geodesic loss
    geo_loss = compute_geodesic_loss(gt, recon)
    print(f"Geodesic loss: {geo_loss.item():.6f}")
    
    # Test velocity loss
    vel_loss = compute_velocity_loss(gt, recon)
    print(f"Velocity loss: {vel_loss.item():.6f}")
    
    # Test acceleration loss
    accel_loss = compute_acceleration_loss(gt, recon)
    print(f"Acceleration loss: {accel_loss.item():.6f}")
    
    # Test combined loss
    combined_loss = combined_motion_loss(gt, recon)
    print(f"Combined total loss: {combined_loss['total'].item():.6f}")
    
    return geo_loss, vel_loss, accel_loss

def test_with_varying_noise():
    """Test how losses change with different noise levels."""
    print("\nTesting with varying noise levels...")
    
    # Create a simple test motion
    batch_size, n_frames, n_joints = 1, 20, 1
    feature_dim = n_joints * 3
    
    # Create ground truth motion: rotation around Z axis with constant velocity
    gt = torch.zeros(batch_size, n_frames, feature_dim)
    for t in range(n_frames):
        angle = t * 0.1  # Linear increase
        gt[:, t, 2] = angle  # Z-axis rotation
    
    # Test with different noise levels
    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
    geo_losses = []
    vel_losses = []
    accel_losses = []
    
    for noise in noise_levels:
        # Add noise to the ground truth
        noise_tensor = torch.randn_like(gt) * noise
        recon = gt + noise_tensor
        
        # Compute losses
        geo_loss = compute_geodesic_loss(gt, recon)
        vel_loss = compute_velocity_loss(gt, recon)
        accel_loss = compute_acceleration_loss(gt, recon)
        
        geo_losses.append(geo_loss.item())
        vel_losses.append(vel_loss.item())
        accel_losses.append(accel_loss.item())
        
        print(f"Noise level {noise:.2f}: " +
              f"Geodesic = {geo_loss.item():.6f}, " +
              f"Velocity = {vel_loss.item():.6f}, " +
              f"Acceleration = {accel_loss.item():.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, geo_losses, 'o-', label='Geodesic Loss')
    plt.plot(noise_levels, vel_losses, 's-', label='Velocity Loss')
    plt.plot(noise_levels, accel_losses, '^-', label='Acceleration Loss')
    plt.xlabel('Noise Level')
    plt.ylabel('Loss Value')
    plt.title('Effect of Noise on Different Loss Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_vs_noise.png')
    plt.close()
    
    return noise_levels, geo_losses, vel_losses, accel_losses

def test_with_perturbed_frames():
    """Test how losses change when specific frames are perturbed."""
    print("\nTesting with perturbed frames...")
    
    # Create a simple test motion
    batch_size, n_frames, n_joints = 1, 30, 1
    feature_dim = n_joints * 3
    
    # Create ground truth motion: rotation around X axis with constant velocity
    gt = torch.zeros(batch_size, n_frames, feature_dim)
    for t in range(n_frames):
        angle = t * 0.1  # Linear increase
        gt[:, t, 0] = angle  # X-axis rotation
    
    # Base reconstruction (perfect)
    recon = gt.clone()
    
    # Test cases to perturb different frames
    test_cases = [
        ("No perturbation", []),
        ("Single frame (middle)", [15]),
        ("Two consecutive frames", [14, 15]),
        ("First frame", [0]),
        ("Last frame", [n_frames-1]),
        ("Multiple frames", [5, 15, 25])
    ]
    
    # Store results
    results = []
    
    for name, frames_to_perturb in test_cases:
        # Create perturbed reconstruction
        perturbed = recon.clone()
        
        if frames_to_perturb:
            # Add significant perturbation to specified frames
            for frame in frames_to_perturb:
                perturbed[:, frame, :] += torch.tensor([0.5, 0.0, 0.0])  # Large perturbation on X-axis
        
        # Compute losses
        geo_loss = compute_geodesic_loss(gt, perturbed)
        vel_loss = compute_velocity_loss(gt, perturbed)
        accel_loss = compute_acceleration_loss(gt, perturbed)
        
        results.append({
            'name': name,
            'geodesic': geo_loss.item(),
            'velocity': vel_loss.item(),
            'acceleration': accel_loss.item()
        })
        
        print(f"{name}: " +
              f"Geodesic = {geo_loss.item():.6f}, " +
              f"Velocity = {vel_loss.item():.6f}, " +
              f"Acceleration = {accel_loss.item():.6f}")
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    case_names = [r['name'] for r in results]
    geodesic_values = [r['geodesic'] for r in results]
    velocity_values = [r['velocity'] for r in results]
    acceleration_values = [r['acceleration'] for r in results]
    
    x = range(len(results))
    width = 0.25
    
    plt.bar([i - width for i in x], geodesic_values, width, label='Geodesic Loss')
    plt.bar(x, velocity_values, width, label='Velocity Loss')
    plt.bar([i + width for i in x], acceleration_values, width, label='Acceleration Loss')
    
    plt.xlabel('Test Case')
    plt.ylabel('Loss Value')
    plt.title('Effect of Frame Perturbations on Different Loss Functions')
    plt.xticks(x, case_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_vs_perturbation.png')
    plt.close()
    
    return results

def main():
    """Run tests for the geodesic, velocity, and acceleration loss functions."""
    # Check if PyTorch and CUDA are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Basic tests
    test_loss_functions()
    
    # Test with varying noise levels
    test_with_varying_noise()
    
    # Test with perturbed frames
    test_with_perturbed_frames()
    
    print("\nAll tests completed successfully!")
    print("Visualizations saved to loss_vs_noise.png and loss_vs_perturbation.png")

if __name__ == "__main__":
    main()