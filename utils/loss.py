import torch
import torch.nn.functional as F
from utils.rot_and_trans import axis_angle_to_quaternion

#------------------------------------------------------------------------------------------------
# Helper functions for quaternion operations
#------------------------------------------------------------------------------------------------

def quaternion_inverse(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a quaternion.
    
    Args:
        quaternion: Quaternion tensor [..., 4]
        
    Returns:
        Inverse quaternion
    """
    inv = quaternion.clone()
    inv[..., 1:] = -inv[..., 1:]
    return inv

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
# Loss functions
#------------------------------------------------------------------------------------------------

def geodesic_loss(pred_rotmats: torch.Tensor, gt_rotmats: torch.Tensor) -> torch.Tensor:
    """
    Compute the geodesic loss between predicted and ground truth rotation matrices.
    
    Args:
        pred_rotmats: Predicted rotation matrices [B, ..., 3, 3]
        gt_rotmats: Ground truth rotation matrices [B, ..., 3, 3]
        
    Returns:
        Geodesic loss (mean of rotation angles)
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

def geodesic_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
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
    
    # Convert to quaternions
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    
    # Compute geodesic distance
    angles = quaternion_geodesic_distance(gt_quats, recon_quats)
    
    # Return mean geodesic distance
    return angles.mean()

def velocity_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
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
    device = ground_truth.device
    
    # Reshape to separate joints
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_joints, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_joints, 3)
    
    # Convert to quaternions for efficient computation
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    
    # Initialize tensors to store angular velocities
    gt_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=device)
    recon_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=device)
    
    # Compute angular velocities
    for t in range(n_frames-1):
        gt_velocity[:, t] = compute_angular_velocity(gt_quats[:, t], gt_quats[:, t+1])
        recon_velocity[:, t] = compute_angular_velocity(recon_quats[:, t], recon_quats[:, t+1])
    
    # Compute loss between velocities
    return F.mse_loss(recon_velocity, gt_velocity)

def acceleration_loss(ground_truth: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
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
    device = ground_truth.device
    
    # Not enough frames for acceleration
    if n_frames <= 2:
        return torch.tensor(0.0, device=device)
        
    # Reshape to separate joints
    gt_reshape = ground_truth.reshape(batch_size, n_frames, n_joints, 3)
    recon_reshape = reconstruction.reshape(batch_size, n_frames, n_joints, 3)
    
    # Convert to quaternions for efficient computation
    gt_quats = axis_angle_to_quaternion(gt_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    recon_quats = axis_angle_to_quaternion(recon_reshape.reshape(-1, 3)).reshape(batch_size, n_frames, n_joints, 4)
    
    # Initialize tensors to store angular velocities
    gt_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=device)
    recon_velocity = torch.zeros(batch_size, n_frames-1, n_joints, device=device)
    
    # Compute angular velocities
    for t in range(n_frames-1):
        gt_velocity[:, t] = compute_angular_velocity(gt_quats[:, t], gt_quats[:, t+1])
        recon_velocity[:, t] = compute_angular_velocity(recon_quats[:, t], recon_quats[:, t+1])
    
    # Compute acceleration (second derivative) as the difference in velocities
    gt_accel = gt_velocity[:, 1:] - gt_velocity[:, :-1]
    recon_accel = recon_velocity[:, 1:] - recon_velocity[:, :-1]
    
    # Compute loss between accelerations
    return F.mse_loss(recon_accel, gt_accel)

def combined_motion_loss(ground_truth: torch.Tensor, 
                          reconstruction: torch.Tensor, 
                          geo_weight: float = 1.0, 
                          vel_weight: float = 1.0, 
                          accel_weight: float = 0.5) -> dict:
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
    geo_loss = geodesic_loss(ground_truth, reconstruction)
    vel_loss =  velocity_loss(ground_truth, reconstruction)
    accel_loss = acceleration_loss(ground_truth, reconstruction)
    
    # Compute weighted total loss
    total_loss = geo_weight * geo_loss + vel_weight * vel_loss + accel_weight * accel_loss
    
    # Return all loss components
    return {
        'total': total_loss,
        'geodesic': geo_loss,
        'velocity': vel_loss,
        'acceleration': accel_loss
    }