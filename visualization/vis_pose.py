"""
3D visualization for human motion using SMPL models and Viser.
Compares original and VAE-reconstructed human motion.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
from dataclasses import dataclass

import numpy as np
import viser
import viser.transforms as tf
import torch
from scipy.spatial.transform import Rotation

from decoupled_vae import DecoupledVAE
#------------------------------------------------------------------------------------------------
# SMPL Model Handling
#------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SmplFkOutputs:
    """Forward kinematics outputs for SMPL model."""
    T_world_joint: np.ndarray  # (num_joints, 4, 4)
    T_parent_joint: np.ndarray  # (num_joints, 4, 4)


class SmplHelper:
    """Helper for SMPL models (numpy version, without blend skinning)."""
    
    def __init__(self, model_path: Path) -> None:
        """
        Initialize SMPL helper from model file.
        
        Args:
            model_path: Path to SMPL .npz file
        """
        if not model_path.exists():
            raise FileNotFoundError(f"SMPL model file not found: {model_path}")
            
        if model_path.suffix.lower() != ".npz":
            raise ValueError("Model should be an .npz file!")
            
        body_dict = dict(**np.load(model_path, allow_pickle=True))
        
        # Load SMPL parameters
        self.J_regressor = body_dict["J_regressor"]
        self.weights = body_dict["weights"]
        self.v_template = body_dict["v_template"]
        self.posedirs = body_dict["posedirs"]
        self.shapedirs = body_dict["shapedirs"]
        self.faces = body_dict["f"]
        self.parent_idx = body_dict["kintree_table"][0]
        
        # Store dimensions
        self.num_joints: int = self.weights.shape[-1]
        self.num_betas: int = self.shapedirs.shape[-1]
        self.num_vertices: int = self.v_template.shape[0]

    def get_tpose(self, betas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get T-pose vertices and joints for given shape parameters.
        
        Args:
            betas: Shape parameters (num_betas,)
            
        Returns:
            v_tpose: Vertices in T-pose (num_vertices, 3)
            j_tpose: Joints in T-pose (num_joints, 3)
        """
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)
        return v_tpose, j_tpose

    def get_all_time_outputs(self, 
                            betas: np.ndarray, 
                            joint_rotmats: np.ndarray, 
                            translation: np.ndarray) -> np.ndarray:
        """
        Compute joint transformations for all time steps.
        
        Args:
            betas: Shape parameters (num_betas,)
            joint_rotmats: Joint rotation matrices (seq_len, num_joints, 3, 3)
            translation: Global translation (seq_len, 3)
            
        Returns:
            Joint transformation matrices (seq_len, num_joints, 4, 4)
        """
        # Get T-pose joint locations based on shape
        v_tpose, j_tpose = self.get_tpose(betas)
        
        # Get sequence length
        seq_len = joint_rotmats.shape[0]
        
        # Initialize parent-relative transformations
        T_parent_joint = np.tile(np.eye(4)[None, None, ...], (seq_len, self.num_joints, 1, 1))
        
        # Set rotation part
        T_parent_joint[:, :, :3, :3] = joint_rotmats
        
        # Set translation part (offset from parent)
        T_parent_joint[:, 0, :3, 3] = j_tpose[0]  # Root joint
        T_parent_joint[:, 1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]  # Child joints
        
        # Compose transformations along kinematic chain
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[:, i] = T_world_joint[:, self.parent_idx[i]] @ T_parent_joint[:, i]
        
        # Apply global translation
        T_translation = np.tile(np.eye(4)[None, ...], (seq_len, 1, 1))
        T_translation[:, :3, 3] = translation
        T_world_joint = T_translation[:, None, :, :] @ T_world_joint
        
        return T_world_joint


#------------------------------------------------------------------------------------------------
# Motion Data Processing
#------------------------------------------------------------------------------------------------

def load_motion_data(file_path: Union[str, Path]) -> Dict:
    """
    Load motion data from NPZ file.
    
    Args:
        file_path: Path to motion data file
        
    Returns:
        Dictionary with motion data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Motion data file not found: {path}")
        
    try:
        data = dict(np.load(path, allow_pickle=True))
        expected_keys = ['trans', 'root_orient', 'pose_body']
        
        # Check if required keys exist
        for key in expected_keys:
            if key not in data:
                raise ValueError(f"Required key '{key}' not found in motion data")
                
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load motion data: {str(e)}")


def extract_single_person_data(motion_data: Dict, person_idx: int = 0) -> Dict:
    """
    Extract data for a single person from potentially multi-person data.
    
    Args:
        motion_data: Motion data dictionary
        person_idx: Index of person to extract
        
    Returns:
        Dictionary with single-person motion data
    """
    result = {}
    
    # Process each component
    for key in ['trans', 'root_orient', 'pose_body']:
        data = motion_data[key]
        
        # Handle multi-person data
        if data.ndim > 2 and key in ['trans', 'root_orient']:
            # Check bounds
            if person_idx >= data.shape[0]:
                print(f"Warning: Person index {person_idx} out of bounds. Using first person.")
                result[key] = data[0]
            else:
                result[key] = data[person_idx]
        elif data.ndim > 3 and key == 'pose_body':
            # Handle 4D pose_body (persons, frames, joints, 3)
            if person_idx >= data.shape[0]:
                print(f"Warning: Person index {person_idx} out of bounds. Using first person.")
                result[key] = data[0]
            else:
                result[key] = data[person_idx]
        else:
            # Single person data
            result[key] = data
            
    # Ensure pose_body has the right shape
    if len(result['pose_body'].shape) == 2 and result['pose_body'].shape[1] % 3 == 0:
        # Flattened pose, reshape to (frames, joints, 3)
        n_joints = result['pose_body'].shape[1] // 3
        result['pose_body'] = result['pose_body'].reshape(result['pose_body'].shape[0], n_joints, 3)
            
    return result


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix.
    
    Args:
        axis_angle: Axis-angle rotations (batch_size, n_joints, 3) or (batch_size, 3)
        
    Returns:
        Rotation matrices (batch_size, n_joints, 3, 3)
    """
    batch_size = axis_angle.shape[0]
    
    # Determine if input is for multiple joints
    if len(axis_angle.shape) > 2:
        n_joints = axis_angle.shape[1]
        reshaped = axis_angle.reshape(-1, 3)  # Flatten to (batch*joints, 3)
    else:
        n_joints = 1
        reshaped = axis_angle.reshape(-1, 3)  # Flatten to (batch, 3)
    
    # Convert to rotation matrices using scipy
    rotmats = Rotation.from_rotvec(reshaped).as_matrix()
    
    # Reshape back to batch format
    return rotmats.reshape(batch_size, n_joints, 3, 3)


def prepare_joint_data(motion_data: Dict, smpl_helper: SmplHelper) -> np.ndarray:
    """
    Process motion data to get joint transformations for visualization.
    
    Args:
        motion_data: Dictionary with keys 'trans', 'root_orient', and 'pose_body'
        smpl_helper: SMPL helper object
        
    Returns:
        Joint transformation matrices for all time steps (seq_len, num_joints, 4, 4)
    """
    # Extract motion components
    trans = motion_data['trans']
    root_orient = motion_data['root_orient']
    pose_body = motion_data['pose_body']
    
    # Get sequence length and reshape pose_body if needed
    seq_len = trans.shape[0]
    
    if len(pose_body.shape) == 3:
        n_body_joints = pose_body.shape[1]
    else:
        # Flattened pose
        n_body_joints = pose_body.shape[1] // 3
        pose_body = pose_body.reshape(seq_len, n_body_joints, 3)
    
    # Create complete rotation array (with root joint)
    all_rots = np.zeros((seq_len, 24, 3))  # SMPL uses 24 joints
    all_rots[:, 0] = root_orient  # Root joint
    all_rots[:, 1:1+n_body_joints] = pose_body  # Body joints
    
    # Convert to rotation matrices
    joint_rotmats = axis_angle_to_matrix(all_rots)
    
    # Get shape parameters
    betas = np.zeros(smpl_helper.num_betas)
    if 'betas' in motion_data:
        betas_data = motion_data['betas']
        if betas_data.ndim > 1:
            betas[:10] = betas_data[0][:10]  # Take first 10 betas of first person
        else:
            betas[:10] = betas_data[:10]  # Take first 10 betas
    
    # Generate joint transformations
    return smpl_helper.get_all_time_outputs(betas, joint_rotmats, trans)


#------------------------------------------------------------------------------------------------
# VAE Reconstruction
#------------------------------------------------------------------------------------------------

def reconstruct_motion_with_vae(
    motion_data: Dict,
    checkpoints_dir: str = "checkpoints",
    device: str = "cpu"
) -> Dict:
    """
    Reconstruct motion using DecoupledVAE.
    
    Args:
        motion_data: Motion data dictionary
        checkpoints_dir: Directory containing VAE model checkpoints
        device: Device to run the model on
        
    Returns:
        Dictionary with reconstructed motion components
    """
    # Extract motion components
    trans = motion_data['trans']
    root_orient = motion_data['root_orient']
    pose_body = motion_data['pose_body']
    # Check that checkpoint files exist
    checkpoint_paths = {
        'translation': f"{checkpoints_dir}/translation_vae/translation_vae_best.pt",
        'orientation': f"{checkpoints_dir}/orientation_vae/orientation_vae_best.pt",
        'pose': f"{checkpoints_dir}/pose_vae/pose_vae_best.pt"
    }
    
    for name, path in checkpoint_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name.capitalize()} VAE checkpoint not found: {path}")
    
    # Initialize VAE model
    model = DecoupledVAE(
        translation_checkpoint_path=checkpoint_paths['translation'],
        orientation_checkpoint_path=checkpoint_paths['orientation'],
        pose_checkpoint_path=checkpoint_paths['pose'],
        device=torch.device(device)
    )
    
    # Set to evaluation mode
    model.eval()
    print("VAE model loaded successfully")
    
    # Reconstruct motion
    print("Reconstructing motion...")
    print(trans.shape, root_orient.shape, pose_body.shape)
    recon_motion = model.reconstruct(trans, root_orient, pose_body)
    
    # Compute and print errors
    original = {
        'trans': trans,
        'root_orient': root_orient,
        'pose_body': pose_body.reshape(pose_body.shape[0], -1) if len(pose_body.shape) > 2 else pose_body
    }
    
    errors = model.compute_errors(original, recon_motion)
    
    print("\nReconstruction Errors:")
    print("-" * 30)
    for component, metrics in errors.items():
        print(f"{component.capitalize()}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  L2: {metrics['l2']:.4f}")
    
    return recon_motion


#------------------------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------------------------

class MotionVisualizer:
    """Class for visualizing original and reconstructed motion."""
    
    def __init__(self, 
                smpl_helper: SmplHelper, 
                offset: np.ndarray = np.array([1.0, 0.0, 0.0])):
        """
        Initialize visualizer.
        
        Args:
            smpl_helper: SMPL helper object
            offset: Offset for reconstructed mesh
        """
        self.smpl_helper = smpl_helper
        self.offset = offset
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("+y")  # Y-up coordinate system
        self.server.gui.configure_theme(control_layout="collapsible")
        
        # Get T-pose vertices and joints
        self.v_tpose, self.j_tpose = smpl_helper.get_tpose(np.zeros((smpl_helper.num_betas,)))
        self.gt_mesh = None
        self.recon_mesh = None
        
    def setup_visualization(self):
        """
        Create visualization meshes for ground truth and reconstructed motions.
        
        Returns:
            Tuple of (ground_truth_mesh, reconstructed_mesh)
        """
        # Create ground truth mesh
        self.gt_mesh = self.server.scene.add_mesh_skinned(
            "/ground_truth",
            self.v_tpose - self.offset,
            self.smpl_helper.faces,
            bone_wxyzs=tf.SO3.identity(batch_axes=(self.smpl_helper.num_joints,)).wxyz,
            bone_positions=self.j_tpose - self.offset,
            skin_weights=self.smpl_helper.weights,
            wireframe=False,
            color=(153, 255, 204)  # Light green
        )
        
        # Create reconstructed mesh with offset
        self.recon_mesh = self.server.scene.add_mesh_skinned(
            "/reconstructed",
            self.v_tpose + self.offset,  # Offset vertices
            self.smpl_helper.faces,
            bone_wxyzs=tf.SO3.identity(batch_axes=(self.smpl_helper.num_joints,)).wxyz,
            bone_positions=self.j_tpose + self.offset,  # Offset joint positions
            skin_weights=self.smpl_helper.weights,
            wireframe=False,
            color=(102, 204, 255)  # Light blue
        )
        
        
        return self.gt_mesh, self.recon_mesh
    
    def update_mesh(self, mesh, joint_data, frame_idx, offset_direction):
        """
        Update a mesh's bone transformations for the current frame.
        
        Args:
            mesh: Skinned mesh to update
            joint_data: Joint transformation matrices
            frame_idx: Current frame index
            offset_direction: Whether to apply offset (for reconstructed mesh)
        """
        # Extract rotation and position from transformation matrices
        wxyz = tf.SO3.from_matrix(joint_data[frame_idx, :, :3, :3]).wxyz
        positions = joint_data[frame_idx, :, :3, 3]
        
        # Apply offset if needed
        if offset_direction == "recon":
            positions = positions + self.offset
        elif offset_direction == "gt":
            positions = positions - self.offset
        
        # Update each bone
        for i, bone in enumerate(mesh.bones):
            bone.wxyz = wxyz[i]
            bone.position = positions[i]
    
    def run_animation_loop(self, gt_joint_data, recon_joint_data, fps=25):
        """
        Animate both meshes at specified framerate.
        
        Args:
            gt_joint_data: Ground truth joint transformations
            recon_joint_data: Reconstructed joint transformations
            fps: Frames per second
        """
        # Create meshes if not already done
        if self.gt_mesh is None or self.recon_mesh is None:
            self.setup_visualization()
        
        # Animation loop variables
        frame_idx = 0
        num_frames = gt_joint_data.shape[0]
        frame_time = 1.0 / fps
        
        print(f"Starting animation loop with {num_frames} frames at {fps} FPS")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                # Update both meshes
                self.update_mesh(self.gt_mesh, gt_joint_data, frame_idx, offset_direction="gt")
                self.update_mesh(self.recon_mesh, recon_joint_data, frame_idx, offset_direction="recon")
                
                # Move to next frame (loop back to start when done)
                frame_idx = (frame_idx + 1) % num_frames
                
                # Wait for next frame
                time.sleep(frame_time)
        except KeyboardInterrupt:
            print("\nAnimation stopped by user")


#------------------------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------------------------

def main(args):
    """
    Main function to visualize original and VAE-reconstructed SMPL human motion.
    
    Args:
        args: Command line arguments
    """
    try:
        # Load SMPL model
        smpl_helper = SmplHelper(args.smpl_model)
        print(f"Loaded SMPL model with {smpl_helper.num_joints} joints")
        
        # Load ground truth motion data
        gt_data_raw = load_motion_data(args.input_file)
        # gt_data = extract_single_person_data(gt_data_raw, args.person_idx)
        gt_data = {
            'trans': gt_data_raw['trans'][args.person_idx],
            'root_orient': gt_data_raw['root_orient'][args.person_idx],
            'pose_body': gt_data_raw['pose_body'][args.person_idx]
        }
        print(f"Loaded motion data from {args.input_file}")
        
        # Print data shapes
        print(f"Data shapes:")
        print(f"  Translation: {gt_data['trans'].shape}")
        print(f"  Root orientation: {gt_data['root_orient'].shape}")
        print(f"  Pose body: {gt_data['pose_body'].shape}")
        
        # Reconstruct motion using DecoupledVAE
        recon_data = reconstruct_motion_with_vae(
            gt_data, 
            checkpoints_dir=args.checkpoints_dir,
            device=args.device
        )
        print("Motion reconstruction complete")
        
        # zero our trans and root_orient to only visualize pose
        gt_data['root_orient'] = np.zeros_like(gt_data['root_orient'])
        recon_data['root_orient'] = np.zeros_like(recon_data['root_orient'])
        gt_data['trans'] = np.zeros_like(gt_data['trans'])
        recon_data['trans'] = np.zeros_like(recon_data['trans'])
        
        # Prepare joint data for both original and reconstructed motion
        gt_joint_data = prepare_joint_data(gt_data, smpl_helper)
        recon_joint_data = prepare_joint_data(recon_data, smpl_helper)
        
        print(f"Joint data shapes:")
        print(f"  Ground truth: {gt_joint_data.shape}")
        print(f"  Reconstructed: {recon_joint_data.shape}")

        print("error: ", torch.norm(torch.tensor(gt_joint_data) - torch.tensor(recon_joint_data)))
        
        # Set up visualization and run animation
        visualizer = MotionVisualizer(smpl_helper, offset=args.viz_offset)
        print(f"Visualization server started at http://localhost:8080")
        
        visualizer.run_animation_loop(gt_joint_data, recon_joint_data, fps=args.fps)
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL Human Motion Reconstruction and Visualization")
    
    # Input/output paths
    parser.add_argument("--input_file", type=Path, default="single_person/sample_motion.npz",
                        help="Path to ground truth motion data (.npz)")
    parser.add_argument("--smpl_model", type=Path, default="smpl/smpl_neutral.npz",
                        help="Path to SMPL model file")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Directory containing VAE model checkpoints")
    
    # Motion data options
    parser.add_argument("--person_idx", type=int, default=0,
                        help="Person index to extract from multi-person data")
    
    # Visualization options
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second for animation")
    parser.add_argument("--viz_offset", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                        help="Offset for reconstructed mesh [x y z]")
    
    # Model options
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Device to run VAE model on")
    
    args = parser.parse_args()
    args.viz_offset = np.array(args.viz_offset)
    
    exit(main(args))