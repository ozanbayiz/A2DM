"""
Flexible 3D visualization for human motion using SMPL models and Viser.
Compares original and VAE-reconstructed human motion with configurable options.
"""

import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import viser
import viser.transforms as tf
import torch
from scipy.spatial.transform import Rotation

from utils.factory import load_model, get_dataset

#------------------------------------------------------------------------------------------------
# SMPL Model Handling
#------------------------------------------------------------------------------------------------

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

    def get_joint_transforms(self, 
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
# Motion Processing
#------------------------------------------------------------------------------------------------

def prepare_joint_data(motion_data: Dict, smpl_helper: SmplHelper, person_idx: int = 0) -> np.ndarray:
    """
    Process motion data to get joint transformations for visualization.
    
    Args:
        motion_data: Dictionary with keys 'trans', 'root_orient', and 'pose_body'
        smpl_helper: SMPL helper object
        person_idx: Index of person to visualize
        
    Returns:
        Joint transformation matrices for all time steps (seq_len, num_joints, 4, 4)
    """
    # Extract motion components
    trans = motion_data['trans'].numpy() if isinstance(motion_data['trans'], torch.Tensor) else motion_data['trans']
    root_orient = motion_data['root_orient'].numpy() if isinstance(motion_data['root_orient'], torch.Tensor) else motion_data['root_orient']
    pose_body = motion_data['pose_body'].numpy() if isinstance(motion_data['pose_body'], torch.Tensor) else motion_data['pose_body']
    
    # Get sequence length
    seq_len = trans.shape[0]
    
    # Ensure pose_body has the right shape - should be (seq_len, 63) or convertible to that
    if len(pose_body.shape) > 2:
        # If it's in format (seq_len, n_joints, 3), flatten it
        pose_body = pose_body.reshape(seq_len, -1)
    
    # Number of body joints (SMPL has 23 body joints + 1 root joint)
    n_body_joints = pose_body.shape[1] // 3
    
    # Create complete rotation array (with root joint)
    all_rots = np.zeros((seq_len, 24, 3))  # SMPL uses 24 joints
    all_rots[:, 0] = root_orient  # Root joint
    
    # Body joints - make sure to handle different possible formats
    pose_body_reshaped = pose_body.reshape(seq_len, n_body_joints, 3)
    all_rots[:, 1:1+n_body_joints] = pose_body_reshaped  # Body joints
    
    # Convert to rotation matrices
    joint_rotmats = np.array([Rotation.from_rotvec(all_rots[i].reshape(-1, 3)).as_matrix() 
                             for i in range(seq_len)])
    
    # Get shape parameters - use zeros for default shape
    betas = np.zeros(smpl_helper.num_betas)
    
    # Generate joint transformations
    return smpl_helper.get_joint_transforms(betas, joint_rotmats, trans)


#------------------------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------------------------

class MotionVisualizer:
    """Class for visualizing original and reconstructed motion."""
    
    def __init__(self, 
                smpl_helper: SmplHelper, 
                offsets: List[np.ndarray] = None,
                colors: List[Tuple[int, int, int]] = None):
        """
        Initialize visualizer.
        
        Args:
            smpl_helper: SMPL helper object
            offsets: List of offsets for each mesh [original, reconstructed]
            colors: List of colors for each mesh [original, reconstructed]
        """
        self.smpl_helper = smpl_helper
        
        # Default offsets and colors if not provided
        if offsets is None:
            self.offsets = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0])]
        else:
            self.offsets = offsets
            
        if colors is None:
            self.colors = [(153, 255, 204), (102, 204, 255)]  # Light green, Light blue
        else:
            self.colors = colors
            
        self.server = viser.ViserServer()
        self.server.scene.set_up_direction("+y")
        self.server.gui.configure_theme(control_layout="collapsible")
        
        # Get T-pose vertices and joints
        self.v_tpose, self.j_tpose = smpl_helper.get_tpose(np.zeros((smpl_helper.num_betas,)))
        self.meshes = []
        
    def setup_visualization(self, num_people: int = 2):
        """
        Create visualization meshes for ground truth and reconstructed motions.
        
        Args:
            num_people: Number of people to visualize (1 or 2)
        
        Returns:
            List of created meshes
        """
        self.meshes = []
        
        names = ["/ground_truth", "/reconstructed"]
        
        for i in range(num_people):
            mesh = self.server.scene.add_mesh_skinned(
                names[i],
                self.v_tpose + self.offsets[i],
                self.smpl_helper.faces,
                bone_wxyzs=tf.SO3.identity(batch_axes=(self.smpl_helper.num_joints,)).wxyz,
                bone_positions=self.j_tpose + self.offsets[i],
                skin_weights=self.smpl_helper.weights,
                wireframe=False,
                color=self.colors[i]
            )
            self.meshes.append(mesh)
        
        return self.meshes
    
    def update_mesh(self, mesh_idx: int, joint_data: np.ndarray, frame_idx: int):
        """
        Update a mesh's bone transformations for the current frame.
        
        Args:
            mesh_idx: Index of mesh to update (0=original, 1=reconstructed)
            joint_data: Joint transformation matrices
            frame_idx: Current frame index
        """
        if mesh_idx >= len(self.meshes):
            raise ValueError(f"Mesh index {mesh_idx} out of range, only have {len(self.meshes)} meshes")
            
        mesh = self.meshes[mesh_idx]
        offset = self.offsets[mesh_idx]
        
        # Extract rotation and position from transformation matrices
        wxyz = tf.SO3.from_matrix(joint_data[frame_idx, :, :3, :3]).wxyz
        positions = joint_data[frame_idx, :, :3, 3] + offset
        
        # Update each bone
        for i, bone in enumerate(mesh.bones):
            bone.wxyz = wxyz[i]
            bone.position = positions[i]
    
    def run_animation_loop(self, joint_data_list: List[np.ndarray], fps: int = 25):
        """
        Animate meshes at specified framerate.
        
        Args:
            joint_data_list: List of joint transformations for each mesh
            fps: Frames per second
        """
        # Ensure we have enough meshes
        if len(self.meshes) < len(joint_data_list):
            self.setup_visualization(len(joint_data_list))
        
        # Animation loop variables
        frame_idx = 0
        num_frames = min(data.shape[0] for data in joint_data_list)
        frame_time = 1.0 / fps
        
        print(f"Starting animation loop with {num_frames} frames at {fps} FPS")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                # Update all meshes
                for i, joint_data in enumerate(joint_data_list):
                    self.update_mesh(i, joint_data, frame_idx)
                
                # Move to next frame (loop back to start when done)
                frame_idx = (frame_idx + 1) % num_frames
                
                # Wait for next frame
                time.sleep(frame_time)
        except KeyboardInterrupt:
            print("\nAnimation stopped by user")


#------------------------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------------------------

def main():
    """Main function to visualize original and VAE-reconstructed SMPL human motion."""
    parser = argparse.ArgumentParser(description="Flexible Human Motion Visualization")
    
    # Input paths
    parser.add_argument("--config_path", type=str, default="outputs/convnext_solo_full/config/config.yaml",
                        help="Path to model config JSON file")
    parser.add_argument("--data_path", type=str, default="data/sns_slahmr_64/dancing_oKULwuO54bc_500_600.npz",
                        help="Path to motion data file (.npz)")
    parser.add_argument("--smpl_model", type=str, default="smpl_neutral.npz",
                        help="Path to SMPL model file")
    
    # Visualization options
    parser.add_argument("--fps", type=int, default=25,
                        help="Frames per second for animation")
    parser.add_argument("--person_idx", type=int, default=0,
                        help="Person index to visualize (for multi-person data)")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run model inference on")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load SMPL model
        smpl_helper = SmplHelper(Path(args.smpl_model))
        print(f"Loaded SMPL model with {smpl_helper.num_joints} joints")
        
        # Load model and dataset
        device = torch.device(args.device)
        model = load_model(config["model"]["name"], config)
        model = model.to(device)
        model.eval()
        
        dataset_class = get_dataset(config["dataset"]["name"], 
                                  data_dir=str(Path(args.data_path).parent),
                                  person_idx=args.person_idx)
        
        # Load the specific data file and get ground truth data
        data_path = Path(args.data_path)
        gt_data, transformed_data = dataset_class.get_gt_data(data_path)
        
        print(f"Loaded ground truth data from {data_path}")
        print(f"Data shapes:")
        print(f"  Translation: {gt_data['trans'].shape}")
        print(f"  Root orientation: {gt_data['root_orient'].shape}")
        print(f"  Pose body: {gt_data['pose_body'].shape}")
        
        # Pass through VAE for reconstruction
        with torch.no_grad():
            transformed_data = transformed_data.unsqueeze(0).to(device)  # Add batch dimension
            reconstructed_transformed, _, _ = model(transformed_data)
            reconstructed_transformed = reconstructed_transformed.squeeze(0).cpu()  # Remove batch dimension
        
        # Invert the transformation to get reconstructed motion
        recon_data = dataset_class.inverse_transformation(reconstructed_transformed)
        
        print("Motion reconstruction complete")
        
        # Prepare joint data for both original and reconstructed motion
        gt_joint_data = prepare_joint_data(gt_data, smpl_helper)
        recon_joint_data = prepare_joint_data(recon_data, smpl_helper)
        
        print(f"Joint data shapes:")
        print(f"  Ground truth: {gt_joint_data.shape}")
        print(f"  Reconstructed: {recon_joint_data.shape}")
        
        # Compute reconstruction error
        error = torch.norm(torch.tensor(gt_joint_data) - torch.tensor(recon_joint_data))
        print(f"Reconstruction error: {error:.4f}")
        
        # Set up visualization and run animation
        visualizer = MotionVisualizer(smpl_helper)
        visualizer.setup_visualization(num_people=2)
        print(f"Visualization server started at http://localhost:8080")
        
        visualizer.run_animation_loop([gt_joint_data, recon_joint_data], fps=args.fps)
        
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())