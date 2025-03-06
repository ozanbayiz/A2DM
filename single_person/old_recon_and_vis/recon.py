#!/usr/bin/env python
"""
Motion Reconstruction

This script:
1. Loads motion data and VAE models
2. Reconstructs motion components using specialized VAEs
3. Evaluates reconstruction quality
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from models.convnext.vae import ConvNextVAE
from single_person.dataloader import inverse_transform_motion, axis_angle_to_matrix, matrix_to_axis_angle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motion_recon")

#---------------------------------------------------------------------------
# Data Structures
#---------------------------------------------------------------------------

class MotionData:
    """Container for motion data components."""
    
    def __init__(self, 
                 translation: np.ndarray, 
                 orientation: np.ndarray, 
                 pose_body: np.ndarray,
                 mask: Optional[np.ndarray] = None):
        """
        Args:
            translation: Translation component of shape (N, seq_len, 3)
            orientation: Orientation component of shape (N, seq_len, 3)
            pose_body: Body pose component of shape (N, seq_len, 63)
            mask: Optional mask of shape (N, seq_len)
        """
        self.translation = translation
        self.orientation = orientation
        self.pose_body = pose_body
        self.mask = mask if mask is not None else np.ones((translation.shape[0], translation.shape[1]))
        
        # Validate dimensions - no longer require exactly 2 persons
        person_count = translation.shape[0]
        assert orientation.shape[0] == person_count, f"Expected {person_count} persons in orientation data"
        assert pose_body.shape[0] == person_count, f"Expected {person_count} persons in pose body data"
        if mask is not None:
            assert mask.shape[0] == person_count, f"Expected {person_count} persons in mask data"
        
        # Store sequence length and person count
        self.seq_len = translation.shape[1]
        self.person_count = person_count
    
    @classmethod
    def from_npz(cls, file_path: str) -> 'MotionData':
        """Load motion data from npz file."""
        logger.info(f"Loading motion data from {file_path}")
        data = np.load(file_path)
        
        translation = data['trans']  
        orientation = data['root_orient']  
        pose_body = data['pose_body']
        mask = data.get('track_mask', np.ones((2, translation.shape[1])))
        
        return cls(translation, orientation, pose_body, mask)

#---------------------------------------------------------------------------
# SMPL Helper
#---------------------------------------------------------------------------

class SmplHelper:
    """Helper for SMPL models (numpy version, without blend skinning)."""
    
    def __init__(self, model_path: Path) -> None:
        """
        Args:
            model_path: Path to SMPL model file (.npz)
        """
        assert model_path.suffix.lower() == ".npz", "Model should be an .npz file!"
        body_dict = dict(**np.load(model_path, allow_pickle=True))
        
        # Load SMPL model parameters
        self.J_regressor = body_dict["J_regressor"]
        self.weights = body_dict["weights"]
        self.v_template = body_dict["v_template"]
        self.posedirs = body_dict["posedirs"]
        self.shapedirs = body_dict["shapedirs"]
        self.faces = body_dict["f"]
        self.num_joints: int = self.weights.shape[-1]
        self.num_betas: int = self.shapedirs.shape[-1]
        self.parent_idx: np.ndarray = body_dict["kintree_table"][0]

    def get_tpose(self, betas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get T-pose vertices and joints for given shape parameters.
        
        Args:
            betas: Shape parameters of size (num_betas,)
            
        Returns:
            Tuple of vertices and joints in T-pose
        """
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)
        return v_tpose, j_tpose

    def get_all_time_outputs(self, 
                             betas: np.ndarray, 
                             joint_rotmats: np.ndarray, 
                             translation: np.ndarray) -> np.ndarray:
        """
        Compute joint transformations for all frames.
        
        Args:
            betas: Shape parameters
            joint_rotmats: Joint rotation matrices either as (T, num_joints, 3, 3) or flattened
                          as (T, num_joints*D) where D is either 3 (axis-angle) or 9 (flattened 3x3 matrix)
            translation: Translation of shape (T, 3)
            
        Returns:
            Joint transformation matrices of shape (T, num_joints, 4, 4)
        """
        v_tpose, j_tpose = self.get_tpose(betas)
        T = joint_rotmats.shape[0]
        
        # Handle flattened joint_rotmats by reshaping
        if len(joint_rotmats.shape) == 2:
            # Special case for (1, 66) shape that appears in the codebase
            if joint_rotmats.shape[1] == 66:  # Likely 22 joints with 3 parameters each
                # Assuming these are axis-angle representations for 22 joints (excluding root)
                # We need to interpret them correctly and reshape
                print(f"Converting joint_rotmats from shape {joint_rotmats.shape} to (T, {self.num_joints}, 3, 3)")
                
                # Debug - show what the data looks like for better understanding
                print(f"joint_rotmats first few values: {joint_rotmats[0, :10]}")
                
                # Option 1: Assume these are flattened 66 = 22 x 3 (axis-angle per joint) 
                # and the root joint is handled separately
                # First, prepare a full array with identity matrices
                rotmats = np.tile(np.eye(3), (T, self.num_joints, 1, 1))
                
                # Handle the 66 parameters (22 joints x 3 axis-angle params)
                # Reshape from (T, 66) to (T, 22, 3)
                flattened_params = joint_rotmats.reshape(T, -1, 3)
                num_joints_in_data = flattened_params.shape[1]
                
                # Start at joint index 1 (assuming index 0 is root)
                for i in range(min(num_joints_in_data, self.num_joints - 1)):
                    joint_idx = i + 1  # Offset by 1 to skip root joint
                    for t in range(T):
                        # Get the axis-angle representation
                        axis_angle = flattened_params[t, i]
                        angle = np.linalg.norm(axis_angle)
                        if angle < 1e-6:
                            # Almost zero rotation, use identity
                            rotmats[t, joint_idx] = np.eye(3)
                        else:
                            # Convert axis-angle to rotation matrix
                            axis = axis_angle / angle
                            c = np.cos(angle)
                            s = np.sin(angle)
                            x, y, z = axis
                            rotmats[t, joint_idx] = np.array([
                                [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                                [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                                [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
                            ])
                
                joint_rotmats = rotmats
            # Check if we have a flattened rotation matrix representation
            elif joint_rotmats.shape[1] == self.num_joints * 9:  # flattened 3x3 matrices
                joint_rotmats = joint_rotmats.reshape(T, self.num_joints, 3, 3)
            elif joint_rotmats.shape[1] == self.num_joints * 3:  # flattened axis-angle
                # Convert axis-angle to rotation matrices
                flattened = joint_rotmats.reshape(T * self.num_joints, 3)
                rotmats = np.zeros((T * self.num_joints, 3, 3))
                for i in range(T * self.num_joints):
                    angle = np.linalg.norm(flattened[i])
                    if angle < 1e-6:
                        rotmats[i] = np.eye(3)
                    else:
                        axis = flattened[i] / angle
                        c = np.cos(angle)
                        s = np.sin(angle)
                        x, y, z = axis
                        rotmats[i] = np.array([
                            [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                            [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
                        ])
                joint_rotmats = rotmats.reshape(T, self.num_joints, 3, 3)
            else:
                raise ValueError(f"Cannot handle joint_rotmats with shape {joint_rotmats.shape}. "
                                 f"Expected either (T, {self.num_joints}, 3, 3), "
                                 f"(T, {self.num_joints*9}), (T, {self.num_joints*3}) or the special case (T, 66).")
        
        # Initialize parent joint transforms
        T_parent_joint = np.tile(np.eye(4)[None, None, ...], (T, self.num_joints, 1, 1))
        T_parent_joint[:, :, :3, :3] = joint_rotmats
        T_parent_joint[:, 0, :3, 3] = j_tpose[0]
        T_parent_joint[:, 1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]
        
        # Compute world joint transforms
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[:, i] = T_world_joint[:, self.parent_idx[i]] @ T_parent_joint[:, i]
        
        # Apply root translation
        T_translation = np.tile(np.eye(4)[None, ...], (T, 1, 1))
        T_translation[:, :3, 3] = translation
        T_world_joint = T_translation[:, None, :, :] @ T_world_joint
        
        return T_world_joint

#---------------------------------------------------------------------------
# Data Processors
#---------------------------------------------------------------------------

class MotionProcessor:
    """Process motion data for VAE input and output."""
    
    @staticmethod
    def process_component(data: np.ndarray, mask: Optional[np.ndarray] = None, 
                          feature_dim: int = 3) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process motion component (translation, orientation, or pose) for VAE input.
        
        Args:
            data: Input data of shape (N, seq_len, feature_dim)
            mask: Optional mask of shape (N, seq_len)
            feature_dim: Feature dimension (3 for trans/orient, 63 for pose)
            
        Returns:
            Tuple of (processed inputs, masks) for each person
        """
        # Calculate relative values (deltas between frames)
        rel_data = data[:, 1:] - data[:, :-1]
        rel_data = np.concatenate([
            np.zeros((data.shape[0], 1, feature_dim), dtype=data.dtype), 
            rel_data
        ], axis=1)
        
        # Process each person separately
        person_inputs = []
        person_masks = []
        
        for person_idx in range(data.shape[0]):
            # Reshape to match VAE input
            person_input = rel_data[person_idx].reshape(data.shape[1], feature_dim)
            person_inputs.append(person_input.astype(np.float32))
            
            # Create or use mask
            if mask is not None:
                person_mask = mask[person_idx][..., None] * np.ones((1, feature_dim))
            else:
                person_mask = np.ones_like(person_input)
                    
            person_masks.append(person_mask.astype(np.float32))
            
        return person_inputs, person_masks
    
    @staticmethod
    def inverse_transform(component_recon: np.ndarray, 
                          init_values: np.ndarray, 
                          mask: Optional[np.ndarray] = None, 
                          dim: int = 3) -> np.ndarray:
        """
        Invert the transformation to recover absolute values from relative ones.
        
        Args:
            component_recon: Reconstructed component tensor (seq_len, dim)
            init_values: Initial values (1, dim)
            mask: Optional mask
            dim: Component dimension (3 for trans/orient, 63 for pose)
        
        Returns:
            Reconstructed absolute values (seq_len, dim)
        """
        # Apply mask if provided
        if mask is not None:
            component_recon = component_recon * mask
        
        # Initialize output with first frame values
        output = np.zeros_like(component_recon)
        output[0] = init_values
        
        # Accumulate deltas
        for i in range(1, len(component_recon)):
            output[i] = output[i-1] + component_recon[i]
        
        return output
    
    @classmethod
    def prepare_input_data(cls, motion_data: MotionData) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Prepare all input data components for VAE processing.
        
        Args:
            motion_data: Motion data object
            
        Returns:
            Dictionary with processed inputs and masks for each component
        """
        trans_input, trans_mask = cls.process_component(
            motion_data.translation, motion_data.mask, feature_dim=3)
        
        orient_input, orient_mask = cls.process_component(
            motion_data.orientation, motion_data.mask, feature_dim=3)
        
        pose_input, pose_mask = cls.process_component(
            motion_data.pose_body, motion_data.mask, feature_dim=63)
        
        return {
            'translation': (trans_input, trans_mask),
            'orientation': (orient_input, orient_mask),
            'pose': (pose_input, pose_mask)
        }

#---------------------------------------------------------------------------
# VAE Model Loading and Inference
#---------------------------------------------------------------------------

class ModelLoader:
    """Handles loading and configuring VAE models from checkpoints."""
    
    @staticmethod
    def load_vae_model(checkpoint_path: str, model_type: str) -> Optional[nn.Module]:
        """
        Load a VAE model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            model_type: Type of model ('translation', 'orientation', or 'pose')
        
        Returns:
            Loaded model or None if loading failed
        """
        logger.info(f"Loading {model_type} VAE model from {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            # Extract state_dict based on checkpoint structure
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to get model parameters from checkpoint config
            if 'config' in checkpoint and 'model_params' in checkpoint['config']:
                logger.info(f"Using model configuration from checkpoint")
                model_params = checkpoint['config']['model_params']
                
                # Create the model with parameters from checkpoint
                model = ConvNextVAE(
                    input_dim=model_params.get('input_dim', 3),
                    width=model_params.get('width', 128),
                    depth=model_params.get('depth', 3),
                    encoder_output_channels=model_params.get('encoder_output_channels', 128),
                    latent_dim=model_params.get('latent_dim', 64),
                    T_in=model_params.get('T_in', 100),
                    down_t=model_params.get('down_t', 5),
                    stride_t=model_params.get('stride_t', 2)
                )
            else:
                # Fall back to inferring parameters from state_dict
                logger.info(f"No configuration found in checkpoint, inferring parameters from state_dict")
                input_dim = ModelLoader._get_input_dim(state_dict, model_type)
                latent_dim, flattened_size = ModelLoader._get_latent_params(state_dict, model_type)
                
                # Setup temporal dimension parameters
                T_in = 100  # Default sequence length
                down_t = 5  # Default downsampling steps
                stride_t = 2  # Default stride
                
                # Calculate T_out and encoder output channels
                T_out = ModelLoader._calculate_T_out(T_in, down_t, stride_t)
                encoder_output_channels = flattened_size // T_out
                
                # Create the model with inferred parameters
                model = ConvNextVAE(
                    input_dim=input_dim,
                    width=128,  # Width from checkpoint
                    depth=3,    # Default depth
                    encoder_output_channels=encoder_output_channels,
                    latent_dim=latent_dim,
                    T_in=T_in,
                    down_t=down_t,
                    stride_t=stride_t
                )
                
                # Configure linear layers with correct dimensions
                device = next(iter(state_dict.values())).device
                ModelLoader._configure_linear_layers(model, flattened_size, latent_dim, device)
            
            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            # Set model to evaluation mode
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} VAE model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _get_input_dim(state_dict: Dict[str, torch.Tensor], model_type: str) -> int:
        """Determine input dimension from state dict or default values."""
        if 'encoder.model.0.weight' in state_dict:
            return state_dict['encoder.model.0.weight'].shape[1]
        else:
            # Default values if we can't determine from state_dict
            return 3 if model_type in ['translation', 'orientation'] else 63
    
    @staticmethod
    def _get_latent_params(state_dict: Dict[str, torch.Tensor], model_type: str) -> Tuple[int, int]:
        """Determine latent dimension and flattened size from state dict or defaults."""
        if 'fc_mu.weight' in state_dict:
            latent_dim = state_dict['fc_mu.weight'].shape[0]
            flattened_size = state_dict['fc_mu.weight'].shape[1]
            return latent_dim, flattened_size
        else:
            # Default values
            latent_dim = 64 if model_type == 'translation' else (96 if model_type == 'orientation' else 128)
            return latent_dim, 384  # Default flattened_size
    
    @staticmethod
    def _calculate_T_out(T_in: int, down_t: int, stride_t: int) -> int:
        """Calculate temporal dimension after downsampling."""
        T_out = T_in
        for _ in range(down_t):
            filter_t, stride, pad_t = stride_t * 2, stride_t, stride_t // 2
            T_out = ((T_out + 2*pad_t - filter_t) // stride) + 1
        return T_out
    
    @staticmethod
    def _configure_linear_layers(model: ConvNextVAE, flattened_size: int, 
                                latent_dim: int, device: torch.device) -> None:
        """Configure the linear layers with correct dimensions."""
        model.flattened_size = flattened_size
        model.fc_mu = nn.Linear(flattened_size, latent_dim).to(device)
        model.fc_logvar = nn.Linear(flattened_size, latent_dim).to(device)
        model.fc_decode = nn.Linear(latent_dim, flattened_size).to(device)

#---------------------------------------------------------------------------
# Evaluation
#---------------------------------------------------------------------------

class MotionEvaluator:
    """Evaluate motion reconstruction quality."""
    
    @staticmethod
    def calculate_errors(original: np.ndarray, 
                         recon: np.ndarray, 
                         mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate error metrics between original and reconstructed motion.
        
        Args:
            original: Original motion data
            recon: Reconstructed motion data
            mask: Optional mask for masked evaluation
            
        Returns:
            Dictionary with error metrics
        """
        if mask is not None:
            # Reshape mask for proper broadcasting
            if len(mask.shape) == 1 and len(original.shape) > 1:
                # Expand mask to match dimensions of data
                mask_expanded = mask.reshape(mask.shape[0], *([1] * (len(original.shape) - 1)))
                mse = np.sum(((recon - original) ** 2) * mask_expanded) / (np.sum(mask) + 1e-8)
            else:
                mse = np.sum(((recon - original) ** 2) * mask) / (np.sum(mask) + 1e-8)
        else:
            mse = np.mean((recon - original) ** 2)
        
        l2 = np.linalg.norm(recon - original)
        
        return {'mse': mse, 'l2': l2}

    @staticmethod
    def print_errors(errors: Dict[str, Dict[str, float]]) -> None:
        """
        Print formatted error metrics.
        
        Args:
            errors: Dictionary with error metrics for each component
        """
        print("\nReconstruction errors:")
        
        components = list(errors.keys())
        metrics = ['mse', 'l2']
        
        # Print individual component errors
        for component in components:
            print(f"  {component.capitalize()}:")
            for metric in metrics:
                print(f"    {metric.upper()}: {errors[component][metric]:.4f}")
        
        # Print total errors
        print("  Total:")
        for metric in metrics:
            total = sum(errors[component][metric] for component in components)
            print(f"    {metric.upper()}: {total:.4f}")

#---------------------------------------------------------------------------
# Motion Reconstruction
#---------------------------------------------------------------------------

class MotionReconstructor:
    """Reconstruct motion using trained VAE models."""
    
    def __init__(self, 
                translation_model: Optional[nn.Module], 
                orientation_model: Optional[nn.Module], 
                pose_model: Optional[nn.Module],
                device: torch.device):
        """
        Args:
            translation_model: VAE model for translation
            orientation_model: VAE model for orientation
            pose_model: VAE model for pose
            device: Device to run models on
        """
        self.translation_model = translation_model
        self.orientation_model = orientation_model
        self.pose_model = pose_model
        self.device = device
    
    def get_model_prediction(self, model: nn.Module, input_data: np.ndarray) -> np.ndarray:
        """Get direct prediction from a model without additional processing"""
        model.eval()
        with torch.no_grad():
            # Convert input to tensor and move to device
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            
            # Check if the input tensor has the correct temporal dimension
            if hasattr(model, 'T_in') and input_tensor.shape[0] != model.T_in:
                # If we have fewer frames than required, pad with zeros to match T_in
                if input_tensor.shape[0] < model.T_in:
                    padding_size = model.T_in - input_tensor.shape[0]
                    padding = torch.zeros((padding_size, *input_tensor.shape[1:]), 
                                         dtype=input_tensor.dtype, 
                                         device=input_tensor.device)
                    input_tensor = torch.cat([input_tensor, padding], dim=0)
                    print(f"Input padded from {input_data.shape[0]} to {model.T_in} frames")
                # If we have more frames than required, truncate
                else:
                    input_tensor = input_tensor[:model.T_in]
                    print(f"Input truncated from {input_data.shape[0]} to {model.T_in} frames")
            
            # Add batch dimension if needed
            if len(input_tensor.shape) == 2:  # (T, features)
                input_tensor = input_tensor.unsqueeze(0)  # (1, T, features)
            
            # Get model outputs
            recon, mu, logvar = model(input_tensor)
            
            # Return reconstruction as numpy array (removing batch dimension if it was added)
            return recon.squeeze(0).cpu().numpy()
    
    def reconstruct_person_motion(self, motion_data: MotionData, person_idx: int = 0) -> Dict[str, np.ndarray]:
        """Reconstruct motion using component models and inverse transformation"""
        result = {}
        
        # Get initial values for inverse transform
        init_trans = motion_data.translation[person_idx][0] if motion_data.translation is not None else None
        init_orient = motion_data.orientation[person_idx][0] if motion_data.orientation is not None else None
        
        # Get mask
        mask = motion_data.mask[person_idx] if motion_data.mask is not None else None
        
        # Process and reconstruct translation if model exists
        if self.translation_model and motion_data.translation is not None:
            # Get relative translation by computing differences
            rel_trans = np.zeros_like(motion_data.translation[person_idx])
            rel_trans[1:] = motion_data.translation[person_idx][1:] - motion_data.translation[person_idx][:-1]
            
            # Get model prediction
            trans_recon = self.get_model_prediction(self.translation_model, rel_trans)
            result["rel_translation"] = trans_recon
        else:
            trans_recon = None
            
        # Process and reconstruct orientation if model exists
        if self.orientation_model and motion_data.orientation is not None:
            # Calculate relative orientations
            rel_orient = np.zeros_like(motion_data.orientation[person_idx])
            orient_data = motion_data.orientation[person_idx]
            
            # Calculate proper SO(3) relative orientations
            for i in range(1, len(orient_data)):
                prev_R = axis_angle_to_matrix(orient_data[i-1])
                curr_R = axis_angle_to_matrix(orient_data[i])
                R_rel = curr_R @ prev_R.T
                rel_orient[i] = matrix_to_axis_angle(R_rel)
            
            # Get model prediction
            orient_recon = self.get_model_prediction(self.orientation_model, rel_orient)
            result["rel_orientation"] = orient_recon
        else:
            orient_recon = None
            
        # Process and reconstruct body pose if model exists
        if self.pose_model and motion_data.pose_body is not None:
            # Use absolute pose values directly
            pose_recon = self.get_model_prediction(self.pose_model, motion_data.pose_body[person_idx])
            result["pose_body"] = pose_recon
        else:
            pose_recon = None
        
        # Use inverse_transform_motion to recover absolute values
        if trans_recon is not None and orient_recon is not None:
            # Create combined motion array for inverse transform
            combined_motion = np.concatenate([
                trans_recon, 
                orient_recon, 
                pose_recon if pose_recon is not None else np.zeros((trans_recon.shape[0], 63))
            ], axis=1)
            
            # Apply inverse transform from dataloader
            trans, orient, pose = inverse_transform_motion(
                combined_motion, init_trans, init_orient, 
                motion_mask=None if mask is None else mask.reshape(-1, 1).repeat(combined_motion.shape[1], axis=1)
            )
            
            result["translation"] = trans
            result["orientation"] = orient
            if pose_recon is not None:
                result["pose_body"] = pose
        else:
            # Handle partial reconstruction
            if trans_recon is not None:
                # Only translation was reconstructed
                trans = np.zeros_like(motion_data.translation[person_idx])
                trans[0] = init_trans
                for i in range(1, len(trans_recon)):
                    trans[i] = trans[i-1] + trans_recon[i]
                result["translation"] = trans
                
            if orient_recon is not None:
                # Only orientation was reconstructed
                orient = np.zeros_like(motion_data.orientation[person_idx])
                orient[0] = init_orient
                R_abs = axis_angle_to_matrix(init_orient)
                for i in range(1, len(orient_recon)):
                    R_rel = axis_angle_to_matrix(orient_recon[i])
                    R_abs = R_rel @ R_abs
                    orient[i] = matrix_to_axis_angle(R_abs)
                result["orientation"] = orient
        
        return result

#---------------------------------------------------------------------------
# Main Application for Reconstruction
#---------------------------------------------------------------------------

class MotionReconApp:
    """Main application for motion reconstruction."""
    
    def __init__(self):
        """Initialize the application."""
        self.args = self.parse_arguments()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load SMPL helper
        self.smpl_helper = SmplHelper(Path(self.args.smpl_model))
        
        # Load models and create motion reconstructor
        translation_vae, orientation_vae, pose_vae = self.load_models()
        self.reconstructor = MotionReconstructor(
            translation_vae, orientation_vae, pose_vae, self.device
        )
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Motion Reconstruction")
        parser.add_argument("--input_file", type=str, required=True, 
                          help="Path to raw npz input file with motion data")
        parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", 
                          help="Directory containing VAE checkpoints")
        parser.add_argument("--smpl_model", type=str, default="smpl/smpl_neutral.npz", 
                          help="Path to SMPL model file")
        parser.add_argument("--save_recon", action="store_true", 
                          help="Save the reconstructed motion to a file")
        parser.add_argument("--output_dir", type=str, default=".", 
                          help="Directory to save reconstructed motion")
        parser.add_argument("--person_idx", type=int, default=0, 
                          help="Person index to reconstruct (default: 0)")
        return parser.parse_args()
    
    def load_models(self) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
        """Load VAE models from checkpoints."""
        logger.info("Loading VAE models...")
        
        # VAE checkpoint paths
        translation_checkpoint = os.path.join(
            self.args.checkpoints_dir, "translation_vae", "translation_vae_best.pt")
        orientation_checkpoint = os.path.join(
            self.args.checkpoints_dir, "orientation_vae", "orientation_vae_best.pt")
        pose_checkpoint = os.path.join(
            self.args.checkpoints_dir, "pose_vae", "pose_vae_best.pt")
        
        # Load models
        translation_vae = ModelLoader.load_vae_model(translation_checkpoint, 'translation')
        orientation_vae = ModelLoader.load_vae_model(orientation_checkpoint, 'orientation')
        pose_vae = ModelLoader.load_vae_model(pose_checkpoint, 'pose')
        
        return translation_vae, orientation_vae, pose_vae
    
    def save_reconstruction(self, reconstructed: Dict[str, np.ndarray]) -> None:
        """Save reconstructed motion to file."""
        if not self.args.save_recon:
            return
        
        input_filename = os.path.basename(self.args.input_file)
        recon_filename = os.path.join(self.args.output_dir, f"reconstructed_{input_filename}")
        
        # Ensure the data is in the correct format for saving
        trans = reconstructed['translation']
        root_orient = reconstructed['orientation']
        pose_body = reconstructed['pose_body']
        
        # Check if we need to expand to a 2-person format
        if len(trans.shape) == 2:  # Single person format
            trans = np.expand_dims(trans, axis=0)
            root_orient = np.expand_dims(root_orient, axis=0)
            pose_body = np.expand_dims(pose_body, axis=0)
            
            # Create a two-person array (with second person being zeros)
            trans = np.vstack([trans, np.zeros_like(trans)])
            root_orient = np.vstack([root_orient, np.zeros_like(root_orient)])
            pose_body = np.vstack([pose_body, np.zeros_like(pose_body)])
        
        np.savez(recon_filename, 
                trans=trans,
                root_orient=root_orient,
                pose_body=pose_body)
                
        logger.info(f"Saved reconstructed motion to {recon_filename}")
    
    def run(self) -> None:
        """Run motion reconstruction without visualization."""
        print(f"Loading file: {self.args.input_file}")
        person_motion_data = MotionData.from_npz(self.args.input_file)
        
        if self.args.person_idx is not None:
            print(f"Using person {self.args.person_idx}")
        else:
            print("Using first available person")
        
        person_idx = self.args.person_idx if self.args.person_idx is not None else 0
        
        # Perform reconstruction for specified person
        reconstructed = self.reconstructor.reconstruct_person_motion(
            person_motion_data, person_idx
        )
        
        # Save reconstruction if output file is specified
        if self.args.save_recon:
            self.save_reconstruction(reconstructed)
        
        # Evaluate reconstruction
        errors = {}
        
        if self.reconstructor.translation_model and "translation" in reconstructed:
            errors["translation"] = MotionEvaluator.calculate_errors(
                person_motion_data.translation[person_idx], 
                reconstructed["translation"],
                person_motion_data.mask[person_idx] if person_motion_data.mask is not None else None
            )
        
        if self.reconstructor.orientation_model and "orientation" in reconstructed:
            errors["orientation"] = MotionEvaluator.calculate_errors(
                person_motion_data.orientation[person_idx], 
                reconstructed["orientation"],
                person_motion_data.mask[person_idx] if person_motion_data.mask is not None else None
            )
        
        if self.reconstructor.pose_model and "pose_body" in reconstructed:
            errors["pose_body"] = MotionEvaluator.calculate_errors(
                person_motion_data.pose_body[person_idx], 
                reconstructed["pose_body"],
                person_motion_data.mask[person_idx] if person_motion_data.mask is not None else None
            )
        
        MotionEvaluator.print_errors(errors)
        
        # Return the reconstructed data
        return person_motion_data, reconstructed, person_idx

#---------------------------------------------------------------------------
# Entry Point
#---------------------------------------------------------------------------

def main():
    """Entry point for the application."""
    app = MotionReconApp()
    app.run()

if __name__ == "__main__":
    main() 