import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Any
from utils.rotation_transformations import (
    transform_relative_orientation,
    inverse_transform_global_orientation,
    transform_relative_translation,
    inverse_transform_translation,
    axis_angle_to_matrix,
    matrix_to_6d,
    matrix_to_axis_angle,
    sixd_to_matrix
)

class SoloDatasetBase(Dataset):
    """Base class for single-person motion datasets with common functionality."""
    
    def __init__(self, data_dir: str, person_idx: int = 0):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing motion data files
            person_idx: Index of the person to use (default: 0)
        """
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def load_data(self, idx_or_path: Union[int, str]) -> Dict[str, np.ndarray]:
        """
        Load data from file.
        
        Args:
            idx_or_path: Either an index into file_paths or a direct file path
            
        Returns:
            Dictionary containing loaded data
        """
        if isinstance(idx_or_path, int):
            file_path = self.data_dir / self.file_paths[idx_or_path]
        else:
            file_path = Path(idx_or_path)
        
        return np.load(file_path)
    
    def get_mask(self, data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
        """
        Create a mask array for the data.
        
        Args:
            data: Loaded data dictionary
            ref_data: Reference data for mask shape
            
        Returns:
            Boolean mask array
        """
        if 'track_mask' in data:
            return data['track_mask'][self.person_idx]
        else:
            return np.ones_like(ref_data)
    
    def create_zero_motion(self, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Create a zero motion dictionary.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Dictionary with zero tensors for motion components
        """
        return {
            'trans': torch.zeros((seq_len, 3), dtype=torch.float32),
            'root_orient': torch.zeros((seq_len, 3), dtype=torch.float32),
            'pose_body': torch.zeros((seq_len, 63), dtype=torch.float32)
        }


class SoloTranslationDataset(SoloDatasetBase):
    """Dataset for translation (position) data only."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.load_data(idx)
        
        # Calculate relative translations for single person
        trans = data['trans'][self.person_idx]
        relative_trans = transform_relative_translation(trans)
        
        # Create mask
        mask = self.get_mask(data, relative_trans)
        
        return torch.tensor(relative_trans, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
    
    def get_gt_data(self, path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get ground truth data and its transformation.
        
        Args:
            path: Path to data file
            
        Returns:
            Tuple of (original data dict, transformed data tensor)
        """
        data = self.load_data(path)
        trans = data['trans'][self.person_idx]
        relative_trans = transform_relative_translation(trans)
        
        # Create result with zeros for unused components
        result = self.create_zero_motion(len(trans))
        result['trans'] = torch.tensor(trans, dtype=torch.float32)
        
        return result, torch.tensor(relative_trans, dtype=torch.float32)
    
    def inverse_transformation(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert transformed data back to original format.
        
        Args:
            data: Transformed data tensor
            
        Returns:
            Dictionary with reconstructed motion components
        """
        seq_len = data.shape[0]
        trans = torch.zeros((seq_len, 3), dtype=torch.float32)
        
        # Accumulate deltas to recover absolute translations
        for i in range(1, seq_len):
            trans[i] = trans[i-1] + data[i-1]
        
        result = self.create_zero_motion(seq_len)
        result['trans'] = trans
        return result


class SoloOrientationDataset(SoloDatasetBase):
    """Dataset for orientation (rotation) data only."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.load_data(idx)
        
        # Calculate relative orientations
        root_orient = data['root_orient'][self.person_idx]
        relative_orient = transform_relative_orientation(root_orient)
        
        # Create mask
        mask = self.get_mask(data, relative_orient)
        
        return torch.tensor(relative_orient, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
    
    def get_gt_data(self, path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data = self.load_data(path)
        root_orient = data['root_orient'][self.person_idx]
        relative_orient = transform_relative_orientation(root_orient)
        
        # Create result with zeros for unused components
        result = self.create_zero_motion(len(root_orient))
        result['root_orient'] = torch.tensor(root_orient, dtype=torch.float32)
        
        return result, torch.tensor(relative_orient, dtype=torch.float32)
    
    def inverse_transformation(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = data.shape[0]
        root_orient = torch.zeros((seq_len, 3), dtype=torch.float32)
        
        # First frame keeps its orientation
        root_orient[0] = data[0]
        
        # Accumulate relative orientations
        for i in range(1, seq_len):
            root_orient[i] = root_orient[i-1] + data[i]
        
        result = self.create_zero_motion(seq_len)
        result['root_orient'] = root_orient
        return result


class SoloPoseDataset(SoloDatasetBase):
    """Dataset for body pose data only."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.load_data(idx)
        
        pose = data['pose_body'][self.person_idx]
        mask = self.get_mask(data, pose)
        
        return torch.tensor(pose, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)
    
    def get_gt_data(self, path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data = self.load_data(path)
        pose_body = data['pose_body'][self.person_idx]
        
        # Create result with zeros for unused components
        result = self.create_zero_motion(len(pose_body))
        result['pose_body'] = torch.tensor(pose_body, dtype=torch.float32)
        
        return result, torch.tensor(pose_body, dtype=torch.float32)
    
    def inverse_transformation(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = data.shape[0]
        result = self.create_zero_motion(seq_len)
        result['pose_body'] = data
        return result


class SoloPose6DDataset(SoloDatasetBase):
    """
    Dataset for body pose data converted to 6D rotation representation.
    Converts joint rotations from axis-angle (3D) to continuous 6D representation.
    """
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.load_data(idx)
        
        # Load pose and convert to 6D representation
        pose = data['pose_body'][self.person_idx]
        pose_6d = self._convert_to_6d(pose)
        
        # Create mask (adjust for 6D representation)
        if 'track_mask' in data:
            mask = data['track_mask'][self.person_idx]
        else:
            mask = np.ones_like(pose[:, 0:1])
        
        return pose_6d, torch.tensor(mask, dtype=torch.bool)
    
    def _convert_to_6d(self, pose_data: np.ndarray) -> torch.Tensor:
        """
        Convert axis-angle pose data to 6D representation.
        
        Args:
            pose_data: Pose data in axis-angle format [T, J*3]
            
        Returns:
            Pose data in 6D representation [T, J*6]
        """
        pose_tensor = torch.tensor(pose_data, dtype=torch.float32)
        
        # Get dimensions
        seq_len = pose_tensor.shape[0]
        total_dims = pose_tensor.shape[1]
        num_joints = total_dims // 3
        
        # Reshape to process all rotations at once
        pose_reshaped = pose_tensor.reshape(-1, 3)
        
        # Convert to rotation matrices then to 6D
        rotation_matrices = axis_angle_to_matrix(pose_reshaped)
        pose_6d = matrix_to_6d(rotation_matrices)
        
        # Reshape back to sequence format
        return pose_6d.reshape(seq_len, num_joints * 6)
    
    def get_gt_data(self, path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data = self.load_data(path)
        pose = data['pose_body'][self.person_idx]
        
        # Create result with zeros for unused components
        result = self.create_zero_motion(len(pose))
        result['pose_body'] = torch.tensor(pose, dtype=torch.float32)
        
        # Convert to 6D representation
        transformed_data = self._convert_to_6d(pose)
        
        return result, transformed_data
    
    def inverse_transformation(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert 6D pose representation back to axis-angle"""
        seq_len = data.shape[0]
        
        # Reshape to process rotations
        pose_reshaped = data.reshape(-1, 6)
        rotation_matrices = sixd_to_matrix(pose_reshaped)
        pose = matrix_to_axis_angle(rotation_matrices)
        pose = pose.reshape(seq_len, -1)  # Reshape back to sequence format
        
        result = self.create_zero_motion(seq_len)
        result['pose_body'] = pose
        return result
    
    def get_original_pose(self, idx: int) -> torch.Tensor:
        """Get the original axis-angle pose without conversion."""
        data = self.load_data(idx)
        pose = data['pose_body'][self.person_idx]
        return torch.tensor(pose, dtype=torch.float32)


class SoloFullMotionDataset(SoloDatasetBase):
    """Dataset for complete motion data including translation, orientation, and pose."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.load_data(idx)
        
        # Get components and apply transformations
        relative_trans = transform_relative_translation(data['trans'][self.person_idx])
        root_orient = data['root_orient'][self.person_idx]
        relative_orient = transform_relative_orientation(root_orient)
        pose_body = data['pose_body'][self.person_idx]  # Use absolute pose values
        
        # Concatenate for single person
        motion = np.concatenate([relative_trans, relative_orient, pose_body], axis=1)
        
        # Create mask
        if 'track_mask' in data:
            motion_mask = data['track_mask'][self.person_idx, :, None] * np.ones((1, motion.shape[1]))
        else:
            motion_mask = np.ones_like(motion)
        
        return torch.tensor(motion, dtype=torch.float32), torch.tensor(motion_mask, dtype=torch.bool)
    
    def get_gt_data(self, path: str) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        data = self.load_data(path)
        
        trans = data['trans'][self.person_idx]
        root_orient = data['root_orient'][self.person_idx]
        pose_body = data['pose_body'][self.person_idx]
        
        # Calculate transformations
        relative_trans = transform_relative_translation(trans)
        relative_orient = transform_relative_orientation(root_orient)
        
        # Concatenate for output
        motion = np.concatenate([relative_trans, relative_orient, pose_body], axis=1)
        
        return {
            'trans': torch.tensor(trans, dtype=torch.float32),
            'root_orient': torch.tensor(root_orient, dtype=torch.float32),
            'pose_body': torch.tensor(pose_body, dtype=torch.float32)
        }, torch.tensor(motion, dtype=torch.float32)
    
    def inverse_transformation(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq_len = data.shape[0]
        
        # Split concatenated motion data
        relative_trans = data[:, :3]  # First 3 dims are relative translation
        relative_orient = data[:, 3:6]  # Next 3 dims are relative orientation
        pose_body = data[:, 6:]  # Rest is pose
        
        # Convert relative translations to absolute
        # Convert to numpy, apply inverse transform, then back to tensor
        relative_trans_np = relative_trans.cpu().numpy()
        init_trans = relative_trans_np[0:1]
        trans_np = inverse_transform_translation(relative_trans_np, init_trans)
        trans = torch.tensor(trans_np, dtype=torch.float32)
        
        # Convert relative orientations to absolute
        relative_orient_np = relative_orient.cpu().numpy()
        init_root_orient = relative_orient_np[0:1]  # Initial orientation
        root_orient_np = inverse_transform_global_orientation(relative_orient_np, init_root_orient)
        root_orient = torch.tensor(root_orient_np, dtype=torch.float32)
        
        return {
            'trans': trans,
            'root_orient': root_orient,
            'pose_body': pose_body
        }
