import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils.rotation_transformations import (
    transform_relative_orientation,
    transform_relative_translation,
    axis_angle_to_matrix,
    matrix_to_6d
)

#------------------------------------------------------------------------------------------------
# Translation Dataset
#------------------------------------------------------------------------------------------------

class SoloTranslationDataset(Dataset):
    def __init__(self, data_dir: str, person_idx: int = 0):
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
    
        # Calculate relative translations for single person
        trans = data['trans'][self.person_idx]
        relative_trans = transform_relative_translation(trans)

        # Create mask for single person
        if 'track_mask' in data:
            mask = data['track_mask'][self.person_idx]
        else:
            mask = np.ones_like(relative_trans)
        
        return torch.tensor(relative_trans, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)


#------------------------------------------------------------------------------------------------
# Orientation Dataset
#------------------------------------------------------------------------------------------------

class SoloOrientationDataset(Dataset):
    def __init__(self, data_dir: str, person_idx: int = 0):
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
        
        # Calculate relative orientations for single person
        root_orient = data['root_orient'][self.person_idx]  # Extract all orientations
        relative_orient = transform_relative_orientation(root_orient)
        
        # Return only the orientation data, not body pose
        motion = relative_orient  # Just return the orientation data
        
        # Create mask for single person
        if 'track_mask' in data:
            motion_mask = data['track_mask'][self.person_idx]
        else:
            motion_mask = np.ones_like(motion)
        
        return torch.tensor(motion, dtype=torch.float32), torch.tensor(motion_mask, dtype=torch.bool)


#------------------------------------------------------------------------------------------------
# Pose Dataset
#------------------------------------------------------------------------------------------------

class SoloPoseDataset(Dataset):
    def __init__(self, data_dir: str, person_idx: int = 0):
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
        
        pose = data['pose_body'][self.person_idx]
        
        # Create mask for single person
        if 'track_mask' in data:
            mask = data['track_mask'][self.person_idx]
        else:
            mask = np.ones_like(pose)
        
        return torch.tensor(pose, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

class SoloPose6DDataset(Dataset):
    """
    Dataset for loading SMPL pose parameters and converting joint rotations 
    from axis-angle (3D) to continuous 6D representation.
    """
    def __init__(self, data_dir: str, person_idx: int = 0):
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
        
        # Load pose in axis-angle format
        pose = data['pose_body'][self.person_idx]  # Shape: [T, J*3]
        
        # Create mask for single person
        if 'track_mask' in data:
            mask = data['track_mask'][self.person_idx]
        else:
            mask = np.ones_like(pose[:, 0:1])
        
        # Convert pose to PyTorch tensor
        pose_tensor = torch.tensor(pose, dtype=torch.float32)
        
        # Get dimensions
        seq_len = pose_tensor.shape[0]
        total_rotation_dims = pose_tensor.shape[1]
        num_joints = total_rotation_dims // 3
        
        # Reshape to [T*J, 3] to process all rotations at once
        pose_reshaped = pose_tensor.reshape(-1, 3)
        
        # Convert all axis-angles to rotation matrices in one batch operation
        rotation_matrices = axis_angle_to_matrix(pose_reshaped)  # [T*J, 3, 3]
        
        # Convert all rotation matrices to 6D representation in one operation
        pose_6d = matrix_to_6d(rotation_matrices)  # [T*J, 6]
        
        # Reshape back to [T, J*6]
        pose_6d = pose_6d.reshape(seq_len, num_joints * 6)
        
        return pose_6d, torch.tensor(mask, dtype=torch.bool)
    
    def get_original_pose(self, idx):
        """
        Get the original axis-angle pose without conversion.
        Useful for debugging or comparison.
        """
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
        pose = data['pose_body'][self.person_idx]
        return torch.tensor(pose, dtype=torch.float32)


#------------------------------------------------------------------------------------------------
# Full Motion Dataset
#------------------------------------------------------------------------------------------------

class SoloFullMotionDataset(Dataset):
    def __init__(self, data_dir: str, person_idx: int = 0):
        self.data_dir = Path(data_dir)
        self.file_paths = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.person_idx = person_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.data_dir / self.file_paths[idx]
        data = np.load(file_path)
        
        # Calculate relative translations for single person
        relative_trans = transform_relative_translation(data['trans'][self.person_idx])
        # Calculate relative orientations for single person
        root_orient = data['root_orient'][self.person_idx]  # Extract all orientations
        relative_orient = transform_relative_orientation(root_orient)
        
        # Use absolute pose values instead of relative differences
        pose_body = data['pose_body'][self.person_idx]  # Use absolute pose values
        
        # Concatenate for single person
        motion = np.concatenate([relative_trans, relative_orient, pose_body], axis=1)  # (100, 69)
        
        # Create mask for single person
        if 'track_mask' in data:
            motion_mask = data['track_mask'][self.person_idx, :, None] * np.ones((1, 69))  # (100, 69)
        else:
            motion_mask = np.ones_like(motion)
        
        return torch.tensor(motion, dtype=torch.float32), torch.tensor(motion_mask, dtype=torch.bool)
