import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DuetMotionDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: path to the directory containing .npz files.
        Each file contains keys: 'trans' (2 x 100 x 3),
        'root_orient' (2 x 100 x 3), 'body_pose' (2 x 100 x 63).
        """
        self.data_files = glob.glob(os.path.join(data_dir, '*.npz'))
        print(f"Found {len(self.data_files)} files in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = np.load(file_path)
        # For each file, create motion for person 1 and person 2:
        # For person 1, concatenate features along the last dimension:
        relative_trans = data['trans'][:,1:] - data['trans'][:,:-1]
        relative_trans = np.concatenate([np.zeros((2, 1, 3)), relative_trans], axis=1)
        relative_orient = data['root_orient'][:,1:] - data['root_orient'][:,:-1]
        relative_orient = np.concatenate([np.zeros((2, 1, 3)), relative_orient], axis=1)
        
        motion1 = np.concatenate([
            relative_trans[0],       # (100, 3)
            relative_orient[0],      # (100, 3)
            data['pose_body'][0]     # (100, 63)
        ], axis=1)  # -> (100, 69)

        # For person 2:
        motion2 = np.concatenate([
            relative_trans[1],       # (100, 3)
            relative_orient[1],      # (100, 3)
            data['pose_body'][1]     # (100, 63)
        ], axis=1)  # -> (100, 69)

        # Concatenate the two persons' features along the feature dimension:
        motion = np.concatenate([motion1, motion2], axis=1)  # -> (100, 138)
        # Convert to float tensor
        motion = torch.tensor(motion, dtype=torch.float32)

        # Create motion mask
        motion_mask = data['track_mask'][..., None] * np.ones((1, 1, 69))  # (2, 100, 69)
        motion_mask = motion_mask.transpose(1, 0, 2)  # (100, 2, 69)
        motion_mask = motion_mask.reshape(100, 138)  # (100, 138)
        motion_mask = torch.tensor(motion_mask, dtype=torch.int8)

        return motion, motion_mask

def get_dataloader(data_dir, batch_size=32, shuffle=True, num_workers=0, sampler=None):
    dataset = DuetMotionDataset(data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        num_workers=num_workers,
        sampler=sampler
    )

# For testing purposes:
if __name__ == '__main__':
    data_dir = "./slahmr"
    loader = get_dataloader(data_dir, batch_size=64)
    for batch in loader:
        print("Batch shape:", batch.shape)  # Expected: (B, 100, 138)
        break
