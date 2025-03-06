import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
import os
import sys

# Get the absolute path to the project root (parent of single_person)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Only add to path if needed (this doesn't change the root path, just makes it visible to Python)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use standard imports
from utils import (
    transform_relative_orientation, 
    inverse_transform_global_orientation, 
    inverse_transform_translation, 
    transform_relative_translation
)
from models.convnext.vae import ConvNextVAE


class DecoupledVAE(nn.Module):
    """
    A class that encapsulates three separate VAEs for human motion reconstruction:
    - Translation VAE: for modeling relative translations in 3D space
    - Orientation VAE: for modeling relative root orientations in SO(3)
    - Pose VAE: for modeling body poses in SO(3)
    
    This class handles loading the models, performing forward passes,
    and inverting transformations to get the original motion representation.
    """
    
    def __init__(
        self, 
        translation_checkpoint_path: Union[str, Path],
        orientation_checkpoint_path: Union[str, Path],
        pose_checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None
    ):
        """
        Initialize the DecoupledVAE by loading three separate VAEs from checkpoint paths.
        
        Args:
            translation_checkpoint_path: Path to translation VAE checkpoint
            orientation_checkpoint_path: Path to orientation VAE checkpoint
            pose_checkpoint_path: Path to pose VAE checkpoint
            device: Device to run models on (default: cuda if available, else cpu)
        """
        super().__init__()
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the three VAE models
        print("Loading VAE models...")
        self.translation_vae = self._load_vae_model(translation_checkpoint_path, "translation_vae")
        self.orientation_vae = self._load_vae_model(orientation_checkpoint_path, "orientation_vae")
        self.pose_vae = self._load_vae_model(pose_checkpoint_path, "pose_vae")
        
        # Set models to evaluation mode
        if self.translation_vae:
            self.translation_vae.eval()
        if self.orientation_vae:
            self.orientation_vae.eval()
        if self.pose_vae:
            self.pose_vae.eval()
    
    def _load_vae_model(self, checkpoint_path: Union[str, Path], model_type: str) -> Optional[nn.Module]:
        """
        Load a VAE model from checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            model_type: Type of model ('translation_vae', 'orientation_vae', or 'pose_vae')
            
        Returns:
            Loaded VAE model or None if loading failed
        """
        print(f"Loading {model_type} model from {checkpoint_path}")
        
        try:
            # Load checkpoint directly
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Extract config from checkpoint - this has all the information needed
            model_params = checkpoint.get('config', {}).get('model_params', {})
            model = ConvNextVAE(**model_params)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Successfully loaded model weights for {model_type}")
            else:
                raise ValueError(f"No 'model_state_dict' found in checkpoint for {model_type}")
            
            return model
            
        except Exception as e:
            print(f"Error loading {model_type} model: {str(e)}")
            return None
    
    def reconstruct(
        self, 
        translation: np.ndarray,  # Shape: (100, 3)
        orientation: np.ndarray,  # Shape: (100, 3) (axis-angle)
        pose_body: np.ndarray,    # Shape: (100, 63) (axis-angle)
    ) -> Dict[str, np.ndarray]:
        """
        Reconstruct motion by passing it through the VAE models.
        
        Args:
            translation: Global translation trajectory
            orientation: Global root orientation in axis-angle
            pose_body: Body pose in axis-angle
            with_gradients: Whether to use gradient tracking
            
        Returns:
            Dict containing the reconstructed motion components
        """
        # Get initial values for inverse transformations
        init_trans = translation[0].copy()
        init_root_orient = orientation[0].copy()
        
        # Convert inputs to relative representation based on dataloader transformations
        rel_trans = transform_relative_translation(translation)
        rel_orient = transform_relative_orientation(orientation)
        
        # Get reconstructions for each component
        with torch.no_grad():
            # Reconstruct translation
            trans_tensor = torch.from_numpy(rel_trans).float().unsqueeze(0).to(self.device)
            trans_recon = self.translation_vae(trans_tensor)[0]
            trans_recon = trans_recon.squeeze(0).cpu().numpy()
            
            # Reconstruct orientation
            orient_tensor = torch.from_numpy(rel_orient).float().unsqueeze(0).to(self.device)
            orient_recon = self.orientation_vae(orient_tensor)[0]
            orient_recon = orient_recon.squeeze(0).cpu().numpy()
            
            # Reconstruct pose
            pose_tensor = torch.from_numpy(pose_body).float().unsqueeze(0).to(self.device)
            pose_recon = self.pose_vae(pose_tensor)[0]
            pose_recon = pose_recon.squeeze(0).cpu().numpy()
        
        # Apply inverse transformations using the dataloader functions
        abs_trans = inverse_transform_translation(trans_recon, init_trans)
        abs_orient = inverse_transform_global_orientation(orient_recon, init_root_orient)
        
        # Body pose is already in absolute format, so we can use it directly
        return {
            'trans': abs_trans,
            'root_orient': abs_orient,
            'pose_body': pose_recon
        }
    
    def compute_errors(
        self,
        original: Dict[str, np.ndarray],
        reconstructed: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        visualize: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute error metrics between original and reconstructed motion.
        
        Args:
            original: Dictionary containing original 'trans', 'root_orient', and 'pose_body'
            reconstructed: Dictionary containing reconstructed 'trans', 'root_orient', and 'pose_body'
            mask: Optional mask to apply to error calculation
            visualize: Whether to visualize errors over time (default: False)
            
        Returns:
            Dictionary of error metrics for each component and total
        """
        errors = {}
        
        # Ensure all components exist in both dictionaries
        components = ['trans', 'root_orient', 'pose_body']
        for comp in components:
            if comp not in original or comp not in reconstructed:
                raise ValueError(f"Component '{comp}' missing from input dictionaries")
        
        # Calculate errors for each component
        for comp in components:
            orig_data = original[comp]
            recon_data = reconstructed[comp]
            
            # Reshape if necessary (for pose_body)
            orig_data = orig_data.reshape(orig_data.shape[0], -1)
            recon_data = recon_data.reshape(recon_data.shape[0], -1)
            
            # Apply mask if provided
            if mask is not None:
                mask_bool = mask.astype(bool)
                orig_data = orig_data[mask_bool]
                recon_data = recon_data[mask_bool]
            
            # Calculate MSE
            mse = np.mean(np.square(orig_data - recon_data))
            
            # Calculate L2 norm (Euclidean distance)
            l2 = np.sqrt(np.sum(np.square(orig_data - recon_data)))
            
            # Calculate frame-by-frame errors for visualization
            frame_errors = np.sqrt(np.sum(np.square(orig_data - recon_data), axis=1))
            
            errors[comp] = {
                'mse': float(mse),
                'l2': float(l2),
                # 'frame_errors': frame_errors  # Store frame-by-frame errors
            }
        
        # Calculate total error
        total_mse = sum(errors[comp]['mse'] for comp in components)
        total_l2 = sum(errors[comp]['l2'] for comp in components)
        
        errors['total'] = {
            'mse': total_mse,
            'l2': total_l2
        }
        
        return errors
    
    def encode(
        self,
        trans: np.ndarray,
        root_orient: np.ndarray,
        pose_body: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Encode motion data into latent representations using each VAE.
        
        Args:
            trans: Translation data of shape (T, 3)
            root_orient: Root orientation data of shape (T, 3) in axis-angle
            pose_body: Body pose data of shape (T, 21, 3) in axis-angle
            
        Returns:
            Dict of latent codes for each motion component
        """
        # Check if all models are loaded
        if not all([self.translation_vae, self.orientation_vae, self.pose_vae]):
            raise ValueError("Not all VAE models were successfully loaded")
        
        # Convert inputs to relative representation based on dataloader transformations
        rel_trans = np.diff(trans, axis=0)
        rel_trans = np.concatenate([np.zeros((1, 3), dtype=rel_trans.dtype), rel_trans], axis=0)
        
        with torch.no_grad():
            # Encode translation
            trans_tensor = torch.from_numpy(rel_trans).float().unsqueeze(0).to(self.device)
            trans_mu, trans_logvar = self.translation_vae.encode(trans_tensor)
            trans_z = self.translation_vae.reparameterize(trans_mu, trans_logvar)
            
            # Encode orientation
            orient_tensor = torch.from_numpy(root_orient).float().unsqueeze(0).to(self.device)
            orient_mu, orient_logvar = self.orientation_vae.encode(orient_tensor)
            orient_z = self.orientation_vae.reparameterize(orient_mu, orient_logvar)
            
            # Encode pose
            pose_tensor = torch.from_numpy(pose_body.reshape(pose_body.shape[0], -1)).float().unsqueeze(0).to(self.device)
            pose_mu, pose_logvar = self.pose_vae.encode(pose_tensor)
            pose_z = self.pose_vae.reparameterize(pose_mu, pose_logvar)
        
        return {
            'trans': trans_z.squeeze(0).cpu().numpy(),
            'root_orient': orient_z.squeeze(0).cpu().numpy(),
            'pose_body': pose_z.squeeze(0).cpu().numpy()
        }
    
    def decode(
        self,
        latent_codes: Dict[str, np.ndarray],
        init_trans: np.ndarray,
        init_root_orient: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decode latent representations back to motion.
        
        Args:
            latent_codes: Dict of latent codes for each motion component
            init_trans: Initial translation value for inverse transform
            init_root_orient: Initial root orientation for inverse transform
            
        Returns:
            Dict containing the decoded motion components
        """
        # Check if all models are loaded
        if not all([self.translation_vae, self.orientation_vae, self.pose_vae]):
            raise ValueError("Not all VAE models were successfully loaded")
        
        # Check if all latent codes are provided
        components = ['trans', 'root_orient', 'pose_body']
        for comp in components:
            if comp not in latent_codes:
                raise ValueError(f"Latent code '{comp}' missing from input dictionary")
        
        with torch.no_grad():
            # Decode translation
            trans_z = torch.from_numpy(latent_codes['trans']).float().unsqueeze(0).to(self.device)
            trans_recon = self.translation_vae.decode(trans_z)
            trans_recon = trans_recon.squeeze(0).cpu().numpy()
            
            # Decode orientation
            orient_z = torch.from_numpy(latent_codes['root_orient']).float().unsqueeze(0).to(self.device)
            orient_recon = self.orientation_vae.decode(orient_z)
            orient_recon = orient_recon.squeeze(0).cpu().numpy()
            
            # Decode pose
            pose_z = torch.from_numpy(latent_codes['pose_body']).float().unsqueeze(0).to(self.device)
            pose_flat = self.pose_vae.decode(pose_z)
            pose_flat = pose_flat.squeeze(0).cpu().numpy()
            
            # Reshape pose back to original dimensions
            n_frames = pose_flat.shape[0]
            pose_dim = pose_flat.shape[1]
            n_joints = pose_dim // 3
            pose_recon = pose_flat.reshape(n_frames, n_joints, 3)
        
        # Apply inverse transformations using the dataloader functions
        trans_abs = inverse_transform_translation(trans_recon, init_trans)
        root_orient_abs = inverse_transform_global_orientation(orient_recon, init_root_orient)
        
        # Body pose is already absolute
        return {
            'trans': trans_abs,
            'root_orient': root_orient_abs,
            'pose_body': pose_recon
        }


    def test_reconstruction(self, data_path, person_idx=0, visualize=False):
        """
        Test the reconstruction quality of the decoupled VAE on a sample motion file.
        
        Args:
            data_path: Path to the motion file (.npz)
            person_idx: Index of the person to reconstruct (default: 0)
            visualize: Whether to visualize the reconstruction error (default: False)
            
        Returns:
            Dict containing error metrics for each motion component
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Load the sample motion
        data = np.load(data_path)
        
        # Extract motion components for the specified person
        trans_orig = data['trans'][person_idx]
        root_orient_orig = data['root_orient'][person_idx]
        pose_body_orig = data['pose_body'][person_idx]
        
        orig = {
            'trans': trans_orig,
            'root_orient': root_orient_orig,
            'pose_body': pose_body_orig
        }
        
        # Encode and decode
        with torch.no_grad():
            recon = self.reconstruct(
                orig['trans'],
                orig['root_orient'],
                orig['pose_body']
            )
        
        errors = self.compute_errors(orig, recon, visualize=True)
        # Print detailed error metrics
        print("\nReconstruction Error Metrics:")
        for component, metrics in errors.items():
            print(f"\n{component.upper()} Errors:")
            for metric_name, value in metrics.items():
                # Handle numpy arrays by converting to float if needed
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value)
                        print(f"  {metric_name}: {value:.6f}")
                    else:
                        print(f"  {metric_name}: {value}")
                else:
                    print(f"  {metric_name}: {value:.6f}")

def main():
    """
    Test the decoupled VAE on a sample motion file.
    """
    # Get absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Create a DecoupledVAE instance
    vae = DecoupledVAE(
        translation_checkpoint_path='checkpoints/translation_vae/translation_vae_best.pt',
        orientation_checkpoint_path='checkpoints/orientation_vae/orientation_vae_best.pt',
        pose_checkpoint_path='checkpoints/pose_vae/pose_vae_best.pt'
    )
    
    # Test reconstruction on sample motion
    sample_path = os.path.join(current_dir, 'sample_motion.npz')
    if not os.path.exists(sample_path):
        print(f"Error: Sample motion file not found at {sample_path}")
        return
    
    print(f"Testing reconstruction on {sample_path}")
    vae.test_reconstruction(sample_path, visualize=True)

if __name__ == "__main__":
    main()
