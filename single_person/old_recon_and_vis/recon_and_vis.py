#!/usr/bin/env python
"""
Motion Reconstruction and Visualization

This script:
1. Loads motion data and VAE models
2. Reconstructs motion components using specialized VAEs
3. Visualizes original and reconstructed motions
4. Evaluates reconstruction quality

This is a compatibility wrapper that uses the functionality from recon.py and vis.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from single_person.recon import (
    MotionData, SmplHelper, MotionProcessor, ModelLoader, MotionEvaluator, 
    MotionReconstructor, MotionReconApp
)
from single_person.vis import MotionVisualizer, MotionVisApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motion_recon_vis")

#---------------------------------------------------------------------------
# Main Application
#---------------------------------------------------------------------------

class MotionReconVisApp:
    """Main application for motion reconstruction and visualization."""
    
    def __init__(self):
        """Initialize the application."""
        self.args = self.parse_arguments()
        self.device = MotionReconApp().device  # Reuse device detection logic
        
        # Load SMPL helper
        self.smpl_helper = SmplHelper(Path(self.args.smpl_model))
        
        # Setup visualization
        self.visualizer = MotionVisualizer(self.smpl_helper)
        
        # Setup reconstruction
        translation_vae, orientation_vae, pose_vae = self._load_models()
        self.reconstructor = MotionReconstructor(
            translation_vae, orientation_vae, pose_vae, self.device
        )
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Motion Reconstruction and Visualization")
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
        parser.add_argument("--fps", type=int, default=30, 
                          help="Frames per second for visualization")
        parser.add_argument("--person_idx", type=int, default=0, 
                          help="Person index to reconstruct (default: 0)")
        return parser.parse_args()
    
    def _load_models(self):
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
    
    def _save_reconstruction(self, reconstructed):
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
        
        import numpy as np
        np.savez(recon_filename, 
                trans=trans,
                root_orient=root_orient,
                pose_body=pose_body)
                
        logger.info(f"Saved reconstructed motion to {recon_filename}")
    
    def setup_visualization_server(self):
        """Set up visualization server."""
        import viser
        server = viser.ViserServer()
        
        # Create GUI elements
        server.gui.add_text("Instructions", initial_value="Use the slider to control playback")
        server.gui.add_checkbox("Auto Play", initial_value=True)
        
        return server
    
    def run(self) -> None:
        """Run the application."""
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
            self._save_reconstruction(reconstructed)
        
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
        
        # Prepare visualization data
        print("Preparing visualization data...")
        orig_joint_data = self.visualizer.prepare_motion_data(
            person_motion_data.translation[person_idx], 
            person_motion_data.orientation[person_idx], 
            person_motion_data.pose_body[person_idx]
        )
        
        recon_joint_data = self.visualizer.prepare_motion_data(
            reconstructed["translation"], 
            reconstructed["orientation"], 
            reconstructed["pose_body"]
        )
        
        # Set up visualization server
        server = self.setup_visualization_server()
        
        # Run animation loop
        self.visualizer.run_animation_loop(
            server, orig_joint_data, recon_joint_data, fps=self.args.fps
        )

#---------------------------------------------------------------------------
# Entry Point
#---------------------------------------------------------------------------

def main():
    """Entry point for the application."""
    app = MotionReconVisApp()
    app.run()

if __name__ == "__main__":
    main() 