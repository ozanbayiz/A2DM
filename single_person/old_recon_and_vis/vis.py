#!/usr/bin/env python
"""
Motion Visualization

This script:
1. Takes reconstructed motion data and visualizes it
2. Provides an interactive 3D visualization in the browser
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import numpy as np
import torch
import viser
import viser.transforms as tf

from single_person.recon import MotionData, SmplHelper, MotionReconApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("motion_vis")

#---------------------------------------------------------------------------
# Visualization
#---------------------------------------------------------------------------

class MotionVisualizer:
    """
    Visualizes 3D human motion data using the Viser library.
    """
    
    def __init__(self, smpl_helper: SmplHelper):
        """
        Initialize the motion visualizer.
        
        Args:
            smpl_helper: SMPL model helper for computing mesh vertices
        """
        self.smpl_helper = smpl_helper
        
    def prepare_motion_data(self, trans: np.ndarray, 
                           root_orient: np.ndarray, 
                           pose_body: np.ndarray) -> List[np.ndarray]:
        """
        Prepare motion data for visualization.
        
        Args:
            trans: Translation data (T, 3)
            root_orient: Root orientation data (T, 3, 3)
            pose_body: Body pose data (T, 23, 3, 3)
            
        Returns:
            List containing [T_matrices, vertices, faces] for visualization
        """
        # Get number of frames
        num_frames = trans.shape[0]
        
        # Prepare output containers
        T_matrices_all = []
        vertices_all = []
        
        # Get faces from SMPL helper (only need to do this once)
        faces = self.smpl_helper.faces
        
        # Process each frame
        for i in range(num_frames):
            # Get joint transforms for current frame
            joint_rotmats = np.concatenate([root_orient[i], pose_body[i]], axis=0)
            T_matrices = self.smpl_helper.get_all_time_outputs(
                np.zeros(self.smpl_helper.num_betas), joint_rotmats[np.newaxis], trans[i:i+1])
            
            # Get vertices for the current frame
            verts, _ = self.smpl_helper.get_tpose(np.zeros(self.smpl_helper.num_betas))
            
            # Store results
            T_matrices_all.append(T_matrices[0])
            vertices_all.append(verts)
            
        # Ensure faces are valid indices (non-negative integers)
        if faces is not None:
            faces = np.maximum(0, faces)
            
        return [np.array(T_matrices_all), np.array(vertices_all), np.array([faces])]
    
    def setup_visualization(self, server: viser.ViserServer, 
                           offset: np.ndarray = np.array([2.0, 0.0, 0.0])) -> Tuple[List[Any], List[Any]]:
        """
        Set up the initial visualization scene.
        
        Args:
            server: Viser server instance
            offset: Offset between original and reconstructed visualization
            
        Returns:
            Tuple of (original mesh handles, reconstructed mesh handles)
        """
        # Add a ground plane for reference using add_grid
        server.add_grid(
            name="ground_plane",
            width=10.0,
            height=10.0,
            width_segments=10,
            height_segments=10,
            plane='xz',  # Ground plane should be on XZ plane
            cell_color=(200, 200, 200),
            section_color=(140, 140, 140),
            position=np.array([0, 0, 0]),
        )
        
        # Add a reference coordinate frame
        server.add_frame(
            name="world",
            wxyz=np.array([1, 0, 0, 0]),
            axes_length=0.5,
        )
        
        # Initialize empty mesh lists
        orig_meshes = []
        recon_meshes = []
        
        return orig_meshes, recon_meshes
    
    @staticmethod
    def update_meshes(meshes: List[Any], joint_data: List[np.ndarray], 
                      frame_idx: int, server: viser.ViserServer, 
                      offset: Optional[np.ndarray] = None) -> List[Any]:
        """
        Update mesh vertices for the current frame.
        
        Args:
            meshes: List of mesh handles
            joint_data: List of joint data
            frame_idx: Current frame index
            server: Viser server instance
            offset: Optional offset to apply to mesh positions
            
        Returns:
            Updated list of mesh handles
        """
        # Get vertices for current frame
        vertices = joint_data[1][frame_idx]
        
        # Apply offset if provided
        if offset is not None:
            vertices = vertices + offset
        
        # Update or create mesh
        if meshes:
            mesh = meshes[0]
            mesh.vertices = vertices
        else:
            faces = joint_data[2][0]
            # Ensure faces are proper integers and within valid range for uint32
            faces = np.clip(faces, 0, np.iinfo(np.uint32).max)
            faces = faces.astype(np.int32)  # Use int32 first to avoid overflow issues
            
            mesh = server.add_mesh(
                name="motion_mesh" if offset is None else "recon_mesh",
                vertices=vertices,
                faces=faces,
                color=(0.8, 0.8, 0.8, 1.0),
            )
            meshes = [mesh]
        
        return meshes
    
    def run_animation_loop(self, server: viser.ViserServer, 
                          orig_joint_data: List[np.ndarray], 
                          recon_joint_data: List[np.ndarray], 
                          fps: int = 30, 
                          offset: np.ndarray = np.array([2.0, 0.0, 0.0])) -> None:
        """
        Run animation loop to visualize original and reconstructed motion.
        
        Args:
            server: Viser server instance
            orig_joint_data: List of original joint data
            recon_joint_data: List of reconstructed joint data
            fps: Frames per second
            offset: Offset between original and reconstructed visualization
        """
        print("Starting animation loop...")
        orig_meshes, recon_meshes = self.setup_visualization(server, offset)
        num_frames = min(len(orig_joint_data[0]), len(recon_joint_data[0]))
        
        # Add UI controls for animation
        with server.add_gui_folder("Animation Controls"):
            fps_slider = server.add_gui_slider("FPS", min=1, max=120, step=1, initial_value=fps)
            time_slider = server.add_gui_slider("Frame", min=0, max=num_frames-1, step=1, initial_value=0)
            play_button = server.add_gui_button("Play/Pause")
            
        # Animation state
        state = {"playing": True, "frame": 0, "direction": 1}
        
        def toggle_playback(_):
            state["playing"] = not state["playing"]
            
        def update_frame(event):
            if not state["playing"]:
                state["frame"] = int(event.value)
        
        # Connect UI events
        play_button.on_click(toggle_playback)
        time_slider.on_update(update_frame)
        
        try:
            while True:
                start_time = time.time()
                
                current_fps = fps_slider.value
                
                if state["playing"]:
                    # Update frame index with ping-pong behavior
                    state["frame"] += state["direction"]
                    if state["frame"] >= num_frames - 1 or state["frame"] <= 0:
                        state["direction"] *= -1
                        state["frame"] += state["direction"]
                    
                    # Update slider to match frame
                    time_slider.value = state["frame"]
                
                # Update meshes
                orig_meshes = self.update_meshes(orig_meshes, orig_joint_data, state["frame"], server)
                recon_meshes = self.update_meshes(recon_meshes, recon_joint_data, state["frame"], server, offset)
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = max(0, 1.0/current_fps - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nAnimation stopped by user.")
        except Exception as e:
            print(f"Animation stopped due to error: {e}")

#---------------------------------------------------------------------------
# Main Application for Visualization
#---------------------------------------------------------------------------

class MotionVisApp:
    """
    Application for visualizing original and reconstructed motion.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.args = self.parse_arguments()
        self.smpl_helper = SmplHelper(self.args.smpl_model)
        self.visualizer = MotionVisualizer(self.smpl_helper)
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Visualize original and reconstructed motion")
        parser.add_argument("--input_file", type=Path, required=True, help="Path to input motion data file")
        parser.add_argument("--smpl_model", type=Path, default=Path("smpl/smpl_neutral.npz"), help="Path to SMPL model file")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second for visualization")
        parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to serve visualization")
        parser.add_argument("--port", type=int, default=8080, help="Port to serve visualization")
        return parser.parse_args()
    
    def setup_visualization_server(self) -> viser.ViserServer:
        """Set up the visualization server."""
        return viser.ViserServer(host=self.args.host, port=self.args.port)
        
    def run(self) -> None:
        """Run the motion visualization application."""
        # Load motion data
        data = np.load(self.args.input_file)
        
        # Extract data for the first person
        if 'trans' in data and 'root_orient' in data and 'pose_body' in data:
            # Direct format
            trans = data['trans']
            root_orient = data['root_orient']
            pose_body = data['pose_body']
        elif 'motion' in data:
            # Motion array format
            motion = data['motion']
            if len(motion.shape) > 2:  # Multi-person data
                print(f"Found {motion.shape[0]} people in the data. Using person 0.")
                motion = motion[0]
            trans = motion[:, :3]
            root_orient = motion[:, 3:12].reshape(-1, 3, 3)
            pose_body = motion[:, 12:].reshape(-1, 23, 3, 3)
        else:
            raise ValueError("Unsupported data format. Expected 'trans', 'root_orient', and 'pose_body' or 'motion'.")
        
        # Prepare visualization data
        print("Preparing visualization data...")
        orig_joint_data = self.visualizer.prepare_motion_data(trans, root_orient, pose_body)
        
        # Load reconstructed data if available
        recon_file = self.args.input_file.with_name(f"{self.args.input_file.stem}_recon.npz")
        if recon_file.exists():
            print(f"Loading reconstruction from {recon_file}")
            recon_data = np.load(recon_file)
            if 'trans' in recon_data and 'root_orient' in recon_data and 'pose_body' in recon_data:
                recon_trans = recon_data['trans']
                recon_root_orient = recon_data['root_orient']
                recon_pose_body = recon_data['pose_body']
            else:
                raise ValueError("Reconstructed data format not supported.")
            
            recon_joint_data = self.visualizer.prepare_motion_data(
                recon_trans, recon_root_orient, recon_pose_body)
        else:
            print("No reconstruction file found. Visualizing original motion only.")
            recon_joint_data = orig_joint_data
        
        # Start visualization server
        server = self.setup_visualization_server()
        
        # Run animation loop
        self.visualizer.run_animation_loop(
            server,
            orig_joint_data,
            recon_joint_data,
            fps=self.args.fps
        )

#---------------------------------------------------------------------------
# Entry Point
#---------------------------------------------------------------------------

def main():
    """Main entry point."""
    app = MotionVisApp()
    app.run()

if __name__ == "__main__":
    main() 