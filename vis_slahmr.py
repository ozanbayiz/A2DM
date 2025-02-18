from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tyro

import viser
import viser.transforms as tf


@dataclass(frozen=True)
class SmplFkOutputs:
    T_world_joint: np.ndarray  # (num_joints, 4, 4)
    T_parent_joint: np.ndarray  # (num_joints, 4, 4)


class SmplHelper:
    """Helper for models in the SMPL family, implemented in numpy. Does not include blend skinning."""

    def __init__(self, model_path: Path) -> None:
        assert model_path.suffix.lower() == ".npz", "Model should be an .npz file!"
        body_dict = dict(**np.load(model_path, allow_pickle=True))

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
        # Compute the shaped vertices and joint positions (t-pose) once.
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)
        return v_tpose, j_tpose

    def get_outputs(
        self, 
        betas: np.ndarray, 
        joint_rotmats: np.ndarray,  # shape (num_joints, 3, 3)
        translation: np.ndarray     # shape (1, 3)
    ) -> SmplFkOutputs:
        v_tpose = self.v_template + np.einsum("vxb,b->vx", self.shapedirs, betas)
        j_tpose = np.einsum("jv,vx->jx", self.J_regressor, v_tpose)

        # Initialize local SE(3) transforms.
        T_parent_joint = np.tile(np.eye(4)[None, ...], (self.num_joints, 1, 1))
        T_parent_joint[:, :3, :3] = joint_rotmats
        T_parent_joint[0, :3, 3] = j_tpose[0]
        T_parent_joint[1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]

        # Forward kinematics.
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[i] = T_world_joint[self.parent_idx[i]] @ T_parent_joint[i]
        
        # Apply global translation.
        T_translation = np.eye(4)
        T_translation[:3, 3] = translation
        T_world_joint = np.einsum('ij,kjl->kil', T_translation, T_world_joint)

        return SmplFkOutputs(T_world_joint, T_parent_joint)

    def get_all_time_outputs(
        self,
        betas: np.ndarray,
        joint_rotmats: np.ndarray,  # shape (T, num_joints, 3, 3)
        translation: np.ndarray     # shape (T, 3)
    ) -> np.ndarray:
        v_tpose, j_tpose = self.get_tpose(betas)
        T = joint_rotmats.shape[0]  # Number of time steps

        # Initialize local joint transforms for each time step.
        T_parent_joint = np.tile(np.eye(4)[None, None, ...], (T, self.num_joints, 1, 1))
        T_parent_joint[:, :, :3, :3] = joint_rotmats

        # Set translations (constant across time).
        T_parent_joint[:, 0, :3, 3] = j_tpose[0]
        T_parent_joint[:, 1:, :3, 3] = j_tpose[1:] - j_tpose[self.parent_idx[1:]]

        # Forward kinematics.
        T_world_joint = T_parent_joint.copy()
        for i in range(1, self.num_joints):
            T_world_joint[:, i] = T_world_joint[:, self.parent_idx[i]] @ T_parent_joint[:, i]

        # Create a translation transform for each time step.
        T_translation = np.tile(np.eye(4)[None, ...], (T, 1, 1))
        T_translation[:, :3, 3] = translation

        # Apply global translation.
        T_world_joint = T_translation[:, None, :, :] @ T_world_joint

        return T_world_joint


# --- GUI elements for animation and file selection ---
@dataclass
class AnimationGuiElements:
    time_slider: viser.GuiInputHandle[int]
    start_button: viser.GuiButton
    stop_button: viser.GuiButton
    file_selector: viser.GuiInputHandle[str]
    animation_active: bool
    time_changed: bool
    file_changed: bool


def make_gui_elements(server: viser.ViserServer, file_options: List[str]) -> AnimationGuiElements:
    tab_group = server.gui.add_tab_group()

    def set_animation_active(_)->None:
        out.animation_active = True

    def set_animation_inactive(_)->None:
        out.animation_active = False

    def set_time_changed(_):
        out.time_changed = True

    def set_file_changed(_):
        out.file_changed = True

    with tab_group.add_tab("Animation"):
        # Time slider from 0 to 99.
        time_slider = server.gui.add_slider("Time", min=0, max=99, step=1, initial_value=0)
        start_button = server.gui.add_button("Start")
        stop_button = server.gui.add_button("Stop")
        # Create a dropdown for file selection; file_options now contain only base names.
        file_selector = server.gui.add_dropdown("Motion File", options=file_options, initial_value=file_options[0])
        
        time_slider.on_update(set_time_changed)
        start_button.on_click(set_animation_active)
        stop_button.on_click(set_animation_inactive)
        file_selector.on_update(set_file_changed)

    out = AnimationGuiElements(
        time_slider, start_button, stop_button, file_selector,
        animation_active=False, time_changed=False, file_changed=False
    )
    return out


def load_motion_data(motion_path: Path, model: SmplHelper):
    """
    Loads the motion data from the given .npz file and precomputes the joint outputs.
    Returns a tuple (alice_all_time_joints, bob_all_time_joints).
    """
    motion_data = np.load(motion_path)

    alice_joint_rots = np.concatenate([
        motion_data["root_orient"][0, ...],  # shape: (100, 3)
        motion_data["pose_body"][0, ...],      # shape: (100, 63)
        np.zeros((100, 6))                     # shape: (100, 6)
    ], axis=1).reshape(100, 24, 3)
    alice_joint_rotmats = (
        tf.SO3.exp(alice_joint_rots.reshape(-1, 3)).as_matrix().reshape(100, 24, 3, 3)
    )

    bob_joint_rots = np.concatenate([
        motion_data["root_orient"][1, ...],
        motion_data["pose_body"][1, ...],
        np.zeros((100, 6))
    ], axis=1).reshape(100, 24, 3)
    bob_joint_rotmats = (
        tf.SO3.exp(bob_joint_rots.reshape(-1, 3)).as_matrix().reshape(100, 24, 3, 3)
    )

    translation = motion_data["trans"]
    translation -= np.mean(translation, axis=(0, 1))

    # Prepare shape coefficients.
    alice_betas = np.zeros((model.num_betas,))
    bob_betas = np.zeros((model.num_betas,))
    alice_betas[: len(motion_data["betas"][0])] = motion_data["betas"][0]
    bob_betas[: len(motion_data["betas"][1])] = motion_data["betas"][1]

    alice_all_time_joints = model.get_all_time_outputs(
        betas=alice_betas,
        joint_rotmats=alice_joint_rotmats,
        translation=translation[0, ...]
    )
    bob_all_time_joints = model.get_all_time_outputs(
        betas=bob_betas,
        joint_rotmats=bob_joint_rotmats,
        translation=translation[1, ...]
    )
    return alice_all_time_joints, bob_all_time_joints


def main(
    model_path: Path = Path("smpl/smpl_neutral.npz"),
    data_dir: Path = Path("sns_slahmr")
) -> None:
    # data_dir is the directory containing the .npz motion files.
    # Build a list of file base names (without directory paths).
    file_list = sorted([f.name for f in data_dir.glob("*.npz")])
    if not file_list:
        print(f"No .npz files found in {data_dir}.")
        return

    server = viser.ViserServer()
    server.scene.set_up_direction("-y")
    server.gui.configure_theme(control_layout="collapsible")

    model = SmplHelper(model_path)
    gui_elements = make_gui_elements(server, file_options=file_list)

    # Create the initial t-pose mesh for both characters.
    v_tpose, j_tpose = model.get_tpose(np.zeros((model.num_betas,)))
    alice_mesh_handle = server.scene.add_mesh_skinned(
        "/alice",
        v_tpose,
        model.faces,
        bone_wxyzs=tf.SO3.identity(batch_axes=(model.num_joints,)).wxyz,
        bone_positions=j_tpose,
        skin_weights=model.weights,
        wireframe=False,
        color=(153, 255, 204),
    )
    bob_mesh_handle = server.scene.add_mesh_skinned(
        "/bob",
        v_tpose,
        model.faces,
        bone_wxyzs=tf.SO3.identity(batch_axes=(model.num_joints,)).wxyz,
        bone_positions=j_tpose,
        skin_weights=model.weights,
        wireframe=False,
        color=(255, 104, 255),
    )

    # Initially load the motion from the default selected file.
    # current_motion_file is computed by joining data_dir with the selected file name.
    current_motion_file = data_dir / Path(gui_elements.file_selector.value)
    alice_all_time_joints, bob_all_time_joints = load_motion_data(current_motion_file, model)

    t = 0
    # Main update loop.
    while True:
        time.sleep(0.02)

        if gui_elements.file_changed:
            # Reload motion data if the file selection changed.
            current_motion_file = data_dir / Path(gui_elements.file_selector.value)
            print(f"Loading motion file: {current_motion_file.name}")
            alice_all_time_joints, bob_all_time_joints = load_motion_data(current_motion_file, model)
            gui_elements.file_changed = False
            # Optionally reset time to zero:
            t = 0

        if gui_elements.time_changed:
            t = gui_elements.time_slider.value
            gui_elements.time_changed = False
        elif gui_elements.animation_active:
            t = (t + 1) % 100
            gui_elements.time_slider.value = t

        # Instead of converting per bone in a loop, convert all at once.
        alice_wxyz = tf.SO3.from_matrix(alice_all_time_joints[t, :, :3, :3]).wxyz
        bob_wxyz = tf.SO3.from_matrix(bob_all_time_joints[t, :, :3, :3]).wxyz

        # Update each bone for both characters.
        for i, bone in enumerate(alice_mesh_handle.bones):
            bone.wxyz = alice_wxyz[i]
            bone.position = alice_all_time_joints[t, i, :3, 3]

        for i, bone in enumerate(bob_mesh_handle.bones):
            bone.wxyz = bob_wxyz[i]
            bone.position = bob_all_time_joints[t, i, :3, 3]


if __name__ == "__main__":
    tyro.cli(main, description=__doc__)
