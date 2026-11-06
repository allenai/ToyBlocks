"""
Code for setting up the viser based viewer for light-weight visualization of Isaac based environments and collected trajectories

Parts adapted from https://github.com/mujocolab/mjlab/blob/main/src/mjlab/viewer/ 
"""
import asyncio
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import torch
import trimesh
from PIL import Image
import viser
import viser.extras
import numpy as np

from toyblocks.tasks.base_env import BaseEnv
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from toyblocks.utils import common_utils, tree_utils, usd_utils
from isaaclab.utils import math
@dataclass
class Frame:
    articulations: dict[str, dict[str, np.ndarray]]
    rigid_objects: dict[str, np.ndarray]
    markers: dict[str, np.ndarray] = field(default_factory=dict)
    """visualization marker poses per tracked env: name -> (n_tracked_envs, 7) pos+wxyz in env-local coords"""
    episode_reset: bool = False
    """whether this is a frame generated from a env reset"""
    timestep: int = 0
    """the timestep of the frame (control step)"""
    substep: int | None = None
    """the sub step of the frame. Usually this is 0, 1, ..., decimation - 1. If it's None, then usually
    this frame is a reset frame"""
    final_episode_step: bool = False
    """whether this is the final step of the episode"""
    info: dict = field(default_factory=dict)
    reward: float | None = None
    """reward in this frame"""
    ret: float | None = None
    """return so far """
    episode_id: int = 0
    """the episode ID of the frame"""

@dataclass
class Entity:
    name: str
    prim_path: str
@dataclass
class RigidObject(Entity):
    handle: viser.GlbHandle
    mesh: trimesh.Trimesh

@dataclass
class Link(Entity):
    handle: viser.GlbHandle
    mesh: trimesh.Trimesh

@dataclass
class Articulation(Entity):
    links: dict[str, Link]


@dataclass
class ViewerCfg:
    include_cameras: bool = False
    include_substeps: bool = False
    env_ids: Sequence[int] = field(default_factory=lambda: [0])
    """indexes of the parallel environments to track with the viewer. By default only the first parallel environment is tracked"""
    max_frames_per_trajectory: int = 5000
    """maximum number of frames to store for each trajectory for viewing/replay before throwing out the oldest frames"""
    max_trajectories_per_env: int = 100
    """maximum number of trajectories per parallel environment to store for viewing/replay before throwing out the oldest trajectories"""
    ignore_empty_trajectories: bool = True
    """Whether to ignore trajectories that have no frames saved for viewing other than the reset frame."""
    share_url: bool = False
    """Use this setting to allow viser to share a public url (handy on a remote machine)."""
    minimal_gui: bool = False
    """Strip the viser front-end down toward the scene itself: dark mode, no logo or share button,
    the control panel docked to the right edge and minimized to a rail strip, so the browser opens
    on the scene alone (viser >= 1.1; older viser docks it but starts it expanded), and no debug
    markers (goal and block frame arrows, cube-corner spheres). For a screen-facing view where the
    panel is not the subject and nothing in the viewer can be interacted with."""
    texture_max_px: int = 2048
    """Longest side a mesh texture is downscaled to before it is uploaded to the browser. The letter
    blocks ship a 4096px decal sheet and a 2048px base texture, and every block's glTF embeds its own
    copy: 14 MB per block as shipped, 8.4 MB at 2048, 2.6 MB at 1024, 0.8 MB at 512. The scene is
    re-sent to every browser that connects, so this is the cost of a reconnect. 2048 halves the decal
    sheet and leaves the glyphs legible; 1024 made them visibly soft. 0 disables the cap."""

class Viewer:
    def __init__(self, cfg: ViewerCfg = ViewerCfg()):
        self.cfg = cfg

        # initialize server
        self._server = viser.ViserServer()
        self._server.gui.set_panel_label("ToyBlocks Viewer")
        if self.cfg.minimal_gui:
            main_panel = getattr(self._server.gui, "main_panel", None)
            if main_panel is None:
                # viser < 1.1: the layout flag is the only way to dock the panel, and it starts expanded.
                self._server.gui.configure_theme(
                    dark_mode=True,
                    show_logo=False,
                    show_share_button=False,
                    control_layout="collapsible",
                    control_width="small",
                )
            else:
                # viser 1.1 deprecated control_layout in favour of main_panel placement. Panel
                # commands are replayed to clients that connect later, so every browser opens with
                # the panel minimized; the widgets stay one click away on the rail.
                self._server.gui.configure_theme(
                    dark_mode=True,
                    show_logo=False,
                    show_share_button=False,
                    control_width="small",
                )
                main_panel.dock_right()
                main_panel.minimize()
        self._server.scene.set_up_direction("+z")
        self._server.scene.add_light_ambient(name="/World/ambient_light", color=(255, 255, 255), intensity=1.0)

        if self.cfg.share_url:
            self._server.request_share_url()

        ### Setup GUI ###
        tabs = self._server.gui.add_tab_group()
        with tabs.add_tab("Main"):
            with self._server.gui.add_folder("Entity"):
                self._entity_selection_info_html = self._server.gui.add_html("")
                self._entity_selection_info_html.content = """
                <div style="font-size: 0.875em; margin-left: .75em">
                <div>No entity selected</div>
                </div>
                """
                self._selected_entity: Entity | None = None
                self._entity_info_folder = self._server.gui.add_folder("Info")
                with self._entity_info_folder:
                    self._entity_metadata_info_html = self._server.gui.add_html("")
                self._entity_joints_folder = self._server.gui.add_folder("Joints")
                with self._entity_joints_folder:
                    self._entity_joint_info_html = self._server.gui.add_html("")
                self._entity_pose_folder = self._server.gui.add_folder("Pose")
                with self._entity_pose_folder:
                    self._entity_pose_info_html = self._server.gui.add_html("")
                self._entity_info_folder.visible = False
                self._entity_joints_folder.visible = False
                self._entity_pose_folder.visible = False
            with self._server.gui.add_folder("Environment"):
                with self._server.gui.add_folder("Info"):
                    self._env_info_html = self._server.gui.add_html("")
                with self._server.gui.add_folder("Physics"):
                    self._physics_info_html = self._server.gui.add_html("")
                with self._server.gui.add_folder("Current Frame"):
                    self._current_frame_step_info_html = self._server.gui.add_html("")
                
                with self._server.gui.add_folder("Frame Info"):
                    self._current_frame_info_html = self._server.gui.add_html("")
                
            with self._server.gui.add_folder("Controls"):
                # dropdown to select which parallel environment to view
                self._env_id_dropdown = self._server.gui.add_dropdown(label="Environment ID", options=[str(i) for i in self.cfg.env_ids], initial_value=str(self.cfg.env_ids[0]))
                @self._env_id_dropdown.on_update
                def _(_):
                    self.view_frame(self._slider.value, self.viewed_env_id, self.viewed_trajectory_id)

                self._episode_id_counter = 0
                self._trajectory_id_slider = self._server.gui.add_slider(label="Trajectory ID", min=0, max=1, step=1, initial_value=0)
                @self._trajectory_id_slider.on_update
                def _(_):
                    self._slider.value = min(self._slider.value, len(self._frames[self.viewed_trajectory_id]) - 1)
                    self.view_frame(self._slider.value, self.viewed_env_id, self.viewed_trajectory_id)
                # trajectory timestep slider
                self._slider = self._server.gui.add_slider(label="Replay Step", min=0, max=1, step=1, initial_value=0)
                @self._slider.on_update
                def _(_):
                    self.view_frame(self._slider.value, self.viewed_env_id, self.viewed_trajectory_id)

                self._play_repeat_control_dropdown = self._server.gui.add_dropdown(label="Play Control", options=["Once", "Loop", "Next Trajectory", "Next Environment"], initial_value="Next Trajectory")
                
                self._display_granularity_dropdown = self._server.gui.add_dropdown(label="Display Granularity", options=["Sub step", "Control step"], initial_value="Control step")
                @self._display_granularity_dropdown.on_update
                def _(_):
                    self._update_play_dt()
                    self._slider.value = self._slider.value * self._env_cfg.decimation if self._display_granularity_is_substep else self._slider.value // self._env_cfg.decimation
                    self._update_slider_max(self.viewed_trajectory_id)
                    self._update_info_htmls(self._env_cfg, task_id=self._task_id)

                if not self.cfg.include_substeps:
                    self._display_granularity_dropdown.visible = False
                # play/pause button
                self._control_button = self._server.gui.add_button(label="Play", icon=viser.Icon.PLAYER_PLAY)
                self._play_dt = 0.04
                async def play_trajectory_loop():
                    while True:
                        self.view_frame(self._slider.value, self.viewed_env_id)
                        if self._play_repeat_control_dropdown.value == "Once":
                            self._slider.value = self._slider.value + 1
                            if self._slider.value >= self._slider.max:
                                self._stop_player()
                                self._stop_player = None
                                break
                        elif self._play_repeat_control_dropdown.value == "Loop":
                            self._slider.value = (self._slider.value + 1) % (self._slider.max + 1)
                        elif self._play_repeat_control_dropdown.value == "Next Trajectory":
                            value = self._slider.value + 1
                            if value >= self._slider.max:
                                self._slider.value = 0
                                self._trajectory_id_slider.value = (self._trajectory_id_slider.value + 1) % (self._trajectory_id_slider.max + 1)
                            else:
                                self._slider.value = value
                        elif self._play_repeat_control_dropdown.value == "Next Environment":
                            value = self._slider.value + 1
                            if value >= self._slider.max:
                                self._env_id_dropdown.value = self._env_id_dropdown.options[(self._env_id_dropdown.options.index(self._env_id_dropdown.value) + 1) % len(self._env_id_dropdown.options)]
                                self._slider.value = 0
                            else:
                                self._slider.value = value

                        await asyncio.sleep(self._play_dt)
                self._stop_player = None
                @self._control_button.on_click
                async def _(_):
                    if self._stop_player is not None:
                        self._stop_player()
                        self._stop_player = None
                    else:
                        loop = asyncio.get_event_loop()
                        task = loop.create_task(play_trajectory_loop())
                        def stop_player():
                            task.cancel()
                            self._control_button.label = "Play"
                            self._control_button.icon = viser.Icon.PLAYER_PLAY
                        self._stop_player = stop_player
                        self._control_button.label = "Pause"
                        self._control_button.icon = viser.Icon.PLAYER_PAUSE

                self._step_btn = self._server.gui.add_button("Step", color="blue", icon="player-skip-forward")
                @self._step_btn.on_click
                async def _(_):
                    self._slider.value = min(self._slider.value + 1, self._slider.max)
                    self.view_frame(self._slider.value, self.viewed_env_id, self.viewed_trajectory_id)           

        # Add ground plane
        self._server.scene.add_grid("/ground", width=10.0, height=10.0, cell_size=0.1)
        
        self._robot_urdf_handles: dict[str, viser.extras.ViserUrdf] = dict()
        # TODO: copy isaac gui viewer pose
        self._server.initial_camera.position = (1.0, 1.0, 2.0)
        self._server.initial_camera.look_at = (0.0, 0.0, 0.0)

        # initialize handles for objects
        self._articulations: dict[str, Articulation] = dict()
        self._rigid_objects: dict[str, RigidObject] = dict()
        self._marker_handles: dict[str, list] = dict()  # name -> list of handles (1 for frame, K for icospheres)
        self._marker_objects: dict = dict()
        # initialize buffers
        self._frames: deque[deque[Frame]] = deque(maxlen=self.cfg.max_trajectories_per_env)
        self._current_frame: Frame | None = None

        self._last_viewed_set = None
        self.track_env = False
        """if viewer.hook_to_env is called and this is set to True, the viewer will track the environment and save the current sim state for viewing.
        
        set this to False to stop tracking the environment to let the environment run efficiently
        """
    def get_server_url(self) -> str:
        return f"http://{self._server.get_host()}:{self._server.get_port()}"

    @property
    def _display_granularity_is_substep(self) -> bool:
        return self._display_granularity_dropdown.value == "Sub step"

    @property
    def viewed_env_id(self) -> int:
        return int(self._env_id_dropdown.value)
    @property
    def viewed_trajectory_id(self) -> int:
        return int(self._trajectory_id_slider.value)

    def select_entity(self, entity: Entity | str | None):
        if isinstance(entity, str):
            if entity in self._rigid_objects:
                entity = self._rigid_objects[entity]
        if entity is None:
            self._entity_info_folder.visible = False
            self._entity_joints_folder.visible = False
            self._entity_pose_folder.visible = False
            self._entity_selection_info_html.visible = True
            return
        self._entity_selection_info_html.visible = False
        self._selected_entity = entity
        self._entity_metadata_info_html.content = f"""
        <div style="font-size: 0.875em; margin-left: .75em">
        <div>Name: {self._selected_entity.name}</div>
        <div>Prim Path: {self._selected_entity.prim_path}</div>
        </div>
        """
        self._entity_metadata_info_html.visible = True
        self._entity_info_folder.visible = True
        
        if isinstance(self._selected_entity, RigidObject) or isinstance(self._selected_entity, Link):
            euler_xyz = math.euler_xyz_from_quat(torch.from_numpy(self._selected_entity.handle.wxyz).unsqueeze(0))
            euler_xyz = [euler_xyz[0][0].numpy().round(4).tolist(), euler_xyz[1][0].numpy().round(4).tolist(), euler_xyz[2][0].numpy().round(4).tolist()]
            self._entity_pose_info_html.content = f"""
            <div style="font-size: 0.875em; margin-left: .75em">
            <div>Position: {f"{self._selected_entity.handle.position.round(4).tolist()}" if self._selected_entity is not None else "N/A"}</div>
            <div>Quaternion: {f"{self._selected_entity.handle.wxyz.round(4).tolist()}" if self._selected_entity is not None else "N/A"}</div>
            <div>Euler Angles: X={"+" if euler_xyz[0] >= 0 else ""}{euler_xyz[0]}, Y={"+" if euler_xyz[1] >= 0 else ""}{euler_xyz[1]}, Z={"+" if euler_xyz[2] >= 0 else ""}{euler_xyz[2]}</div>
            </div>
            """
            self._entity_pose_folder.visible = True
        elif isinstance(self._selected_entity, Articulation):
            # TODO
            pass

    def _add_prim(self, prim_path: str, stage):
        """Add one prim's visual meshes to the scene, keeping each sub-mesh's own material.

        A prim with several textured sub-meshes cannot become one trimesh without an atlas
        repack, and that repack rewrites UVs: the letter blocks' glyph is a UV window into a
        shared decal sheet, so the decal faces land on the base texture and every block renders
        as bare wood. A glTF scene carries one material per sub-mesh, so nothing is repacked.

        Textures are downscaled to `cfg.texture_max_px` first, so the upload -- and the concatenated
        mesh `export_assets` pickles -- carry the smaller copies.

        :return: The viser handle, and the concatenated mesh the caller records for `export_assets`
        """
        parts = usd_utils.prim_to_trimesh_parts(prim_path, stage, visual_only=True)
        self._shrink_textures(parts)
        mesh = trimesh.util.concatenate(parts)

        if len(parts) > 1:
            scene = trimesh.Scene()
            for i, part in enumerate(parts):
                scene.add_geometry(part, node_name=f"part_{i}")
            handle = self._server.scene.add_glb(name=prim_path, glb_data=scene.export(file_type="glb"))
        else:
            handle = self._server.scene.add_mesh_trimesh(name=prim_path, mesh=mesh)

        return handle, mesh

    def _shrink_textures(self, parts: list[trimesh.Trimesh]) -> None:
        """Downscale every texture image on `parts` to at most `cfg.texture_max_px` a side, in place.

        Covers both material classes trimesh reads from USD: `PBRMaterial` (baseColor, normal,
        emissive, metallicRoughness textures) and `SimpleMaterial` (a single `image`).
        """
        max_px = self.cfg.texture_max_px
        if max_px <= 0:
            return
        attrs = ("baseColorTexture", "normalTexture", "emissiveTexture", "metallicRoughnessTexture", "image")
        for part in parts:
            material = getattr(part.visual, "material", None)
            for attr in attrs:
                image = getattr(material, attr, None)
                if not isinstance(image, Image.Image) or max(image.size) <= max_px:
                    continue
                scale = max_px / max(image.size)
                size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
                setattr(material, attr, image.resize(size, Image.LANCZOS))

    def load_assets(self, base_env: BaseEnv):
        # clear out all the old viser handles and old dicts
        for k in self._articulations.keys():
            for link in self._articulations[k].links.values():
                link.handle.remove()
        for k in self._rigid_objects.keys():
            self._rigid_objects[k].handle.remove()
        for handles in self._marker_handles.values():
            for handle in handles:
                handle.remove()
        self._articulations = dict()
        self._rigid_objects = dict()
        self._marker_handles = dict()
        self._marker_objects = dict()

        # load current assets in stage
        articulations = base_env.scene.articulations
        rigid_objects = base_env.scene.rigid_objects
        stage = base_env.scene.stage
        for k in articulations.keys():
            if k == "robot" and base_env.robot.urdf_path is not None:
                # check if robot has a corresponding URDF and use that instead
                self._robot_urdf_handles[k] = viser.extras.ViserUrdf(self._server, Path(base_env.robot.urdf_path).absolute())
            else:
                self._articulations[k] = Articulation(name=k, prim_path=k, links=dict())
                for i, link_prim_path in enumerate(base_env.robot.articulation.root_physx_view.link_paths[0]):
                    handle, mesh = self._add_prim(link_prim_path, stage)
                    link = Link(name=link_prim_path, prim_path=link_prim_path, handle=handle, mesh=mesh)
                    self._articulations[k].links[link_prim_path] = link
                    @handle.on_click
                    def _(_):
                        self.select_entity(link)

        for k, rigid_object in rigid_objects.items():
            def create_rigid(k, rigid_object):
                prim_path = rigid_object.root_physx_view.prim_paths[0]
                handle, mesh = self._add_prim(prim_path, stage)
                self._rigid_objects[k] = RigidObject(name=k, prim_path=prim_path, handle=handle, mesh=mesh)
                @handle.on_click
                def _(_):
                    self.select_entity(self._rigid_objects[k])
            create_rigid(k, rigid_object)

        # The markers are Isaac's debug visuals (goal and block frame arrows, cube-corner spheres).
        # They are meaningless where nothing in the viewer can be acted on, and they are most of the
        # handles updated per frame -- eight spheres per marked block -- so minimal_gui drops them:
        # with no marker objects registered, frames carry no marker poses and no handles are made.
        markers = dict() if self.cfg.minimal_gui else base_env.visualization_markers
        for name, marker in markers.items():
            # wrap visualize() to cache the last world-space translations and orientations
            marker._viewer_last_translations = None
            marker._viewer_last_orientations = None
            original_visualize = marker.visualize
            def _make_cached_visualize(m, orig):
                def cached_visualize(translations=None, orientations=None, **kwargs):
                    if translations is not None:
                        m._viewer_last_translations = translations.cpu().numpy() if hasattr(translations, 'cpu') else np.asarray(translations)
                    if orientations is not None:
                        m._viewer_last_orientations = orientations.cpu().numpy() if hasattr(orientations, 'cpu') else np.asarray(orientations)
                    return orig(translations=translations, orientations=orientations, **kwargs)
                return cached_visualize
            marker.visualize = _make_cached_visualize(marker, original_visualize)
        self._marker_objects = markers
        self._cube_corner_colors = [(255,0,0),(255,128,0),(255,255,0),(0,255,255),(0,255,0),(0,0,255),(255,0,255),(0,255,128)]

        self._env_id_map = {id: i for i, id in enumerate(self.cfg.env_ids)}

    def export_assets(self):
        import pickle
        with open("assets.pkl", "wb") as f:
            pickle.dump(dict(articulation_meshes={k: v.mesh for k, v in self._articulations.items()}, rigid_object_meshes={k: v.mesh for k, v in self._rigid_objects.items()}), f)

    @property
    def current_frame(self) -> Frame:
        return self._current_frame

    def view_frame(self, frame: Frame | int = 0, env_id: int = 0, trajectory_id: int = 0):
        """view a given frame in viser and updates all handles and objects (RigidObject, Articulation, Entity)"""
        view_set = (frame, trajectory_id, env_id, self._display_granularity_is_substep)
        if self._last_viewed_set == view_set:
            return
        self._last_viewed_set = view_set
        if isinstance(frame, int):
            if self.cfg.include_substeps:
                if self._display_granularity_is_substep or frame == 0:
                    frame = self._frames[trajectory_id][frame]
                else:
                    frame = self._frames[trajectory_id][frame * self._env_cfg.decimation]
            else:
                frame = self._frames[trajectory_id][frame]
        self._current_frame = frame
        if self._display_granularity_is_substep:
            self._update_slider_max(trajectory_id)
        for k, articulation in frame.articulations.items():
            if k in self._robot_urdf_handles:
                self._robot_urdf_handles[k].update_cfg(articulation["actuated_joint_pos"][self._env_id_map[env_id]])
            else:
                for link_prim_path, pose in articulation["link_poses"].items():
                    # isaac specific code to map other parallel environments to the first one
                    link_prim_path = link_prim_path.replace(f"/World/envs/env_{env_id}", "/World/envs/env_0")
                    self._articulations[k].links[link_prim_path].handle.position = pose[self._env_id_map[env_id], :3]
                    self._articulations[k].links[link_prim_path].handle.wxyz = pose[self._env_id_map[env_id], 3:]
        for k, pose in frame.rigid_objects.items():
            k = k.replace(f"/World/envs/env_{env_id}", "/World/envs/env_0")
            self._rigid_objects[k].handle.position = pose[self._env_id_map[env_id], :3]
            self._rigid_objects[k].handle.wxyz = pose[self._env_id_map[env_id], 3:]
        for name, pose in frame.markers.items():
            # pose: (n_tracked_envs, K, 7)
            env_data = pose[self._env_id_map[env_id]]  # (K, 7)
            K = len(env_data)
            # if K==1, it's a coordinate frame marker
            # otherwise, it's cube corner markers (K=8)
            if K == 1:
                if name not in self._marker_handles:
                    self._marker_handles[name] = [self._server.scene.add_frame(
                        name=f"markers/{name}", axes_length=0.1, axes_radius=0.005
                    )]
                self._marker_handles[name][0].position = env_data[0, :3]
                self._marker_handles[name][0].wxyz = env_data[0, 3:]
            else:
                if name not in self._marker_handles:
                    self._marker_handles[name] = [
                        self._server.scene.add_icosphere(
                            name=f"markers/{name}/{i}", radius=0.005, color=self._cube_corner_colors[i % 8],
                            position=env_data[i, :3],
                        )
                        for i in range(K)
                    ]
                else:
                    for i, handle in enumerate(self._marker_handles[name]):
                        handle.position = env_data[i, :3]
        
        self.select_entity(self._selected_entity)

        self._current_frame_step_info_html.content = f"""
        <div style="font-size: 0.875em; margin-left: .75em">
        <div>Episode ID: {frame.episode_id}</div>
        <div>Timestep: {frame.timestep}</div>
        <div>Substep: {frame.substep}</div>
        <div>Reset Frame: {frame.episode_reset}</div>
        <div>Final Episode Step: {frame.final_episode_step}</div>
        </div>
        """
        flattened_info = dict()
        def flatten_dict(d, path=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    flatten_dict(v, path=f"{path}{k}/")
            else:
                flattened_info[path] = d[self._env_id_map[env_id]]
        flatten_dict(frame.info)
        self._current_frame_info_html.content = f"""
        <div style="font-size: 0.875em; margin-left: .75em">
        <div>Reward: {f"{frame.reward[self._env_id_map[env_id]]:.5f}" if frame.reward is not None else "N/A"}</div>
        <div>Return: {f"{frame.ret[self._env_id_map[env_id]]:.5f}" if frame.ret is not None else "N/A"}</div>
        {"".join([f"<div>{k}: {f'{v:.5f}' if isinstance(v, float) or isinstance(v, np.float32) else v}" for k, v in flattened_info.items()])}
        </div>
        """

    def hook_to_env(self, env):
        """hook the viewer to a given base environment. Whenever the environment takes a step or resets and self.track_env is True, the viewer will save the current sim state for viewing."""
        base_env: BaseEnv = env.unwrapped
        self._env_cfg = base_env.cfg
        self._task_id = base_env.spec.id
        self.load_assets(base_env)
        self.track_env = True
        self._tracked_env_ids = torch.tensor(self.cfg.env_ids, device=base_env.device, dtype=torch.int32)
        self._update_play_dt()

        def _after_sim_step(substep, *args, **kwargs):
            ret = base_env.__class__._after_sim_step(base_env, substep, *args, **kwargs)
            if not self.track_env:
                return ret
            if (self.cfg.include_substeps or substep == base_env.cfg.decimation - 1):
                # add all but last substep frame. last substep frame is added after the step function call
                base_env._update_visualization_markers()
                frame = self._add_trajectory_frame_from_env(base_env, env_ids=self._tracked_env_ids)
                frame.timestep += 1
                frame.substep = substep
            return ret
        base_env._after_sim_step = _after_sim_step
        
        def step(*args, **kwargs):
            ret = base_env.__class__.step(base_env, *args, **kwargs)
            if not self.track_env:
                return ret
            extras = common_utils.to_numpy(base_env.extras)
            if "final_info" in extras:
                del extras["final_info"]
                del extras["final_observation"]
                del extras["_final_info"]
            self._frames[-1][-1].reward = base_env.reward_buf[self._tracked_env_ids].cpu().numpy()
            self._frames[-1][-1].ret = base_env._returns[self._tracked_env_ids].cpu().numpy()
            if self._frames[-1][-1].episode_reset:
                self._frames[-1][-1].ret = None
                self._frames[-1][-1].reward = None
            self._frames[-1][-1].info = tree_utils.slice(extras, self.cfg.env_ids)
            return ret
        base_env.step = step
        def _before_auto_reset(reset_env_ids: Sequence[int], *args, **kwargs):
            ret = base_env.__class__._before_auto_reset(base_env, reset_env_ids, *args, **kwargs)
            if not self.track_env:
                return ret
            overlapped_env_ids = reset_env_ids[torch.isin(reset_env_ids, self._tracked_env_ids)]
            self._frames[-1][-1].final_episode_step = True
            self._frames[-1][-1].reward = base_env.reward_buf[overlapped_env_ids].cpu().numpy()
            self._frames[-1][-1].ret = base_env._returns[overlapped_env_ids].cpu().numpy()

            extras = common_utils.to_numpy(tree_utils.slice(base_env.extras, overlapped_env_ids))
            if "final_info" in extras:
                extras = extras["final_info"]
                del extras["episode"]
            self._frames[-1][-1].info = extras
            return ret
        base_env._before_auto_reset = _before_auto_reset
        def _after_reset_idx(reset_env_ids: Sequence[int], *args, **kwargs):
            ret = base_env.__class__._after_reset_idx(base_env, reset_env_ids, *args, **kwargs)
            if not self.track_env:
                return ret
            if self.cfg.ignore_empty_trajectories and len(self._frames) > 0 and len(self._frames[-1]) > 0 and self._frames[-1][-1].episode_reset:
                self._frames.pop()
            self._frames.append(deque(maxlen=self.cfg.max_frames_per_trajectory))
            overlapped_env_ids = reset_env_ids[torch.isin(reset_env_ids, self._tracked_env_ids)]
            base_env._update_visualization_markers()
            frame = self._add_trajectory_frame_from_env(base_env, env_ids=overlapped_env_ids)
            frame.episode_id = self._episode_id_counter
            self._episode_id_counter += 1
            frame.episode_reset = True
            extras = common_utils.to_numpy(tree_utils.slice(base_env.extras, overlapped_env_ids))
            if "_final_info" in extras:
                del extras["_final_info"]
            if "final_observation" in extras:
                del extras["final_observation"]
            if "final_info" in extras:
                del extras["final_info"]
            frame.info = extras
            if frame.episode_id == 0:
                self.view_frame(0, self.viewed_env_id, self.viewed_trajectory_id)
            return ret
        base_env._after_reset_idx = _after_reset_idx
        self._update_info_htmls(base_env.cfg, task_id=self._task_id)

    def _update_info_htmls(self, env_cfg: BaseEnvCfg, task_id: str | None = None):
        self._env_info_html.content = f"""
        <div style="font-size: 0.875em; margin-left: .75em">
        {f"Task ID: {task_id}" if task_id is not None else ""}
        <div>Robot ID: {env_cfg.robot_id}</div>
        <div>Control Mode: {env_cfg.control_mode}</div>
        </div>
        """
        self._physics_info_html.content = f"""
        <div style="font-size: 0.875em; margin-left: .75em">
        <div>Physics dt: {env_cfg.sim.dt}</div>
        <div>Control dt: {env_cfg.sim.dt * env_cfg.decimation}</div>
        <div>Display Granularity: {"Sub step" if self._display_granularity_is_substep else "Control step"}</div>
        </div>
        """
    def _update_play_dt(self):
        if self._display_granularity_is_substep:
            self._play_dt = self._env_cfg.sim.dt
        else:
            self._play_dt = self._env_cfg.sim.dt * self._env_cfg.decimation
    def _update_slider_max(self, trajectory_id: int):
        if self._display_granularity_is_substep:
            self._slider.max = len(self._frames[trajectory_id]) - 1
        else:
            self._slider.max = len([f for f in self._frames[trajectory_id] if f.substep == self._env_cfg.decimation - 1])
        
    def _create_trajectory_frame_from_env(self, base_env: BaseEnv, env_ids: Sequence[int]):
        """adds frame using the environment data"""
        articulations = base_env.scene.articulations
        rigid_objects = base_env.scene.rigid_objects
        articulation_data = dict()
        
        rigid_object_data = dict()
        for k, articulation in articulations.items():
            actuated_joint_indices = []
            for actuator in articulation.actuators.values():
                actuated_joint_indices.extend(actuator.joint_indices)
            actuated_joint_indices = torch.stack(actuated_joint_indices)
            articulation_data[k] = dict(
                link_poses=dict(),
                joint_pos=articulation.data.joint_pos[env_ids].cpu().numpy(),
                actuated_joint_pos=articulation.data.joint_pos[env_ids][:, actuated_joint_indices].cpu().numpy()
            )
            for i, link_prim_path in enumerate(base_env.robot.articulation.root_physx_view.link_paths[0]):
                data = articulation.data.body_link_pose_w[env_ids, i]
                data[:, :3] = data[:, :3] - base_env.scene.env_origins[env_ids]
                articulation_data[k]["link_poses"][link_prim_path] = data.cpu().numpy()
        for k, rigid_object in rigid_objects.items():
            data = rigid_object.data.root_link_pose_w[env_ids]
            data[:, :3] = data[:, :3] - base_env.scene.env_origins[env_ids]
            rigid_object_data[k] = data.cpu().numpy()

        marker_data = {}
        env_ids_np = env_ids.cpu().numpy() if hasattr(env_ids, 'cpu') else np.asarray(env_ids)
        origins_np = base_env.scene.env_origins[env_ids].cpu().numpy()
        n_envs = base_env.num_envs
        n_tracked = len(env_ids_np)
        for name, marker in self._marker_objects.items():
            t = marker._viewer_last_translations
            if t is None:
                continue
            K = len(t) // n_envs  # instances per env e.g. point cloud markers
            t_local = t.reshape(n_envs, K, 3)[env_ids_np] - origins_np[:, None, :]  # (n_tracked, K, 3)
            q = marker._viewer_last_orientations
            if q is not None:
                q_tracked = q.reshape(n_envs, K, 4)[env_ids_np]  # (n_tracked, K, 4)
            else:
                q_tracked = np.zeros((n_tracked, K, 4), dtype=np.float32)
                q_tracked[..., 0] = 1.0  # identity wxyz
            marker_data[name] = np.concatenate([t_local, q_tracked], axis=-1)  # (n_tracked, K, 7)

        frame = Frame(articulations=articulation_data, rigid_objects=rigid_object_data, markers=marker_data, timestep=base_env.episode_length_buf[0].item())
        return frame
    def _add_trajectory_frame_from_env(self, base_env: BaseEnv, env_ids: Sequence[int]):
        """adds frame using the environment data"""
        frame = self._create_trajectory_frame_from_env(base_env, env_ids)
        self._frames[-1].append(frame)
        if len(self._frames) > 0:
            if self.cfg.ignore_empty_trajectories and len(self._frames[-1]) == 1:
                self._trajectory_id_slider.max = max(0, len(self._frames) - 2)
            else:
                self._trajectory_id_slider.max = len(self._frames) - 1
        self._update_slider_max(self.viewed_trajectory_id)
        return frame

    def clear_frames(self):
        self._frames = deque(maxlen=self.cfg.max_frames_per_trajectory)
