# Recorded Dataset Fields

Datasets are stored in [LeRobot v3 format](https://github.com/huggingface/lerobot). Execution records one row per engine tick (4 physics steps, 0.04 s), so episodes are written at 25 fps. All pose and motion quantities are expressed in the **robot base frame** (local frame) unless stated otherwise. The local frame is defined by the robot's base link position and orientation at the time of recording, with poses transformed via `world_to_local`.

Quaternions use `[qw, qx, qy, qz]` (scalar-first) convention. Euler angles use intrinsic ZYX (roll, pitch, yaw) in radians.

## Episode Metadata

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `is_first` | (1,) | bool | True for the first frame of the episode. |
| `is_last` | (1,) | bool | True for the last frame of the episode. |
| `is_episode_successful` | (1,) | bool | Whether the episode achieved the PDDL goal (set on last frame). |
| `pddl_goal` | (1,) | string | The PDDL goal expression for this episode. |
| `task_index` | (1,) | int64 | Index into `meta/tasks.parquet`, which holds the task description text (`"failed"` for failed episodes). |
| `glyph_map` | (1,) | string | JSON `{original_block_name: glyph}` for this episode. `pddl_goal` and the task text use the swapped glyphs, while the pose columns keep the original names; this map reconciles the two. Empty when no glyphs were swapped. |
| `block_variable_map` | (1,) | string | JSON `{rewritten_name: provenance}` covering both glyph blocks and colour cuboids, keyed by the rewritten name used in `pddl_goal` and the task text. Empty when nothing was rewritten. |
| `scene_id` | (1,) | string | Identifier for the scene configuration (table/object layout). |
| `robot_id` | (1,) | string | Identifier for the robot configuration used. |
| `source_problem` | (1,) | string | Name of the PDDL problem directory this episode originates from. |
| `instance_id` | (1,) | int64 | Index of the problem instance within the generated task. |
| `error` | (1,) | string | Error description if the episode failed (empty on success). |

## Robot State Observations

All state observations are in the **robot base frame**.

| Field | Shape | Dtype | Axes | Description |
|-------|-------|-------|------|-------------|
| `observation.state` | (8,) | float32 | `[joint_0..joint_6, gripper]` | Compact state: 7 joint positions (rad) + gripper position. |
| `observation.state.joint_position` | (7,) | float32 | `[joint_0..joint_6]` | Joint positions in radians. |
| `observation.state.joint_velocity` | (7,) | float32 | `[joint_0..joint_6]` | Joint velocities in rad/s. |
| `observation.state.joint_acceleration` | (7,) | float32 | `[joint_0..joint_6]` | Joint accelerations in rad/s². |
| `observation.state.joint_effort` | (7,) | float32 | `[joint_0..joint_6]` | Applied joint torques in Nm. |
| `observation.state.gripper_position` | (1,) | float32 | `[gripper]` | Gripper joint position (raw value from 8th joint). |
| `observation.state.ee_position` | (6,) | float32 | `[x, y, z, roll, pitch, yaw]` | End-effector pose: position (m) + Euler angles (rad), in robot base frame. |
| `observation.state.wrench` | (6,) | float32 | `[fx, fy, fz, tx, ty, tz]` | Wrist wrench: forces (N) and torques (Nm) at the wrist link. |
| `observation.state.current_stiffness` | (13,) | float32 | `[joint_0..joint_12]` | Active PD stiffness gains for all joints (arm + gripper). |
| `observation.state.current_damping` | (13,) | float32 | `[joint_0..joint_12]` | Active PD damping gains for all joints. |
| `observation.state.current_effort_limit` | (13,) | float32 | `[joint_0..joint_12]` | Current effort (torque) limits per joint in Nm. |
| `observation.state.current_velocity_limit` | (13,) | float32 | `[joint_0..joint_12]` | Current velocity limits per joint in rad/s. |
| `observation.state.current_torque` | (13,) | float32 | `[joint_0..joint_12]` | Applied torques for all joints (arm + gripper). |
| `observation.state.high_level_state` | (1,) | string | — | The symbolic (PDDL) state the current high-level action starts from, one ground fact per line, e.g. `(on block_k block_z)`. |

## Robot Link Poses

| Field | Shape | Dtype | Axes | Reference Frame | Description |
|-------|-------|-------|------|-----------------|-------------|
| `robot.poses.<link>` | (7,) | float32 | `[x, y, z, qw, qx, qy, qz]` | Robot base frame | Pose of each robot body, one field per link. `meta/recording_config.json` records the frame as `robot_link_pose_frame: robot_base`. |

## Scene Property Observations

| Field | Shape | Dtype | Axes | Description |
|-------|-------|-------|------|-------------|
| `observation.state.friction_properties` | (3,) | float32 | `[static_friction, dynamic_friction, restitution]` | Physics material properties of the manipulated object. |
| `observation.state.scale_properties` | (3,) | float32 | `[sx, sy, sz]` | Scale factors applied to the manipulated object. |
| `observation.state.cube_scale` | (1,) | float32 | — | Uniform scale factor of the cube (if applicable). |
| `observation.state.table_height_delta` | (1,) | float32 | — | Delta from default table height in metres. |
| `observation.state.table_friction` | (1,) | float32 | — | Friction coefficient of the table surface. |
| `observation.state.profile_id` | (1,) | int64 | — | Scene-profile slot the env was spawned in (see [env generation](env_generation.md)). |

## Object Pose Observations

| Field | Shape | Dtype | Axes | Reference Frame | Description |
|-------|-------|-------|------|-----------------|-------------|
| `observation.state.object_pose.<name>` | (7,) | float32 | `[x, y, z, qw, qx, qy, qz]` | Robot base frame | 6-DoF pose (position in m, quaternion) of each tracked object, transformed to the robot base frame. One field per object in the scene (e.g. `block_r`, `block_red1`). |

## Contact Force Observations

| Field | Shape | Dtype | Axes | Reference Frame | Description |
|-------|-------|-------|------|-----------------|-------------|
| `observation.state.contact_force.<sensor>` | (3,) | float32 | `[fx, fy, fz]` | Robot base frame | Net contact force vector (N) from the named contact sensor, rotated into the robot base frame. Present only when the scene defines contact sensors; the shipped datagen scenes define none. |

## Camera Observations

[Execution records no cameras by default](execution.md#what-a-recorded-episode-contains), so none of these columns exist in an executed dataset unless it was run with `--record-videos`. They normally come from [re-rendering](rerender.md).

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `observation.images.<cam>` | (H, W, 3) | video (uint8 RGB) | RGB image from the named camera, encoded as AV1 video at dataset fps. The resolution comes from the scene's camera config: 360×640 (H×W) for the shipped scenes' cameras. Camera names carry a `camera:` prefix, e.g. `observation.images.camera:front_camera`. |
| `observation.depth.<cam>` | (H, W, 1) or (H, W, 3) | video (uint16) | Depth map in millimetres (0–4095), divide by 1000 for metres. Stored as 12-bit `gray12le` (1 channel, the default) or as `rg8` (the same value packed into 3-channel 8-bit RGB: R = low byte, G = high byte). Read `video.depth_encoding` (`gray12le_mm` or `rg8_mm`) in the feature's `info` in `meta/info.json` before decoding. Source is Isaac Lab `distance_to_image_plane`. |
| `observation.images.<cam>_<variant>` | (224, 224, 3) | video (uint8 RGB) | Optional RGB sidecar from `--extra_rgb`, e.g. `_224`. |
| `observation.depth.<cam>_<variant>` | varies | video | Optional depth sidecar from `--extra_depth`, e.g. `_g12_224`, `_rg8_224` or `_rg8_55_224`. Has its own resolution, channel count and `video.depth_max_mm`; read them from `meta/info.json`. `rg8_55_224`, the pipeline default, packs depth as `(R << 5) \| G` and saturates at 1023 mm, unlike `rg8`'s `R + (G << 8)`, yet both record `video.depth_encoding: rg8_mm`; tell them apart by the key suffix or by `video.depth_max_mm` (1023 vs 4095). |
| `observation.camera_pose.<cam>` | (7,) | float32 | Camera extrinsic pose `[x, y, z, qw, qx, qy, qz]` in the robot base frame. |
| `observation.camera_intrinsics.<cam>` | (9,) | float32 | Flattened 3×3 camera intrinsic matrix `[fx, s, cx, 0, fy, cy, 0, 0, 1]` (row-major). |

Tactile cameras (if present) follow the same `observation.images.<cam>` convention but may have different resolution.

## Actions

All action fields represent **per-timestep commands** sent to or derived from the controller.

Row *t* holds the observation of state *t* together with the action that drives state *t* to state *t+1*. The last row of an episode is the terminal state: its action columns are zero and `action.is_valid` is `False`. Train on `action.is_valid` rather than testing for zeros, since an all-zero `action.joint_target` is itself a reachable pose.

| Field | Shape | Dtype | Axes | Description |
|-------|-------|-------|------|-------------|
| `action` | (9,) | float32 | `[joint_0..joint_6, gripper, hard_config]` | Primary action vector: `action.relative_joint_target` (7, rad) + `action.gripper_position` + `action.pd_mode` (labelled `hard_config`). This is the field typically consumed by policies. |
| `action.is_valid` | (1,) | bool | — | `False` only on the terminal row, whose action columns are zero. |
| `action.joint_position` | (7,) | float32 | `[joint_0..joint_6]` | Delta joint position: `current_pos - previous_pos` (rad). |
| `action.joint_target` | (7,) | float32 | `[joint_0..joint_6]` | Absolute joint position target commanded by the controller (rad). |
| `action.relative_joint_target` | (7,) | float32 | `[joint_0..joint_6]` | Joint target relative to previous joint position: `target - previous_pos` (rad). |
| `action.ee_target` | (7,) | float32 | `[x, y, z, qw, qx, qy, qz]` | End-effector pose of the commanded joint target (forward kinematics of `action.joint_target`), in robot base frame. |
| `action.relative_ee_target` | (7,) | float32 | `[x, y, z, qw, qx, qy, qz]` | `action.ee_target` relative to the previous end-effector pose: position difference (m) + rotation-difference quaternion. |
| `action.gripper_position` | (1,) | float32 | `[gripper]` | Binarized gripper command: 1.0 = open, 0.0 = closed (threshold at 0.3 by default). |
| `action.gripper_raw` | (1,) | float32 | `[gripper_raw]` | Raw continuous gripper target value before binarization. |
| `action.ee_position` | (6,) | float32 | `[x, y, z, roll, pitch, yaw]` | Delta end-effector pose: position change (m) + Euler angle change (rad), computed from consecutive EE poses in robot base frame. |
| `action.pd_mode` | (1,) | float32 | — | PD controller mode, from the stiffness profile closest to the current joint-0 stiffness: 0.0 for `soft` / `softer`, 1.0 for `stiff` / `medium` / `flip` (see `src/toyblocks/utils/stiffness_profiles.py`). |
| `action.high_level_action` | (1,) | string | — | Name of the current high-level (PDDL) action (e.g. `pick`, `place`, `reorient`). |
| `action.high_level_args` | (1,) | string | — | JSON list of the current high-level action's arguments, e.g. `["robot", "block_w", "right_flip", "o7", "o19", "workspace_table"]`. |
| `action.low_level_action` | (1,) | string | — | Name of the current low-level skill executor. |
| `action.low_level_args` | (1,) | string | — | JSON-encoded arguments of the current low-level executor. |

## Annotation Columns

[Annotation](annotate.md) adds these columns to a merged dataset, in place. They are not written during execution.

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `goal.action` | (1,) | string | The current action as text, e.g. `pick`, `place`, `flip right`, `rotate cw`, `push left`. |
| `goal.action_secondary` | (1,) | string | A secondary action string for the same frame. |
| `goal.position` | (2,) | float32 | `(u, v)` pixel coordinates of the current action's target object in the front camera. NaN when the frame has no target. |
| `goal.position_target` | (2,) | float32 | For push/align actions only: pixels of the block being pushed against. NaN otherwise. |
| `goal.position_gripper` | (2,) | float32 | `goal.position` in the gripper-camera image. NaN when the target is outside that image or the dataset has no gripper-camera columns. |
| `goal.position_target_gripper` | (2,) | float32 | `goal.position_target` in the gripper-camera image. |
| `goal.hl_prompt` | (1,) | string | Only with `--hl-prompt`: the given prompt, on every row. |

## Other Recording Paths

Datasets recorded outside vecgen, by evaluation or the `RecordEpisode` wrapper (`src/toyblocks/wrappers/record_episode.py`), can carry extra columns:

- `action.policy`: the raw policy action, sized to the env's action space.
- `action.hl_input_prompt`, `action.hl_model_output`, `action.hl_prev_action` (strings) and `action.hl_done` (bool), when a high-level model chooses the actions.
- `observation.state.object_mass` and `observation.state.table_mass`: live PhysX masses, NaN when the query fails.
- `observation.images.<cam>_hl_overlay`: the camera stream with the high-level prediction drawn on it.

## USD Assets and Provenance

**USD assets are not saved in the dataset.** Object meshes (e.g. letter-shaped toy blocks) and robot models are loaded at simulation time from the `assets/` directory (gitignored, distributed separately) via `OBJECT_ASSET_DIR` and `ROBOT_ASSET_DIR`. The dataset only contains rendered camera images, numeric state, and actions — not the raw 3D geometry.

However, the dataset does include **provenance metadata** under `meta/` that records what was used to generate the data:

| File | Description |
|------|-------------|
| `meta/recording_config.json` | Recording parameters: gripper binarize threshold, gripper type (`robotiq`), open/close commands, joint velocity limits, `record_videos`, and `robot_link_pose_frame`. |
| `meta/provenance.json` | Git commit/remote, the `recording_provenance` dict used at creation, and SHA-256 checksums of every file copied into `meta/extra/`. In a merged dataset it is one aggregate: the git metadata of the merge, `merged_from_n_datasets`, and one `sources[]` entry per source dataset (its path, `problem_name`, `extra_dir` and files). |
| `meta/extra/` | Copies of source files, only for the paths the recorder was given: `domain.pddl`, `<problem>.pddl`, and, from the env directory, `block_config.json`, `init.py` and the `<id>.json` instance files. A merged dataset keeps one `domain.pddl` (from the first source that has it) at `meta/extra/domain.pddl`, and each source's other copies under `meta/extra/<problem>/`. |
| `meta/render_params.json` | Re-rendered datasets only, before merge: how each split was rendered and every per-episode randomization value; see [Re-rendering](rerender.md#output). Merge does not carry it into the merged `meta/`. |

Vecgen passes only `instance_json_ids` and `generator: vecgen` as provenance, so for datasets from [execution](execution.md) `meta/extra/` is empty and `provenance.json` holds the git metadata alone. To reproduce such a dataset's simulation environment, you need:
1. The ToyBlocks repository at the recorded git commit (from `meta/provenance.json`).
2. The `assets/` directory with the matching USD meshes and robot models.
3. The PDDL problem and the env package it was generated from. The problem is named by the `source_problem` column; the dataset does not carry copies.
