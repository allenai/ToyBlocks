import datetime
import json
from dataclasses import dataclass
from pathlib import Path

import tyro

import wandb

WANDB_PROJECT_NAME = "blocksuite-ci"
DATETIME_NOW = datetime.datetime.now().strftime("%m%d%y-%H:%M:%S")


@dataclass
class Args:
    output_dir: Path


def main() -> int:
    args = tyro.cli(Args)

    if not args.output_dir.is_dir():
        print(f"[ERROR]: seems the output dir @ {args.output_dir} is not a valid folder")
        return 1

    results_filepath = args.output_dir / "results.json"
    if not results_filepath.is_file():
        print(f"[ERROR]: couldn't find {results_filepath} file with the 'execute-plans' results")
        return 1

    data = {}
    with open(results_filepath, "r") as fhandle:
        data = json.load(fhandle)

    success_vids_front_folder = (
        args.output_dir / "blocks_success" / "videos" / "observation.images.camera:front_camera"
    )
    success_vids_front_paths = list(success_vids_front_folder.rglob("*.mp4"))
    success_vids_gripper_paths = [
        path.parent.parent.parent
        / "observation.images.camera:gripper_camera"
        / "chunk-000"
        / path.name
        for path in success_vids_front_paths
    ]

    failure_vids_front_folder = (
        args.output_dir / "blocks_failure" / "videos" / "observation.images.camera:front_camera"
    )
    failure_vids_front_paths = list(failure_vids_front_folder.rglob("*.mp4"))
    failure_vids_gripper_paths = [
        path.parent.parent.parent
        / "observation.images.camera:gripper_camera"
        / "chunk-000"
        / path.name
        for path in failure_vids_front_paths
    ]

    run_name = f"datagen-ci-test | {DATETIME_NOW}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        for key in ("total", "success", "failure", "success_rate"):
            if (run_stat := data.get(key)) is not None:
                run.summary[key] = run_stat

        for i_vid, (vid_front, vid_gripper) in enumerate(
            zip(success_vids_front_paths, success_vids_gripper_paths)
        ):
            log_data = {
                "video_outcome": True,
                "vid-front": wandb.Video(
                    vid_front, caption=f"success({i_vid}) - front camera", fps=30, format="mp4"
                ),
                "vid-gripper": wandb.Video(
                    vid_gripper, caption=f"success({i_vid}) - gripper camera", fps=30, format="mp4"
                ),
            }
            run.log(log_data)

        for i_vid, (vid_front, vid_gripper) in enumerate(
            zip(failure_vids_front_paths, failure_vids_gripper_paths)
        ):
            log_data = {
                "video_outcome": False,
                "vid-front": wandb.Video(
                    vid_front, caption=f"failure({i_vid}) - front camera", fps=30, format="mp4"
                ),
                "vid-gripper": wandb.Video(
                    vid_gripper, caption=f"failure({i_vid}) - gripper camera", fps=30, format="mp4"
                ),
            }
            run.log(log_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
