import datetime
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

    results_filepath = args.output_dir / "replay_results.json"
    if not results_filepath.is_file():
        print(f"[ERROR]: couldn't find {results_filepath} file with the replay results")
        return 1

    comparison_vids_dir = (
        args.output_dir / "comparison_videos"
    )
    comparison_vids_paths = list(comparison_vids_dir.rglob("*.mp4"))

    run_name = f"replay-ci-test | {DATETIME_NOW}"
    with wandb.init(project=WANDB_PROJECT_NAME, name=run_name) as run:
        for i_vid, vid in enumerate(comparison_vids_paths):
            log_data = {
                "replay-comparison": wandb.Video(
                    vid, caption=f"compare ({i_vid})", fps=30, format="mp4"
                ),
            }
            run.log(log_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
