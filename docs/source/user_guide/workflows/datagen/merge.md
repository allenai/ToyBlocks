# Stage 5: Merge

By this point the pipeline has produced one small LeRobot dataset per problem, from
[execution](execution.md) or [rendering](render.md). Training wants a single dataset
instead. Merge combines the per-problem datasets into **one training dataset**: it re-indexes
episodes, frames and tasks so they are contiguous, writes the data into parquet files of about
100 MB each, links every source video into place, and writes the merged metadata.

The videos are hardlinked rather than copied, so the merged dataset's videos take no extra disk
space. The flip side of that choice is covered under [Common pitfalls](#common-pitfalls).

The command-line tool is `scripts/datagen/merge_datasets.py`, a thin wrapper around
`toyblocks.vecgen.recording.merge`.

## Where it fits

- **Input:** a root directory containing per-problem datasets, from rendering (the usual
  case) or straight from execution.
- **Output:** one merged LeRobot dataset, read by [annotation](annotate.md) and then by training.

The pipeline always runs this stage. Run it by hand to re-merge after a patch, or to combine
datasets from several runs.

## Running it

Point `--input_dirs` at the directory that contains the `problem_*` directories, and
`--output_dir` at where the merged dataset should go:

```bash
python scripts/datagen/merge_datasets.py \
    --input_dirs data/datasets/unmerged/<experiment-id>/render/out/job_plain \
    --output_dir data/datasets/merged/train/<experiment-id> \
    --cameras all --num_workers 16
```

Pass the **root** that contains the `problem_*` directories, not a single dataset. Discovery
expands each input directory itself, as described in
[How sources are discovered](#how-sources-are-discovered). `--cameras all` keeps every video
stream, and `--num_workers` sets how many processes share the work.

## Through the pipeline

The pipeline exposes this stage through four wrapper flags: `--merge-cameras` (default `all`),
`--merge-num-workers` (default `16`), `--merge-output-dir` and `--merge-allow-missing-problems`.
The last one is not evaluated here but in the render stage; see
[Common pitfalls](#common-pitfalls).

The pipeline merges `<output-root>/datasets/unmerged/<experiment-id>/render/out/job_*` when the
render stage ran, and `.../execution` otherwise. The merged dataset lands in
`<output-root>/datasets/merged/train/<experiment-id>/` unless `--merge-output-dir` says otherwise
(`--output-root` defaults to `data`). The pipeline then copies its run config
(`run_datagen_pipeline_config_<timestamp>.json`) into the merged directory, and writes
`episode_0.png`, a plot of episode 0, next to it.

`run_datagen_pipeline.py` also hardcodes `--union_schema` on every merge invocation, so its own
`--merge-union-schema` flag is never read. `--merge-depth-frames` is read by nothing anywhere:
the merge does no `.npy` depth extraction.

## Output

The output directory holds one LeRobot v3 dataset, with re-indexed `data/` parquet files,
hardlinked videos and merged `meta/`.

To confirm it loads, open it with LeRobot:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local", root=str(merged_dir))
```

`LeRobotDataset(..., episodes=[0])` is the cheap version of this check. Loading a subset of
episodes requires the feature **order** in `info.json` to match the parquet column order, which is
why the merge reorders the non-video features.

Merge does its own LeRobot check when you pass `--validate` or `--union_schema`, but a failed
check does not fail the merge; see [Validation is advisory](#validation-is-advisory).

## How sources are discovered

You give merge a root directory, and it finds the datasets inside on its own. Discovery is
automatic and **keeps successes only**: for the vectorized layout, the merge takes
`base_dir/problem_*/blocks_success` and never `blocks_failure`.

**Merge reads the rendered tree, not the execution tree.** Once the render stage is marked
complete, the pipeline merges **`render/out/job_*` and nothing else**, and the execution tree is
no longer read. A problem that was executed but never rendered therefore vanishes from the
merged dataset. The local flow has [a three-state gate](local.md#what-counts-as-rendered) to
stop that from happening silently.

The glob is deliberately wide, so that every [render pass](render.md#render-passes) is picked up.
That means a `job_*` directory left by an **earlier** run is merged again, whether or not this run
produced it: dropping a `--render-pass` from an otherwise identical command keeps shipping that
pass's copies. The pipeline only warns about it, so clean up old job directories before
re-running.

## What `--union_schema` actually does

Different problems can record slightly different columns, and `--union_schema` controls how the
merged metadata describes them. It does **not** change which columns the merged parquet has:
that union always happens. It changes what `meta/info.json` *declares*, and it turns validation
on.

It can still change parquet *content* at the margin. When a source lacks a column, the merge
fills it with NaN, and the width of that fill is read from the declared features, so declaring
more can change the shape of those filled values.

Two consequences are worth knowing:

- **It decides whether LeRobot validation runs at all.** Without `--validate` and without
  `--union_schema`, the check is skipped, and the log says so.
- **The pipeline always passes it**, as described under [Through the pipeline](#through-the-pipeline).

## Validation is advisory

**A failed LeRobot validation never fails the merge.** The exception is caught, turned into a
warning, and `main()` still returns 0. So a merged dataset that no `LeRobotDataset` can open
still exits successfully, and you should read the warnings.

These are the obstacles to a `LeRobotDataset` load, in the order you are likely to hit them:

1. Without `--union_schema`, columns that some problems have but the template does not declare
   make the cast fail (`column names don't match` / `CastError`).
2. `dataset[0]` needs a working video backend. A torchcodec or ffmpeg failure is reported as
   "validation skipped, video decoding unavailable", not as a data defect.
3. Merged episodes whose clip was never rendered resolve to **zeros** rather than raising an
   error.

Consumers that read the parquet directly, such as HVLA's dataloader or pyarrow, are not affected
by the first one.

### Coverage warnings that do not fail the merge

| Warning | Meaning |
|---|---|
| `INCOMPLETE VIDEO COVERAGE` | A `(dataset, video key)` pair has more than zero but fewer clips than that dataset's `total_episodes`. Those episodes declare a video that is not on disk, **which a path-resolving dataloader reads as zeros**. |
| `present in N/M datasets` | A video key exists in some sources and not others. With `--cameras all`, the merged `info.json` can declare a key that only some problems rendered. |
| stray files | Files in a video directory not named `file-NNN.mp4`, such as an abandoned encode temp file, are reported and skipped. |

All three are log warnings only. Merge also does **not** check that the source and the render have
the same number of episodes.

Two checks *do* fail the run. After the data pass, the merge verifies every `data/` and `meta/`
parquet footer, and it checks that every planned data file and video link exists, with no
unplanned file present.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--input_dirs` | (required) | One or more roots holding the datasets to merge. |
| `--output_dir` | (required) | The destination. It is **deleted before writing**; see [Common pitfalls](#common-pitfalls). |
| `--cameras` | `all` | The video feature keys to keep, or `all` / `*` for every video feature any source declares. Naming cameras silently drops all the others, including depth streams and the `--extra_rgb` / `--extra_depth` sidecars. |
| `--num_workers` | `16` | The process-pool size for the data pass. |
| `--max_datasets` | `None` | Merge only the first N source datasets. `--episodes` is a backward-compatible alias. |
| `--union_schema` | off | Declare in `info.json` every non-video column that any source declares, plus columns present in the parquet that no source declared. **Implies `--validate`**, at `--validate_mode`'s default of `fast`. |
| `--validate` | off | Run the LeRobot loadability check after the merge. |
| `--validate_mode` | `fast` | `fast` checks the footers, `LeRobotDataset(episodes=[0])` and `dataset[0]`. `full` checks every frame. |
| `--validate_timeout` | `900` | Seconds before validation is abandoned; `0` means no limit. |

## Common pitfalls

### The output directory is deleted first

The merge `rmtree`s `--output_dir` before it starts, so never point it at anything you want to
keep. It also takes an exclusive lock at `<output_dir>.merge.lock`, a sibling of the output so its
own rmtree cannot remove it. A second merge into the same output refuses to start while the
lock's heartbeat is under 600 s; an older lock is taken over.

### `du` and deleting the source do not behave as expected

Because videos are hardlinked, `du -sh` on the merged directory tells you almost nothing about
real disk use, and deleting the unmerged source does not free that space while the merged tree
exists. Only a merge across filesystems copies. Because the links share inodes with the source,
the merge **raises an error rather than overwrite** an existing destination: overwriting would
write straight through into the source dataset.

This is also why you must never check `-N` multiplication by comparing bytes; see
[Rendering](render.md).

### An unreadable source disappears from the merge

A source dataset with any unreadable data parquet is dropped entirely, with a warning listing its
episode count, before offsets are assigned, so merged indices stay contiguous. The merge still
succeeds and those episodes are silently absent. Search the log for that warning.

### Offsets follow `info.json`, not the parquet footers

`info.json total_frames` decides the offsets. When it disagrees with the actual number of footer
rows, the merge warns ("offsets follow info.json") and continues.

### `--merge-allow-missing-problems` drops episodes permanently

This flag is evaluated in the render stage, not in the merge. Without it, a problem with a
finished execution but no render makes the run return 1 and leaves the render marker unset,
so `--resume` renders that problem. **With** it, the stage is marked complete and those
episodes are gone from the dataset permanently.

## Next

[Annotate](annotate.md) the merged dataset to add the goal columns that vision-language policies
train on.
