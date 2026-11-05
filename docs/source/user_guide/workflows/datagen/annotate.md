# Stage 6: Annotation

A merged dataset records what the robot did, but a vision-language policy also needs to know, at
every frame, *what it is currently trying to do* and *where in the image to look*. Annotation adds
that information. It writes **per-frame goal columns** into a merged dataset, in place: the
current action as text, such as `pick` or `push left`, and the target object's pixel coordinates
in the front camera image.

The goal text comes from the high-level action labels recorded during execution, and the pixel
coordinates come from projecting the recorded object poses through the recorded front camera.

## Where it fits

- **Input:** a merged dataset from [stage 5, merge](merge.md), with recorded front-camera poses
  and intrinsics (normally from [re-rendering](rerender.md)).
- **Output:** the same dataset, with the `goal.*` columns added to its parquet files and to
  `meta/info.json`.

The pipeline runs this stage automatically. It is the last stage: after it, the dataset is ready
for training.

## Running it

Point the annotator at the merged dataset and give it a number of parallel workers:

```bash
python src/blocksuite/data_annotation/annotate_target_xy.py \
    data/datasets/merged/train/<experiment-id> -j 64
```

Run the file by its path like this; the pipeline does the same. There is no console script for it.

## Through the pipeline

The pipeline runs annotation automatically unless you pass `--skip-annotate`. Its
`--annotate-num-workers` flag (default `64`) becomes `-j`, and it annotates the merged dataset
in place at `<output-root>/datasets/merged/train/<experiment-id>/` (or `--merge-output-dir`). A
non-zero exit from the annotator fails the run.

The pipeline also skips this stage on its own, with a warning, when the rerender stage did not run
and `--vecgen-record-videos` was not set. Pixel goals need a camera, and
[execution records none by default](execution.md#what-a-recorded-episode-contains).

## Output

Annotation adds these columns, and declares them in `meta/info.json` as well:

- `goal.action` and `goal.action_secondary`
- `goal.position` and `goal.position_target`
- `goal.position_gripper` and `goal.position_target_gripper`
- `goal.hl_prompt`, when `--hl-prompt` is given

Their shapes and meanings are listed in the
[Annotation Columns table](dataset_fields.md#annotation-columns) of the dataset field reference.

A successful exit does not guarantee the columns are filled in, so check a parquet file after a
run: the `goal.action` strings should be non-empty and `goal.position` should not be all NaN. See
[Common pitfalls](#common-pitfalls) for why.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `path` (positional) | (required) | The merged dataset root, a directory of datasets (each is annotated), or a bare name resolved under `datasets/unmerged/` **at the repository root**. That is not the pipeline's `data/datasets/unmerged/`, so pass a real path for pipeline output. |
| `--workers`, `-j` | `os.cpu_count()` | The number of parallel workers. |
| `--skip-existing` | off | Skip datasets whose first data parquet already has the goal columns. |
| `--hl-prompt` | `None` | A high-level prompt string, written to a `goal.hl_prompt` column on every row. |
| `--allow-nominal-camera` | off | Annotate even when the dataset records no front-camera pose or intrinsics, falling back to built-in constants. Only correct for a few legacy presets; see [Common pitfalls](#common-pitfalls). |

## Common pitfalls

### Annotation refuses to run without a recorded front camera

By default the annotator **refuses** to run unless the sample parquet has **both**
`observation.camera_pose.camera:front_camera` and
`observation.camera_intrinsics.camera:front_camera`. Either one missing is enough to refuse. The
pipeline treats that non-zero exit as a hard failure of the run.

The two values fall back independently, so a dataset can be half fallen back while still
reporting `recorded`. Catching that case is exactly what this check is for.

`--allow-nominal-camera` bypasses the check, and is correct **only** for the legacy `table` and
`color_only_blocks_table` presets. On any other preset it annotates against the wrong camera, so
re-render the dataset instead.

### Annotation can succeed and write nothing useful

The only hard requirement is that the parquet has `observation.state.object_pose.*` columns. If
`action.high_level_action` or `action.high_level_args` is missing, the computation returns
immediately with **empty strings and all-NaN positions**. The goal columns are still written,
`info.json` still declares them, and the run still exits 0.

So "annotation succeeded" does not mean "the goals are populated". Check a parquet, as described
under [Output](#output).

### An interrupted annotation can corrupt the dataset

Each parquet file is rewritten in place with a plain `pq.write_table`. There is no temp file, no
atomic rename and no footer check, unlike the merge's own writer. **An interrupted annotation
leaves a truncated data file in the merged dataset.** If a run is interrupted, re-merge before
annotating again, or keep a copy of the merged dataset first.

### The goal columns are not covered by validation or stats

Annotation runs after the merge's LeRobot validation, so that check never sees the `goal.*`
columns, and `meta/stats.json` has no entries for them.

## Next

The dataset is now ready for training and for [evaluation](../evaluations.md).
