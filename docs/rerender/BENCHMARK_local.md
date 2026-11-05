# Local re-render: coupled vs pipelined, measured

Dev box, 2026-11-03. One RTX 5090 (32,607 MiB), 32 CPUs on the `powersave` governor (constant
across every arm), 60 GiB RAM. Branch `alvaro/rerender-local` @ `911b73f1`.

**Headline: the cheap change wins. `--rerender-local --local-gpu-workers 2` is 28% faster than
today's default. The pipelined scheduler is not a throughput feature on this hardware — it is an
observability and incrementality feature, and it costs nothing to have.**

## Method

Four arms, so that pipelining and oversubscription are not confounded — comparing
"coupled at 1 slot" against "pipelined at 2 slots" would have credited pipelining with a win that
belongs entirely to the extra slot.

| arm | flags | isolates |
|---|---|---|
| A | `--rerender-local --local-gpus 0` | today's behaviour, baseline |
| B | `--rerender --local-gpus 0` | scheduler overhead only, zero overlap |
| C | `--rerender-local --local-gpus 0 --local-gpu-workers 2` | oversubscription alone |
| D | `--rerender --local-gpus 0 --local-gpu-workers 2` | pipelining + oversubscription |

4 problems per arm, 2 repetitions, arms run strictly one at a time in A,B,C,D order within each
rep. Same `--task-gen-seed 42`, so every arm generates identical problems; 16 episodes, 8 envs,
`rerender_flags_camtest.yaml`, `--skip_cameras left_camera`. Timings come from the pipeline's own
`[sched]` log lines, so the window is **execution + re-render only** — task-gen, env and merge are
excluded. Harness: `bench_rerender.py` / `run_arms.sh` / `analyze_arms.py` (scratchpad).

## Results

| arm | mode | slots | mean wall | runs | vs A | exec s | render s | render % | overlap s | ttfr |
|---|---|---|---|---|---|---|---|---|---|---|
| A | coupled | 1 | 921s | 895, 948 | — | 921 | — | — | 0 | — |
| B | pipelined | 1 | 906s | 854, 958 | −1.6% | 270 | 636 | 70% | 0 | 247s |
| C | coupled | 2 | **667s** | 661, 673 | **−27.6%** | 1296 | — | — | 629 | — |
| D | pipelined | 2 | 733s | 693, 773 | −20.4% | 358 | 992 | 74% | 616 | 269s |

`exec`/`render` are sums of item-seconds, so they exceed wall when items overlap. The coupled arms
cannot separate them at all: the render lives inside the execution job's shell chain, so there is
one process span per problem covering both.

Run-to-run variance is ~5% at one slot (A: 895/948, B: 854/958) and under 1% at two slots
(C: 661/673). The two-slot arms are resource-bound and therefore more repeatable.

### B − A = −1.6%: the abstraction is free

Within noise, as designed. At one slot there is no overlap to be had — only reordering — so this
arm exists purely to price the scheduler. It costs nothing.

### C − A = −27.6%: oversubscription is the whole win

And it is three lines of code (`--local-gpu-workers N` multiplying the slot list). Mean GPU
utilisation rises from 43% to 68%, peak VRAM from 7.7 GiB to 15.4 GiB of 32.6 available.

### D − C = +9.9%: pipelining *loses* at two slots, consistently

Both reps (693 vs 661, 773 vs 673). This was the opposite of the prediction, and the cause is
measurable. Breaking the overlap down by what was running against what:

| arm | exec∥exec | exec∥render | render∥render |
|---|---|---|---|
| C (coupled) | 584s | — | — |
| D (pipelined) | 74s | 200s | **322s** |

The 322s of render∥render is a fact about the schedule: D's render-first ordering promotes renders,
so they end up beside each other. Whether that is what *costs* the 9.9% is **not established**, and
the obvious story — render∥render is the expensive pairing because both processes saturate the RTX
path, while execution is partly CPU-bound — does not survive being checked against the data.

Each item's life can be split into seconds spent solo, beside an execution, and beside a render, so
`solo + rate_exec·t_exec + rate_render·t_render = (its 1-slot duration)` is one equation per item and
the pairing costs are a least-squares fit. Pooled over both reps that gives a render 0.93 as fast
beside an execution and 0.46 as fast beside another render — exactly the expected shape. But the fit
is degenerate: 8 items, 2 parameters, and `t_exec`/`t_render` are collinear because the schedule
produces them together. Refitting per rep gives 0.79/0.59 and 1.58/0.18 — opposite answers. A
bootstrap puts rate_exec at [−0.24, 1.72] and rate_render at [0.13, 0.97]; P(render∥render is worse
than exec∥render) is 0.84, and P(two concurrent renders are slower than running them in sequence) is
0.60. That is a hint, not a mechanism. (`pairing.py` / `pairing_ci.py` in the scratchpad.)

What *is* solid from the same logs: at two slots a render takes 56% longer than it does alone
(159s → 248s) and an execution 32% longer (68s → 89s), and render is 70–74% of all item-seconds.
The D−C gap itself rests on 2 reps per arm — both D runs were slower than both C runs, which under
exchangeability is p ≈ 0.17.

Settling it needs a direct 2×2: time one execution alone, one render alone, exec∥render and
render∥render, several reps each. That is about half an hour of GPU and it is the only way to get
the pairing costs without the collinearity. Until then, treat "render∥render is the problem" as the
leading hypothesis and not as a result.

## Resources and integrity

| slots | peak VRAM | mean GPU util | peak host RAM |
|---|---|---|---|
| 1 | 7.7–8.3 GiB | 43% | 17–18 GiB |
| 2 | 15.4 GiB | 68% | **27.8 GiB** |

**Host RAM, not VRAM, is the binding constraint** — roughly +10 GiB per extra slot, so 3 slots is
about 38 GiB and 4 about 48 GiB of 60 GiB before page cache. VRAM at two slots is only 47% of the
card.

Every one of the 8 runs passed the integrity gate: `completed: true` on all four problems,
**64/64 episodes**, 256 mp4 files, no zero-byte videos, no `video_defects`, and
`validate_dataset_videos.py` clean. This gate is not optional — a host-RAM OOM mid-encode produces
`moov atom not found` videos inside a dataset whose `render_params.json` still reads
`completed: true`, so a faster arm that fails it would be a failed arm, not a fast one.

## Recommendation

**For throughput today: `--rerender-local --local-gpu-workers 2`.** 28% faster, integrity verified,
and it needs none of the scheduler.

**Keep the default at 1 worker.** Host RAM is the constraint and an OOM corrupts data silently, so
oversubscription should stay a measured opt-in rather than something every run inherits.

**The pipelined mode earns its place on grounds other than wall clock**, and these are real:

- **Attribution.** Coupled mode reports one span per problem; you cannot tell execution time from
  render time, which is how "render is 70% of the work" went unmeasured until now.
- **Failure isolation.** In coupled mode a render failure and an execution failure are the same
  non-zero exit from one `&&` chain, in one interleaved log. Pipelined gives two items, two logs,
  two exit codes.
- **Incremental progress.** First fully rendered problem at ~250–270s, versus not until a whole
  chain completes.
- **Bounded un-rendered disk.** Render-first ordering holds at most one un-rendered execution
  dataset at a time instead of all N.
- **Resume.** Re-render is a real stage for `apply_resume`, so a partial run resumes into the
  re-render rather than redoing execution.

## Follow-up worth doing

**Run the 2×2 pairing microbenchmark first.** It costs about half an hour and it decides whether
there is anything to fix; the observational fit above cannot.

If it confirms that exec∥render is the cheap pairing, the change is **slot anti-affinity** — prefer
an execution item when a render is already in flight, and vice versa. It needs the feed to know what
is currently running, which is a real design change rather than a flag flip. Note that simply
disabling render-first at >1 slot does *not* achieve it: plain FIFO still ends up pairing the
renders together once the executions drain.

Worth knowing before building it: if renders really do gain nothing from running two-up, then total
render time is a floor that no schedule on one GPU can go under, and these runs are already within a
few percent of it (636s of render against C's 667s wall). On this workload anti-affinity would buy
single digits at best, and `--local-gpu-workers 3` would buy nothing at all — the second slot's
entire 28% comes from hiding the executions behind the renders, and there are no executions left to
hide. The win would be on an exec-heavier mix (cheaper render settings, longer-horizon tasks) or on
more GPUs, where the render floor itself divides.
