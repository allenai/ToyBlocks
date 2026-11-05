"""Source-episode enumeration shared by rerender.py and the local pipeline.

Enumerates a vecgen execution root's episodes in the canonical order rerender.py renders them
(sorted problems x [blocks_success, blocks_failure] x episode 0..n-1), with the per-root caches, and
resolves the member problems of ``problem_grp_*`` group directories (``rerender.py --split_groups``).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path


def _split_episode_count(sub: Path) -> int:
    """Episodes in one LeRobot split. meta/info.json is authoritative and free; fall back to the
    union of episode_index across the parquets when it is missing or unusable."""
    try:
        n = json.loads((sub / "meta" / "info.json").read_text()).get("total_episodes")
        if isinstance(n, int) and n > 0:
            return n
    except (OSError, ValueError, TypeError):
        pass
    import pandas as pd  # heavy; only on the fallback path
    ids = set()
    for f in sorted((sub / "data").glob("chunk-*/file-*.parquet")):
        try:
            ids.update(pd.read_parquet(f, columns=["episode_index"])["episode_index"].tolist())
        except Exception:                                  # unreadable parquet -> contributes 0
            pass
    return len(ids)


ENUM_CACHE_NAME = ".rerender_enum_cache.json"
ENUM_CACHE_SCHEMA = 1


def _enum_cache_load(root: Path, problems: list, splits_wanted: tuple):
    """Return the cached [(problem, split, ep)] list, or None if absent/stale/unusable.

    Validated against `problems` -- the problem_* dir names from the ONE root listing the caller
    already had to do. That listing is a single weka round-trip; the scan it replaces is ~6 per
    problem (exists + is_dir + read x 2 splits), which on a 2400-problem source is ~25 minutes of
    pure latency before the first worker can spawn.

    Staleness contract: this detects problems being ADDED or REMOVED. It does NOT detect a problem's
    episode count changing in place -- datagen sources are write-once, and rerendering never touches
    the source. If you do mutate one, delete <src>/.rerender_enum_cache.json.

    COVERAGE contract: a --skip_failure_episodes run only ever WALKS blocks_success, so the cache it
    leaves behind is a partial enumeration of the source. The cache file lives next to the SOURCE and
    is shared by every run against it, so without this check one skipped run would poison every later
    full run: it would cache-hit, get the success-only list, and silently plan against a global
    episode list missing ~45% of its rows -- with a fingerprint that matches on both sides, so the
    plan fence would not catch it either. Refuse any cache that does not cover the splits we want.
    Coverage is read from `scanned` when present and otherwise DERIVED from the rows, so caches
    already written by a skipped run (schema 1, no `scanned`) are correctly rejected rather than
    trusted -- that is why this does not need a schema bump, which would invalidate every good
    cache on weka and force a ~25-minute rescan per source.
    """
    f = root / ENUM_CACHE_NAME
    try:
        c = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if c.get("schema") != ENUM_CACHE_SCHEMA or c.get("problems") != problems:
        return None
    out = []
    try:
        rows = list(c["splits"])
        covered = set(c["scanned"]) if c.get("scanned") else {s for _p, s, _n in rows}
        if not set(splits_wanted) <= covered:
            print(f"[enumerate] cache covers {sorted(covered)} but this run needs "
                  f"{list(splits_wanted)}; rescanning", file=sys.stderr, flush=True)
            return None
        for prob, split, n in rows:
            out.extend((prob, split, e) for e in range(n))
    except (KeyError, TypeError, ValueError):
        return None
    return out


def _enum_cache_store(root: Path, problems: list, splits: list, scanned: tuple) -> None:
    """Best-effort write. A read-only or full source must not fail the run -- it just means the
    next run pays the scan again.

    `scanned` is the split tuple this walk actually visited, NOT the splits it happened to find.
    A --skip_failure_episodes run stores blocks_success only; recording that is what lets
    _enum_cache_load refuse to hand this partial list to a later full run."""
    try:
        # PID in the temp name: two runs against one source enumerate concurrently (a normal run
        # and a --skip_failure_episodes one, say), and a shared temp path makes them interleave
        # their writes into the SAME file before either renames it. rename(2) is atomic, but it
        # would then publish a spliced document -- so the per-writer temp is what actually makes
        # the line below atomic rather than merely last-writer-wins.
        tmp = root / f"{ENUM_CACHE_NAME}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"schema": ENUM_CACHE_SCHEMA, "src": str(root),
                                   "problems": problems, "splits": splits,
                                   "scanned": list(scanned)}))
        tmp.replace(root / ENUM_CACHE_NAME)          # atomic: a torn cache would be silently wrong
    except OSError as e:
        print(f"[enumerate] could not write {ENUM_CACHE_NAME} ({e}); next run rescans",
              file=sys.stderr, flush=True)


# --- vecgen groups --------------------------------------------------------------------------
# Vectorized datagen packs several PDDL problems into one env: execution writes ONE problem_grp_*
# dir per group, its episodes interleaved across the group's problems, each row carrying the
# problem it belongs to in `source_problem`. --split_groups renders every member problem as its own
# unit (a "virtual problem": the group's source and env, only that problem's episodes, written to
# out/<problem>/), so a 10-problem group becomes 10 one-wave items instead of one ten-wave item.
GROUP_PREFIX = "problem_grp_"
GROUP_CACHE_NAME = ".rerender_group_cache.json"
GROUP_CACHE_SCHEMA = 2  # 2: members of a <group>__sNNN shard carry the __sNNN suffix
_SHARD_SUFFIX = re.compile(r"__s\d+$")


def _group_split_members(sub: Path) -> dict:
    """{source_problem: sorted episode indices} for one split of a group, from its parquet.
    Episodes in an unreadable parquet have no known member and are left out (as rendering would)."""
    import pyarrow.parquet as pq  # heavy; only for group sources
    members: dict = {}
    for f in sorted((sub / "data").glob("chunk-*/file-*.parquet")):
        try:
            t = pq.read_table(f, columns=["episode_index", "source_problem"]).to_pydict()
        except Exception as e:                           # noqa: BLE001 - same as _split_episode_count
            print(f"[enumerate] {f}: unreadable ({type(e).__name__}); its episodes are skipped",
                  file=sys.stderr, flush=True)
            continue
        for ep, prob in zip(t["episode_index"], t["source_problem"]):
            members.setdefault(str(prob), set()).add(int(ep))
    return {p: sorted(v) for p, v in sorted(members.items())}


def group_members(src: str, splits_wanted: tuple, use_cache: bool = True) -> dict:
    """{group_dir: {split: {member_problem: [episode_index, ...]}}} for every problem_grp_* dir of
    `src` whose data carries `source_problem`. Cached in <src>/.rerender_group_cache.json under the
    same staleness and coverage contract as the episode cache: the planner scans once and every
    worker reads the cache instead of re-reading every group's parquet."""
    root = Path(src)
    groups = sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(GROUP_PREFIX))
    if not groups:
        return {}
    f = root / GROUP_CACHE_NAME
    if use_cache:
        try:
            c = json.loads(f.read_text())
            if (c.get("schema") == GROUP_CACHE_SCHEMA and c.get("groups") == groups
                    and set(splits_wanted) <= set(c.get("scanned", []))):
                return {g: {s: m for s, m in v.items() if s in splits_wanted}
                        for g, v in c["members"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    members: dict = {}
    for g in groups:
        for split in splits_wanted:
            sub = root / g / split
            if not ((sub / "meta" / "info.json").exists() and (sub / "data").is_dir()):
                continue
            m = _group_split_members(sub)
            # A recorder shard (<group>__sNNN) holds more episodes of the SAME member problems, so
            # its members carry the shard suffix: unique names, their own out/<problem>__sNNN dir
            # (which the merge takes as a shard of the problem), and the base problem's env.
            shard = _SHARD_SUFFIX.search(g)
            if m and shard:
                m = {f"{p}{shard.group(0)}": eps for p, eps in m.items()}
            if m:
                members.setdefault(g, {})[split] = m
    owners: dict = {}
    for g, v in members.items():
        for m in {m for split in v.values() for m in split}:
            if m in owners:
                raise SystemExit(f"[enumerate] --split_groups: problem {m} appears in both "
                                 f"{owners[m]} and {g}; member names must be unique")
            owners[m] = g
    if use_cache:
        try:
            tmp = root / f"{GROUP_CACHE_NAME}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"schema": GROUP_CACHE_SCHEMA, "groups": groups,
                                       "scanned": list(splits_wanted), "members": members}))
            tmp.replace(f)
        except OSError as e:
            print(f"[enumerate] could not write {GROUP_CACHE_NAME} ({e}); next run rescans",
                  file=sys.stderr, flush=True)
    return members


def _expand_groups(out: list, members: dict, splits_wanted: tuple) -> list:
    """Replace each group's rows with its member problems' rows: members sorted by name, each
    contiguous (split, then source episode index), so problem-aligned items cover whole members."""
    expanded, done = [], set()
    for prob, split, ep in out:
        if prob not in members:
            expanded.append((prob, split, ep))
            continue
        if prob in done:
            continue
        done.add(prob)
        by_split = members[prob]
        for member in sorted({m for v in by_split.values() for m in v}):
            for s in splits_wanted:
                expanded.extend((member, s, e) for e in by_split.get(s, {}).get(member, ()))
    return expanded


def enumerate_source_episodes(src: str, skip_failures: bool = False, split_groups: bool = False):
    """Canonical global episode list: [(problem, split, episode_index), ...] matching rerender.py's
    discover_jobs order (sorted problem_* dirs x [blocks_success, blocks_failure] x ep 0..n-1; or a
    flat dataset as a single 'blocks' split).

    With split_groups, each vecgen group's rows are replaced by its member problems' rows
    (see _expand_groups): the problem is the member's name and episode_index stays the index
    in the GROUP's split, which is what rerender.py's episode filter selects on.

    Result is cached in <src>/.rerender_enum_cache.json (see _enum_cache_load for the staleness
    contract). Set RERENDER_ENUM_CACHE=0 to bypass it entirely.
    """
    out = _enumerate_source_episodes(src, skip_failures)
    if not split_groups:
        return out
    splits_wanted = ("blocks_success",) if skip_failures else ("blocks_success", "blocks_failure")
    use_cache = os.environ.get("RERENDER_ENUM_CACHE", "1") not in ("0", "false", "no")
    members = group_members(src, splits_wanted, use_cache)
    if members:
        n = sum(len({m for v in g.values() for m in v}) for g in members.values())
        print(f"[enumerate] --split_groups: {len(members)} group(s) -> {n} member problem(s)",
              file=sys.stderr, flush=True)
    return _expand_groups(out, members, splits_wanted)


def _enumerate_source_episodes(src: str, skip_failures: bool = False):
    """The per-directory enumeration (see enumerate_source_episodes)."""
    root = Path(src)
    if not root.exists():
        raise SystemExit(f"source not found: {src}")
    use_cache = os.environ.get("RERENDER_ENUM_CACHE", "1") not in ("0", "false", "no")
    # Only blocks_success survives the merge (merge_success.get_dataset_paths), so rendering
    # blocks_failure is pure cost for a merge-bound run -- 45% of episodes on the reorient source.
    # MUST stay in lockstep with rerender.py's discover_jobs: --episode_offset indexes this exact
    # ordered list, so a filter applied on one side only silently re-targets every item.
    splits_wanted = ("blocks_success",) if skip_failures else ("blocks_success", "blocks_failure")

    def _n_eps(sub: Path) -> int:
        """EPISODE count for a split -- NOT the parquet-file count.

        Older sources wrote one file-NNN.parquet per episode, so the two were the same number and
        this counted files. aug13-style sources pack a whole split (250 episodes) into ONE
        file-000.parquet; counting files silently reduced such a problem to ~2 episodes and the
        fleet would have rendered 103 of 12800 episodes while reporting success. Keep this in step
        with rerender.py's _count_source_episodes -- item offsets index into this list.
        """
        return _split_episode_count(sub)

    def _is_ds(sub: Path) -> bool:
        return (sub / "meta" / "info.json").exists() and (sub / "data").is_dir()

    out = []
    problem_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("problem_"))
    if problem_dirs:
        names = [d.name for d in problem_dirs]
        if use_cache:
            cached = _enum_cache_load(root, names, splits_wanted)
            if cached is not None and skip_failures:
                # Filter rather than invalidate: the cache stores per-split rows, so the
                # success-only list is a strict subset and needs no rescan.
                cached = [t for t in cached if t[1] in splits_wanted]
            if cached is not None:
                print(f"[enumerate] cache hit: {len(names)} problems, {len(cached)} episodes "
                      f"(skipped the full scan; rm {root / ENUM_CACHE_NAME} to force a rescan)",
                      file=sys.stderr, flush=True)
                return cached
        # This walk is O(problems x splits) weka round-trips and MUST cover the whole source even
        # when only a few problems will be rendered: item offsets are absolute into this list
        # (see beaker_plan.select_items), so a filtered scan would re-target every slice. On a
        # 2400-problem source it can run for many minutes; without progress it is indistinguishable
        # from a hang, and a stuck-looking orchestrator invites someone to kill a healthy run.
        _t0 = time.time()
        splits = []                                   # (problem, split, n_eps) -> the cache payload
        _n = len(problem_dirs)
        # Announce the scan BEFORE it starts. The cache miss is the whole reason for the silence,
        # and saying so up front is what tells an operator the run is working rather than wedged.
        print(f"[enumerate] cache miss -- cold scan of {_n} problem(s) under {root}. "
              f"This is O(problems x splits) weka round-trips and can take many minutes; "
              f"progress follows.", file=sys.stderr, flush=True)
        # Progress on BOTH counts and time. The old rule (every 250, and only when >=500 problems)
        # left a 397-problem source completely silent for its whole scan -- 12 min of nothing on
        # sep12, which reads exactly like the hang the comment above warns about. Time-based
        # emission also covers a small source whose per-problem cost is high.
        _step = max(1, min(250, _n // 20))
        _last = _t0
        for _i, pd_ in enumerate(problem_dirs):
            _now = time.time()
            if _i and (_i % _step == 0 or _now - _last >= 30):
                print(f"[enumerate] scanned {_i}/{_n} problems, {len(out)} episodes "
                      f"({_now - _t0:.0f}s elapsed)", file=sys.stderr, flush=True)
                _last = _now
            for split in splits_wanted:
                sub = pd_ / split
                if _is_ds(sub):
                    _k = _n_eps(sub)
                    splits.append((pd_.name, split, _k))
                    for e in range(_k):
                        out.append((pd_.name, split, e))
        print(f"[enumerate] scanned {_n}/{_n} problems, {len(out)} episodes "
              f"({time.time() - _t0:.0f}s elapsed)", file=sys.stderr, flush=True)
        if use_cache:
            _enum_cache_store(root, names, splits, splits_wanted)
    elif _is_ds(root):
        for e in range(_n_eps(root)):
            out.append((root.name, "blocks", e))
    return out
