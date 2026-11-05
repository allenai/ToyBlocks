# Evaluating the Hierarchical VLA on TAMP-Like Problems

This document proposes the task suite for training and evaluating our hierarchical VLA —
**Molmo2 finetuned on our data as the high-level (HL) planner** and **ManiFlow trained on our
data as the low-level (LL) policy** — on problems where success requires symbolic reasoning,
multi-step lookahead, and correct sequencing, not just reactive manipulation.

The central question: **can a hierarchical VLA solve problems that classically require a TAMP
planner?** Every task below is expressible in our existing blocks TAMP domain
(`pnp_table_table`, `pnp_table_block`, `pnp_block_block`, `pnp_block_table`, `push_to_right`,
`push_next_to`, `rotate`, `reorient`, `isolate`), so we can generate ground-truth demonstrations
and compute predicate-level success automatically.

Every task table carries a **Why it suits HL + LL** column. The neighbouring *Reasoning required*
column says what the task demands in the abstract; this one says why the task earns a slot in *this*
evaluation — which half of the hierarchy it loads, and what a failure on it would tell us. Tasks
that load only one half are as valuable as the hard ones: they are the controls that make failures
on the hard tasks attributable to the HL planner or the LL policy rather than to their combination.

---

## Scene Complexity Axis (applies to every task)

Every task family is evaluated in three scene settings:

| Setting | Description | What it tests |
|---|---|---|
| **Clean** | Only the blocks required by the goal are present | Pure planning + manipulation competence |
| **Few distractors** | 1–3 irrelevant blocks | Goal-relevant grounding; HL must identify which blocks matter |
| **Many distractors** | 6–12+ irrelevant blocks, cluttered workspace | Robust grounding, collision-aware placement, distractor blocks may occlude or physically obstruct the goal blocks |

In the *many distractors* setting, include variants where distractors **physically interfere**
(e.g., a distractor sits where the line must be built, or on top of a required block), so the
HL planner must emit clearing actions that are not part of the literal instruction. This is the
strongest test of lookahead — the required action sequence is longer than the instruction implies.

---

## TAMP Datagen Coverage

Every task below is marked with how far the datagen pipeline can take it. The distinction that
matters is **can the goal be written down at all** versus **can we build the starting scene** —
these are very different amounts of work, and the original version of this document conflated them.

| Mark | Meaning | Work needed |
|---|---|---|
| 🟢 | **Generable today.** A generator exists in `scripts/task_gen/`; most have produced data in `jul26_hvla`. | None |
| 🟡 | **Expressible, no generator yet.** The goal fits the supported predicates and the reset sampler can produce the initial state. | One new `task_gen` script |
| 🔵 | **Goal fine, initial state missing.** The goal is an ordinary predicate conjunction; the blocker is that the scene must *start* pre-arranged. | New reset-state sampler |
| ⛔ | **Not specifiable in the current system.** No way to write the goal, tagged with a blocker code below. | New predicate + vectorized checker, or a different supervision signal |

### What a goal can say

This is the hard constraint everything else follows from. A goal is a conjunction of literals
drawn from `SUPPORTED_GOAL_PREDICATES` in `src/blocksuite/tamp/blocks/pddl_goal_tensor_conversions.py`,
because `generate_task_from_pddl.py` needs a vectorized checker for every literal:

```
on   on_table   on_left   on_right   on_front   on_back   next_to   ori_of   clear   pick_obstructs
```

Every one of these is either **pairwise between two named blocks** or a property of a single named
block. There is no absolute position, no table region, no cardinality, and no size. `on_table_side`
is declared in `domain.pddl` but has no checker, so it cannot appear in a goal.

### Blocker codes for ⛔

- **G1 — no predicate for it.** The goal needs a spatial or quantitative relation outside the list
  above: table regions ("the left side", "the center"), 2D layouts and multi-block support
  (pyramid, square, grid), or comparisons over object properties. Needs a new predicate *and* a
  matching vectorized checker.
- **G2 — the goal is not a world state.** It counts actions ("flip it 3 times") or describes a
  process ("put everything back"). TAMP plans to a final state and will legitimately satisfy such
  a goal with fewer actions, or find it already satisfied.
- **G3 — the goal is open or disjunctive.** Goals of the form "any two red cubes" need `exists` or
  `or`. `generate_task_from_pddl.py` rejects quantified goals outright, and `(or ...)` is silently
  flattened into a conjunction by the goal parser, which is worse than failing. Verified against
  `scripts/task_gen/stack_red_blocks_v2.py`, whose `exists` goals fail this check while the
  concrete-block v1 generator passes.
- **G4 — the goal depends on reset-time randomization.** A PDDL goal is fixed per problem, but
  block size, table height, and friction are sampled *per env* from ranges. Worse, `block_size_range`
  yields **one size per env applied to every block** (`flip_env.py::_setup_scene_objects`), so blocks
  of differing sizes never coexist in a scene. Any goal referencing relative size is unwriteable.

### Note on "derived" goals

An earlier version of this document treated goals that must be *inferred* (alphabetical order,
arithmetic, largest digit, "spell an animal") as non-generable. That was wrong. The **generator**
computes the answer and emits concrete predicates; the instruction string carries the semantic
framing separately. Deriving the goal is the model's job at eval time, not the pipeline's. This is
exactly how `color_zebra_stack` already works — it is marked 🟢 and the HL must still induce the
alternation rule. These tasks are therefore 🟡, not blocked.

The one real constraint: the instruction template must not leak the answer, and the *held-out* split
has to be defined over the derivation (unseen digit sets, unseen words) rather than over goal syntax.

---

## 1. Letter-Block Spelling (stacks and lines)

Uses the full **A–Z letter block set** with orientation constraints (`OriOf`), building on
`scripts/task_gen/spelling.py`. The 26-letter vocabulary means the word space is effectively
unbounded — train on one word distribution and hold out words, letters, and word lengths for eval.

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🟢 Spell word as a **stack** | Word bottom-to-top, each letter face-up/readable | Scene: C, A, T scattered flat. Instruction: *"Stack the blocks to spell CAT, reading top to bottom."* Expected: reorient each letter upright, stack T, then A, then C on top | Ordering + orientation; must reorient letters before stacking | HL owns the letter order and the reorient-before-stack dependency; LL owns placement on a rising tower where one bump erases every earlier subgoal. Both halves are loaded and the failures look different |
| 🟢 Spell word as a **line** | Letters left-to-right in reading order (`push_next_to` / `NextTo`) | Scene: D, O, G scattered. Instruction: *"Arrange the blocks in a row to spell DOG."* Expected: D leftmost, O next to D, G next to O, all readable from the front camera | Spatial ordering, left-right convention | Symbolic ordering over an error-accumulating non-prehensile execution. The measured 36–59% rate means the LL ceiling is already known, so HL gains are readable against it |
| 🟢 **Held-out words** from trained letters | Words never seen in training, spelled with letters that were | Training covered CAT, DOG, SUN. Eval: *"Spell TAG"* — T, A, G all appeared in training words, but TAG never did | Compositional generalization of the HL planner | The LL skill set is identical either side of the split, so any drop is HL composition failure and nothing else |
| 🟢 **Held-out letters** | Words containing letters that never appeared in training goals (but were seen as distractors) | Letters J, Z never appeared in any training goal. Eval: *"Spell JAZZ"* | Letter grounding generalization | Separates the VLM's visual vocabulary from the planner's goal vocabulary: the letters were seen, just never as targets |
| 🔵 **Re-spell** (anagram) | Rearrange an existing word into a new one | Scene: stack reading TAC (T bottom, C top). Instruction: *"Make it spell CAT."* Expected: full unstack to table, restack in the order T, A, C bottom-to-top | Must *unstack* first — classic non-monotonic TAMP structure | Demands deliberately taking apart a satisfied arrangement, which a reactive policy has no reason to do. The teardown has to come from the HL |
| 🔵 **Fix the word** | Correct a mostly-right arrangement; the error is not pointed out | Scene: line reads C, A, T but the A is upside-down. Instruction: *"Spell CAT."* Expected: flip only the A, leave C and T untouched | HL must diagnose the delta between current and goal state | The HL has to compare observation against goal before acting, and the LL cost is a single reorient — close to a pure perception-into-planning probe |
| 🟡 **Alphabetical order** | Line up scene letters alphabetically | Scene: K, B, F scattered. Instruction: *"Line these up in alphabetical order."* Expected: B, F, K left to right — the order is never stated | Goal ordering must be *derived*, not stated | The goal is never stated, so it tests precisely what Molmo2 adds over the TAMP planner: prior symbol knowledge used to synthesize the target |
| 🟡 **Semantic word choice** | Spell a word satisfying a category constraint | Scene: {C, A, T, X, Q}. Instruction: *"Spell the name of an animal."* Expected: CAT; X and Q must be left alone | Open-ended goal synthesis by the HL — VLM knowledge beyond TAMP | TAMP can verify this goal but could never invent it; the strongest argument in the suite for a VLM head instead of a symbolic planner |
| 🟢 Letter **orientation only** | Make all letters readable without relocating them | Scene: E, M, R placed correctly but M is rotated 180° (reads as W) and E lies face-down. Instruction: *"Make all letters readable from the front."* Expected: two `reorient` actions, zero pick-and-place relocations | Perceptual grounding of orientation, `reorient` action selection | No relocation keeps LL variance low, so this reads out orientation grounding directly. At 82% it is also the calibration point for every reorient-plus-stack family |

## 1b. Digit-Block Tasks

Uses the **1–9 digit block set**. Digits add a symbolic-numeric dimension that letters don't:
the block's *label* can interact with counting, ordering, and arithmetic.

> **Asset limit: there is no `0` block.** The `train_blocks` library ships `1.usd`–`9.usd` only,
> so any number containing a zero (2026, 10, 40) cannot be built. `scripts/task_gen/make_number.py`
> rejects these up front rather than failing later in the pipeline.

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🟡 **Ascending / descending line** | Order the digit blocks present by value | Scene: 7, 2, 5 scattered. Instruction: *"Line the digits up from smallest to largest."* Expected: 2, 5, 7 left to right | Numeric ordering derived from labels | The ordering comes from label semantics rather than the instruction, while the LL side is plain pushes, so failures land on the HL by construction |
| 🟢 **Make a number** | Compose a multi-digit number as a line/stack | Scene: 1, 3, 7 plus distractor digits. Instruction: *"Spell 137 in a row."* Repeated digits need a second copy (121 uses `block_1` and `block_1_2`) | Multi-digit sequencing (digit blocks reused only if duplicates exist) | The instruction states the answer outright, making this the control condition against which the derived-goal digit tasks are scored |
| 🟡 **Digit-count matching** | Convert a displayed symbol into a cardinality | Scene: digit block 4 and six plain red cubes. Instruction: *"Stack as many red cubes as the digit shows."* Expected: a stack of exactly 4, two cubes left over | Reading a symbol and converting it to a cardinality — symbol/quantity binding | Binds a read symbol to a count and then requires stopping at it, two failure modes that flat policies show clearly and that the LL cannot cause |
| 🟡 **Arithmetic goal** | Select the block satisfying an arithmetic expression | Scene: digits 3, 5, 8. Instruction: *"Put the block that shows 2+3 on the green cube."* Expected: the 5 block moves; 3 and 8 stay | Simple arithmetic grounded in block selection | One pick-and-place with all the difficulty in selection: cheap to run, and essentially every failure is attributable to the HL |
| 🔵 **Sequence completion** | Extend a numeric sequence found in the scene | Scene: line already reads 2, 4, 6; digits 7, 8, 9 available nearby. Instruction: *"Continue the sequence."* Expected: 8 placed at the end | Rule inference over numeric labels | The rule lives in the scene rather than the prompt, so it tests whether the HL genuinely conditions on what it sees instead of on instruction templates |
| ⛔ **G1** **Odd/even sorting** | Partition digits by parity | Scene: 1, 2, 5, 8. Instruction: *"Put odd digits on the left side and even digits on the right."* Expected: {1, 5} left, {2, 8} right | Category derived from label semantics — but "left side" / "right side" are table regions, and no region predicate exists | Parity is knowledge only the HL can supply and the manipulation is trivial, so it would be a clean HL probe once a region predicate exists |
| 🟡 **Nearest / extreme selection** | Compare labels and act on the extreme | Scene: 3, 6, 9 and a red cube. Instruction: *"Put the largest digit on the red cube."* Expected: 9 on red; 3 and 6 untouched | Comparison across all visible labels | A comparison over every visible label resolved into a single action — the cheapest pure-HL probe in the suite |

## 2. Color-Pattern Construction

Uses uniform colored cubes (red, green, blue, plus extras if available).

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🟢 **Alternating / zebra stack** | Stack alternating colors | Scene: 2 red + 2 green cubes. Instruction: *"Make a zebra stack."* Expected: R-G-R-G (or G-R-G-R) bottom-to-top; either is valid | Pattern rule induction; block *identity* doesn't matter, only color class | The rule generalizes over color classes rather than block identities, so it tests whether the HL abstracts; the LL side is ordinary stacking it already does well |
| 🟢 **Alternating line** | Same rule but horizontal, aligned row | Scene: 3 red + 3 blue cubes. Instruction: *"Line them up alternating red and blue."* Expected: R-B-R-B-R-B, aligned and touching | Pattern + spatial alignment | Currently an LL and executor probe more than a planning one — 4 episodes out of 256 means execution dominates the signal until the push executor improves |
| 🟢 **Rainbow / fixed color order** | Stack or line in a specified color order | Scene: red, green, blue cubes. Instruction: *"Stack them like a rainbow: red on the bottom, then green, then blue."* | Instruction-to-order mapping | The order is handed over in the instruction, making this the LL-isolating control for the two induced-pattern rows around it |
| 🔵 **Pattern completion** | Continue a partial pattern found in the scene | Scene: line already reads R-G-R; one green and one blue cube nearby. Instruction: *"Continue the pattern."* Expected: green appended (R-G-R-G); blue would be wrong | Rule inference from the *scene*, not the instruction | The HL must read the rule off the table and commit to the one correct continuation; picking the wrong color is an unambiguous planning error, not an execution slip |
| ⛔ **G3** **Pattern with choice** | Satisfy a pattern rule when multiple valid assignments exist | Scene: 3 red + 3 green cubes. Instruction: *"Make an alternating stack of 4."* Any of several block choices works, but the HL must not mix them mid-stack | HL must commit to a consistent assignment (any valid one) — needs `exists`, which the pipeline rejects | Tests plan coherence across time: the HL must hold one valid assignment instead of drifting mid-stack, which is the characteristic open-loop VLA failure |
| 🟢 **Symmetric / mirrored line** | Build a palindromic color sequence | Scene: 2 red, 2 green, 1 blue. Instruction: *"Make a line that reads the same from both ends."* Expected: e.g. R-G-B-G-R | Longer-horizon ordering with a non-trivial rule | A longer horizon under a rule constraining both ends, so subgoal-fraction metrics stay informative where binary success would sit at zero |
| ⛔ **G1** **2D shapes** | Arrange blocks in an L, a square, a 2x2 grid, or a pyramid | Scene: 3 cubes. Instruction: *"Build a pyramid."* Expected: two cubes placed touching side-by-side first, third centered on top — top-first ordering fails | Spatial layout planning; `on` is single-support, so "resting on two blocks at once" has no encoding, and a 2x2 grid needs simultaneous x and y relations | Support relations the symbolic vocabulary cannot express, forcing the HL to plan geometrically rather than over predicates — and the LL to place with real precision |

## 3. Counting & Numerosity

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🟢 **Stack exactly N** | Stack N blocks with more than N available | Scene: 6 cubes. Instruction: *"Stack 3 blocks."* Expected: one 3-stack, 3 cubes untouched — stacking a 4th is a failure | Cardinality; knowing when to *stop* | Deciding when to stop is a call no reactive controller makes well, over-running is trivially detectable, and N extrapolates cleanly beyond the training range |
| 🟢 **Stack N of a color** | Count restricted to a category | Scene: 4 red + 3 blue cubes. Instruction: *"Stack 3 red blocks."* Expected: 3-stack of red only; any blue in the stack fails | Counting restricted to a category | Couples a count with a category filter, so grounding errors and counting errors leave distinguishable end states and can be scored apart |
| ⛔ **G2** **Flip a block N times** | Repeat `reorient` a fixed number of times | Scene: one red cube. Instruction: *"Flip the red block 3 times."* Expected: exactly 3 flips (odd count → net orientation change is checkable) | Counting *actions*, not objects — the goal is a trajectory property, and the planner will reach the same `ori_of` in one flip | Would test whether the HL can emit one subgoal repeatedly while tracking its own progress; the LL primitive is already solved, so the count is the whole task |
| ⛔ **G2** **Rotate N quarter-turns** | Repeat `rotate` with a specified increment | Instruction: *"Rotate the blue block 90° twice."* Expected: net 180° yaw; one 180° rotation also acceptable if action counting isn't enforced | Same as above: only the net `ori_of` is checkable, so "twice" is unsupervisable | The same repetition-tracking probe on a different primitive, with net orientation available as at least a partial check |
| ⛔ **G1** **Move N to a region** | Relocate a count of blocks to a spatial region | Scene: 5 cubes in the center. Instruction: *"Push 2 blocks to the right side of the table."* Expected: exactly 2 in the right region | **Was previously marked 🟢 in error.** "The right side of the table" is an absolute region; `on_table_side` has no vectorized checker and no generator emits it. Only block-relative encodings (`on_right(a, b)`) are available, which is a different task | Pairs a count with a destination, which is how most real household instructions are phrased, so it is worth having once regions exist |
| 🟢 **Split into groups** | Partition into equal groups | Scene: 4 cubes. Instruction: *"Make two stacks of 2."* Expected: two separate 2-stacks; one 4-stack fails | Partitioning; multiple simultaneous subgoals | Two subgoals advance in parallel, so the HL must remember which stack it is on — an interleaving test that single-goal tasks cannot produce |
| 🔵 **Numeric comparison** | Satisfy a relational cardinality constraint | Scene: 2-stack on the left, 2-stack on the right, 1 loose cube. Instruction: *"Make the left stack taller than the right one."* Expected: loose cube onto the left stack (or move one from right to left) | Relational, not absolute, cardinality. The generator can pin it to one concrete `on` literal, but the scene must start with two built stacks | The target depends on the scene, so the HL has to measure before acting rather than replaying a template, and more than one action sequence is correct |

## 4. Rearrangement Requiring Lookahead (the core TAMP stress tests)

These are the tasks where greedy/reactive policies fail and planning is mandatory.

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🔵 **Buried target** | Move a block that is currently underneath others | Scene: stack is yellow-on-blue-on-**red** (red at the bottom); green cube on the table. Instruction: *"Put the red block on the green block."* Expected: unstack yellow, unstack blue, then move red onto green — 3 actions for a 1-action instruction | Must unstack blockers first (subgoal not in the instruction) | A one-clause instruction with a three-action solution: the most direct measure of whether the HL emits subgoals nobody asked for, on primitives the LL already has |
| 🔵 **Swap two blocks** | Invert a 2-block stack | Scene: red on blue. Instruction: *"Swap them so blue is on red."* Expected: red → table, blue → red. A policy that grabs blue first is stuck (red is under it) | Needs a temporary table placement — no single action achieves the goal | Requires a temporary placement that satisfies no goal literal, so the HL must accept apparent regression — exactly what imitation-trained flat policies avoid |
| 🔵 **Sussman-style anomaly** | 3-block goal where greedy subgoal order undoes progress | Scene: C is on A; B alone on the table. Instruction: *"Stack them so A is on B and B is on C."* Greedy trap: satisfying "A on B" first (move C off A, put A on B) must be undone to get B onto C. Correct plan: C → table, B → C, A → B | Goal-interaction reasoning; the hallmark TAMP benchmark | Greedy subgoal ordering fails by construction, so it separates genuine planning from plausible-looking action sequencing better than any other task here |
| 🔵 **Occupied destination** | Place a block at a location currently occupied | Scene: blue sits on the marked target region; red nearby. Instruction: *"Put red where blue is."* Expected: move blue aside first, then place red | Must move blue away first. Encode the destination as a third block rather than a marked region, or this becomes **G1** | The blocker is plainly visible and the fix is one extra action, making this the minimal lookahead probe — cheap enough to run at every checkpoint |
| 🔵 **Reversal** | Invert a full stack | Scene: stack reads A-B-C bottom-to-top. Instruction: *"Reverse the stack."* Expected: full teardown (C → table, B → table), then rebuild C, B, A — minimum 4–5 pick-and-places | Full teardown + rebuild, longest-horizon variant | The longest horizon in the suite over a fixed skill set, so success against stack height traces the point where the hierarchy stops holding its plan |
| 🟢 **Conditional goal** (`r_on_red.pddl` style) | Target chosen among candidates by a category condition | Scene: letter R block, two red cubes, one blue cube. Instruction: *"Put the R block on a red block."* Either red cube is valid; the blue one is not | Grounding a category-level goal to a specific object. Note the generator names *one* concrete red cube; the "either is valid" framing lives in the instruction only | Category-to-instance grounding resolved in a single action, isolating HL object selection from LL execution; at 200 episodes it is the best-supported task in this family |

Every goal in this family is already writeable — plain `on` / `on_table` conjunctions over named
blocks. The single blocker is uniform: the scene must **start pre-stacked**, and the reset-state
sampler does not produce that. One stacked-initial-state sampler turns this entire family 🟢, which
makes it the highest-value datagen extension in this document — five tasks for one piece of
infrastructure, and the planner already solves them.

## 5. Sorting & Spatial Organization

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| ⛔ **G1** **Sort by color into regions** | Partition all blocks by color into table regions | Scene: 3 red + 3 blue cubes interleaved. Instruction: *"Put all red blocks on the left and all blue blocks on the right."* Expected: 6 relocations (minus any already in place) | Category-wise iteration over objects — blocked purely on the missing region predicate, not on the sorting itself | Iteration over a set until it is exhausted: the HL has to track which blocks it has already handled, which no single-object task exercises |
| 🟢 **Isolate** (`isolate.pddl`) | Move one block away from all others | Scene: 4 cubes clustered together. Instruction: *"Move the green block away from the others."* Expected: green ends up with clearance from every other block | Spatial predicate reasoning (negative constraints) via negated `pick_obstructs`. Generable in principle, but **the planner timed out on all 256 instances** in `jul26_hvla` — see the run notes | The one task where TAMP itself exhausts its budget, so a learned HL that solves it would beat the planner that supervises everything else. Needs a non-TAMP data source first |
| 🟢 **Push-align** (`push_align.pddl`) | Push blocks into an aligned row | Scene: 3 cubes at scattered y-offsets. Instruction: *"Push the blocks into a straight line."* Expected: pushes ordered so earlier-aligned blocks aren't knocked out of place | Non-prehensile planning, ordering pushes to avoid collisions | Non-prehensile and LL-dominated with a checkable ordering requirement — a useful counterweight to a suite otherwise heavy on pick-and-place |
| ⛔ **G1** **Clear a region** | Empty a marked region | Scene: 3 cubes in the table center, 2 outside it. Instruction: *"Clear the center of the table."* Expected: exactly the 3 center cubes move out; required action count depends on the scene, not the instruction | Goal defined by *absence* over a region — needs both a region predicate and negation support | The goal is stated as an absence, so the HL reasons about what should not be there instead of matching a target arrangement — and the action count comes from the scene |
| ⛔ **G4** **Line up by size** | Order blocks by physical size | Scene: 3-, 4-, 5-cm cubes scattered. Instruction: *"Line them up from smallest to largest."* Expected: 3-4-5 left to right | Not just a missing predicate: `block_size_range` samples **one size per env shared by all blocks**, so a scene containing differently-sized cubes cannot be built, and the correct ordering would not be known until reset time anyway | Ordering by a perceived physical property rather than a printed label, which tests visual estimation instead of symbol reading — the only task here that does |

## 6. Collision & Physical-Feasibility Reasoning

Tasks where the *symbolic* plan is trivial but geometric/collision reasoning determines success.
These probe whether the HL accounts for physical feasibility — the "motion" half of TAMP.

| Task | Goal | Example | Reasoning required | Why it suits HL + LL |
|---|---|---|---|---|
| 🔵 **Obstructed placement** | Place a block where the only valid spot is occupied | Scene: A and B near the table edge; the only free spot beside B holds a yellow distractor. Instruction: *"Put A next to B."* Expected: move yellow away first, then place A — the clearing action is never mentioned | Must move the distractor first — a clearing action derived from geometry, not the instruction. Goal is a plain `next_to` | The symbolic plan is one action and geometry forces a second, so it asks whether the HL reasons about feasibility at all — the cheapest test of the motion half |
| 🔵 **Pick order in clutter** | Retrieve a block tightly surrounded by others | Scene: target red cube boxed in by 4 touching neighbors, gripper can't descend. Instruction: *"Pick up the red cube and put it on the plate."* Expected: nudge/move one neighbor (or pick an angled approach) before grasping | Choosing which neighbors to move (or approach direction) to make the grasp feasible. Goal is a plain `on` | Loads both halves at once: the HL picks which neighbor to clear from geometry alone, and the LL still has to make a grasp with almost no margin |
| 🔵 **Blocked push path** | Push a block along a corridor containing an obstacle | Scene: cube A on the left, goal region on the right, cube B sitting midway on the straight-line path. Instruction: *"Push A to the right side."* Expected: move B out of the corridor first, or route the push around it | Corridor reasoning for non-prehensile actions. Express the destination as `on_right(A, ref_block)`; phrasing it as a table region makes it **G1** | The LL executes pushes essentially open-loop, so a bad HL choice fails visibly instead of degrading gracefully — a sharp, well-attributed signal |
| 🔵 **Build near existing structure** | Extend an arrangement without toppling completed work | Scene: a finished 3-stack; instruction: *"Place the yellow cube right next to the stack."* Expected: approach and placement that don't clip the stack — knocking it over fails even though the placement predicate would hold | Placement and approach must not topple prior work. Needs the pre-stacked sampler; the "don't topple" part is enforced by the existing `on` checks on the stack | LL precision dominates and the failure mode is unmistakable; the HL only has to recognize that nearby completed work is fragile, so this reads out the LL almost alone |
| ⛔ **G1** **Tight packing** | Fit N blocks into a region barely large enough | Scene: marked tray region fitting exactly 3 cubes; 3 cubes outside. Instruction: *"Put all three cubes in the tray."* Expected: fill back/corner slots first — placing the front cube first blocks access to the back slot | Placement ordering (insert from the back/corner first) — requires a bounded region predicate, and there is no tray asset | The clearest case of motion-level constraints dictating symbolic decisions: the correct order follows from reachability, not from the goal |
| ⛔ **G1** **Reachability-forced staging** | Goal location at the edge of the reachable workspace | Scene: goal region at the far edge; cube starts at the near edge on the opposite side. Instruction: *"Put the cube on the marked spot."* Expected: an intermediate re-grasp/staging placement mid-table when a single carry is kinematically infeasible | Recognize kinematic limits and re-stage. The goal is an absolute table pose, which no predicate can express | Forces the HL to model its own embodiment limits — something a symbolic planner gets for free from the motion layer but a learned planner must acquire from data |
| ⛔ **G4** **Unstable intermediate states** | Stack where base choice determines stability | Scene: one large flat block and one small block. Instruction: *"Stack these two."* Expected: small on large; large on small topples | Requires two differently-sized blocks in one scene, which the shared per-env `block_size_range` cannot produce | Needs a physical stability prior that no predicate encodes, which is exactly the kind of common sense a pretrained VLM should contribute over a planner |

Most of this family is **goal-writeable but scene-blocked** (🔵): the goals are ordinary `on` /
`next_to` literals, and the missing piece is an **adversarial initial-state sampler** that places an
obstacle at the goal spot, in the push corridor, or surrounding the pick target. CuRobo already does
the collision checking, so the planner would solve these scenes if we could sample them.

The three ⛔ entries are different in kind and will not be fixed by a sampler: they need an absolute
position or region predicate (tight packing, reachability staging) or heterogeneous block sizes
(unstable stacking), neither of which the current system has.

The clean / few / many distractor axis interacts strongly with this family: with many distractors,
*every* task from families 1–5 gains an implicit collision-reasoning component. This means TAMP
datagen on green tasks with many distractors **already produces collision-reasoning supervision
organically**, even though the targeted probes above are eval-only.

## 7. Compositional / Chained Instructions

Concatenate primitives to test long-horizon instruction following:

- 🟢 "Spell CAT, then flip the Q block." (independent subtasks — order-free; the combined goal is
  still a single predicate conjunction, so TAMP datagen works)
  *Why it suits HL + LL:* both skills are trained and the clauses do not interact, so a failure
  here is multi-clause language parsing rather than planning — the lowest rung of composition.
- 🟢 "Stack 3 red blocks, then put the A block on top." (dependent ordering, but still expressible
  as one final-state goal)
  *Why it suits HL + LL:* the second clause is only executable after the first, so the HL must
  sequence rather than interleave, and per-clause scoring localizes exactly where the chain breaks.
- ⛔ **G1** "Sort blocks by color, then make a zebra stack from the left group." (the zebra half is fine;
  "the left group" is a table region, which has no predicate)
  *Why it suits HL + LL:* the second clause refers to a set the first clause created, forcing the
  HL to plan against its own predicted future state rather than the observed one.
- 🟡 "Line up the digits ascending, then put the letter matching the count on top of the last one."
  (cross-family: digits + counting + letters. The generator computes both the ordering and the
  matching letter, then emits one `on_right` chain plus one `on` literal — a single conjunction)
  *Why it suits HL + LL:* fuses three families never paired in training, with each component
  independently measurable, so compositional failure can be separated from component failure.
- ⛔ **G2** "Undo": build a stack, then "put everything back on the table." (temporal goal; the final
  state is just `on_table` for everything, which the initial state already satisfies, so a
  state-goal planner emits the empty plan and supervises nothing)
  *Why it suits HL + LL:* reverting requires remembering what was done, which is the one place a
  learned HL with episode context could outperform the state-goal planner that supervises it.

---

## Blocker Summary

Across the 53 tasks in this document:

| Mark | Count | Meaning |
|---|---|---|
| 🟢 | 18 | Generable today |
| 🟡 | 7 | Expressible; needs one new generator script |
| 🔵 | 14 | Goal is writeable; needs a new initial-state sampler |
| ⛔ | 14 | Not specifiable in the current system |

**The single highest-leverage change is an initial-state sampler that can start a scene pre-arranged**
(blocks already stacked, a distractor already occupying the goal spot). That one piece of
infrastructure unlocks 14 tasks — the entire lookahead family plus most of the collision family —
without touching the predicate vocabulary or the planner. Writing the 7 🟡 generators is cheaper per
task but adds no new capability.

The 14 ⛔ tasks break down by what they would actually require:

| Code | Count | Tasks | What would unlock them |
|---|---|---|---|
| **G1** — no predicate | 8 | Odd/even sorting, 2D shapes, move N to a region, sort by color into regions, clear a region, tight packing, reachability staging, "sort then zebra" | A region / absolute-pose predicate with a vectorized checker. Six of the eight are blocked *only* on the phrase "left side" / "the center" / "the tray" — a single `in_region(?b, ?r)` predicate plus region objects would clear them. 2D shapes additionally need multi-block support |
| **G2** — not a world state | 3 | Flip N times, rotate N quarter-turns, "undo" | A different supervision signal entirely — action-sequence or trajectory-level goals rather than state goals. Not a predicate problem |
| **G3** — open / disjunctive | 1 | Pattern with choice | `exists` / `or` support in the goal parser *and* the vectorized success check. Currently `exists` is rejected and `or` is silently flattened, which is the more dangerous failure |
| **G4** — reset-time randomization | 2 | Line up by size, unstable intermediate states | Per-block size control. Today `block_size_range` samples one size per env shared by every block, so heterogeneous-size scenes cannot be built at all |

Worth noting that G1 dominates, and most of it is one missing concept: **the system has no way to
refer to a place on the table**, only to other blocks. Every spatial goal is pairwise-relative. Adding
region objects would be a larger win than any individual generator.

---

## Generalization: Interpolative vs. Extrapolative

Define the training distribution explicitly per task family so every eval condition can be
labeled **in-distribution**, **interpolative** (inside the convex hull of training values, but
unseen), or **extrapolative** (outside it). Suggested splits:

| Family | Training set | Interpolative eval | Extrapolative eval |
|---|---|---|---|
| Spelling | Words of length 3 and 5, from 20 of 26 letters | Unseen length-4 words from trained letters | Length 6–7 words; words using the 6 held-out letters |
| Digits | Sequences/orderings over digits {0–5, 8, 9} | Unseen orderings of trained digits | Sequences involving 6, 7; 4-digit numbers when training used ≤3 |
| Counting | Stack N for N ∈ {2, 3, 5} | N = 4 | N = 6, 7 (also stresses LL: taller stacks, higher grasps) |
| Patterns | 2-color alternation, 3-block mirrors | 2-color alternation with unseen color pair | 3-color cycles, longer mirrors, patterns over *letters* instead of colors |
| Repetition | Flip 2 or 3 times | — (integer gap too small) | Flip 5 times |
| Distractors | 0–4 distractors | 5 | 8–12 |
| Scene scale | 4–8 total blocks | 6 (unseen combos) | 12–15 total blocks |

Report interpolation and extrapolation success **separately per family** — averaging them hides
the most informative signal. Extrapolative counting and word length are the clearest tests of
whether the HL learned a *rule* versus memorized goal templates.

## Composition

Composition is evaluated at three levels, in increasing difficulty:

1. **Within-family recombination.** Novel combinations of trained atoms: unseen words from
   trained letters, unseen color orders from trained pattern rules, trained counts applied to
   unseen block categories. (Cheapest; covered by the held-out splits above.)
2. **Cross-family single instruction.** A goal that fuses two trained families never paired in
   training: "spell a word using only red letter blocks" (spelling × color), "stack digits in
   ascending order" (digits × ordering × stacking), "make an alternating stack of exactly 4"
   (pattern × counting).
3. **Sequential chaining** (Section 7). Multi-clause instructions where later clauses depend on
   earlier outcomes. Score per-clause to localize where the chain breaks.

Key control: verify each component is individually solved (≥ some threshold in isolation)
before attributing compositional failures to composition itself.

## Additional Evaluation Axes

Beyond distractors, generalization, and composition:

1. **Horizon-length scaling.** Report success as a function of minimum plan length (1, 2, 4, 6, 8+
   actions). TAMP-competence shows up as a slower success decay with plan length.
2. **Instruction phrasing.** Paraphrase goals ("stack them like a rainbow" vs. "red on green on
   blue") to test language robustness independent of planning.
3. **Initial-state adversity.** Same goal, initial states ranked by difficulty: blocks free on
   table → goal blocks pre-stacked wrongly → goal blocks buried under distractors.
4. **Mid-episode perturbation (recovery).** Knock over the stack or teleport a placed block back
   to the table partway through; measure whether the HL replans. This directly tests closed-loop
   reasoning vs. open-loop plan memorization.
5. **Distractor confusability.** Distractors visually similar to goal blocks (same color, glyphs
   like O/0, I/1, 6/9, M/W) vs. clearly distinct — separates grounding errors from planning
   errors. The letter/digit set makes this especially sharp: O-vs-0 and 6-vs-9 (orientation-
   dependent identity) are adversarial grounding probes.
6. **Ablation baselines.** For each task, also run: (a) LL-only with oracle HL subgoals
   (upper-bounds LL execution), (b) TAMP replay (`scripts/evaluate.py` replay mode, oracle
   upper bound), (c) random/zero action floors. This decomposes failures into
   *HL planning error* vs. *LL execution error*.

## Metrics

- **Task success rate** — all goal predicates satisfied at episode end (existing predicate
  checkers: `On`, `OnTable`, `OriOf`, `NextTo`, isolation).
- **Subgoal completion fraction** — partial credit as the fraction of goal predicates satisfied;
  essential for long-horizon tasks where binary success saturates at 0.
- **Progress-over-time** — max predicates satisfied at any point vs. at episode end; a gap
  indicates the policy *undoes* its own progress (a lookahead failure signature).
- **Action efficiency** — executed HL steps ÷ minimum plan length (from the TAMP planner).
- **Failure taxonomy** — wrong object grounding / wrong action choice / correct plan but LL
  execution failure / no-op or dithering. Attributable via the ablation baselines above.
- **Recovery rate** — success after perturbation ÷ success without perturbation.

## Suggested Priority

If compute-constrained, the highest-signal subset is:

1. **Held-out words as stacks** (clean / few / many distractors) — composition + orientation, with
   the interpolative/extrapolative word-length split.
2. **Buried target + swap** — minimal, unambiguous lookahead tests.
3. **Stack exactly N with excess blocks**, N extrapolated beyond training — stopping condition +
   rule extrapolation, cheap to generate.
4. **Ascending digit line** — numeric ordering derived from labels; unseen digit subsets give
   interpolation for free.
5. **Obstructed placement** — the cheapest pure collision-reasoning probe.
6. **Mid-episode perturbation on any of the above** — closed-loop replanning.

## Mapping to Existing Infrastructure

| Task family | Existing generator / needs new |
|---|---|
| Spelling (stack, line, distractors) | `scripts/task_gen/spelling.py` (generalize `LETTER_BLOCKS` / `WORD_ASSIGNMENTS` to the full A–Z set and arbitrary word input) |
| Digit tasks (ordering, sequences, numbers) | **New generator** (same structure as spelling, goal ordering computed from labels) |
| Stack N (colored) | `scripts/task_gen/stack_red_blocks*.py`, `any_on_any.py` |
| Flip / rotate N times | `scripts/task_gen/flip.py`, `rotate.py`, `pnp_flip_back.py` (extend for repetition counts) |
| Conditional goals | `scripts/task_gen/r_on_red.py`, `random_n_goal_pred.py` |
| Isolate / push-align | `scripts/task_gen/isolate.py`, `problems/push_align*.pddl` |
| Zebra / rainbow / mirrored patterns | **New generator** (straightforward `On`/`NextTo` chains over color classes) |
| Swap / reversal / buried target (🔵) | **New initial-state sampler** (pre-stacked scenes); goals themselves need nothing new |
| 2D shapes, region-based sorting (⛔ G1) | **New predicate + vectorized checker** — `in_region` / pose-target goals. A generator alone is not enough |
| Collision-reasoning scenes (🔵) | **New initial-state sampler** (place distractors at goal locations, in push corridors, or surrounding pick targets; goals reuse existing predicates) |
| Derived-goal tasks: alphabetical, arithmetic, extreme digit (🟡) | **New generator** only — the generator computes the answer and emits concrete literals |
| Perturbation protocol | **New eval hook** in `scripts/evaluate.py` (mid-episode state reset of one object) |

Distractor counts are already supported via `DistractorPool` in the task specs, so the
clean / few / many axis is a sweep over `--distractors` ranges in existing generators.

---

## First Datagen Run (`jul26_hvla`)

16 families x 8 PDDL problems x 32 reset states = 256 attempts each, run (Jul 2026) through
the legacy three-stage TAMP pipeline, which is not part of this release. Every one of the 95
jobs exited 0 — the shortfalls below are task attrition, not infrastructure failure.

| Family | Plans / 256 | Episodes | vs 128 |
|---|---|---|---|
| stack_exactly_n | 255 | 249 | ok |
| split_into_groups | 220 | 220 | ok |
| letters_orientation | 250 | 206 | ok |
| conditional_r_on_red | 254 | 200 | ok |
| color_zebra_stack | 199 | 189 | ok |
| make_number_line | 256 | 150 | ok |
| color_rainbow | 152 | 137 | ok |
| push_next_to | 195 | 92 | short |
| spell_line | 232 | 84 | short |
| make_number | 206 | 71 | short |
| chained_conjunctions | 121 | 59 | short |
| spell_stack | 113 | 30 | short |
| spell_stack_distract | 83 | 21 | short |
| color_alternating_line | 256 | 4 | **executor limit** |
| color_mirror_line | 222 | 3 | **executor limit** |
| isolate | 0 | 0 | **no plans** |

Three findings worth knowing before generating more data:

1. **Chained `on_right` does not survive past ~3 blocks.** Execution success falls off a
   cliff with the length of the `on_right` chain — 3-block lines (`spell_line`,
   `make_number_line`) reach 36–59%, but 4- and 5-block color lines collapse to ~1.5%,
   failing with `EFFECT_CHECK_FAILED` on `push_to_right`. Each push disturbs blocks placed
   earlier, and the goal check requires every `on_right` to hold at once. Keep line layouts
   at 3 blocks, or add an executor that places rather than pushes.
2. **`reorient` is the bottleneck for glyph blocks.** Every letter/digit family fails
   predominantly with `EFFECT_CHECK_FAILED: ori_of(<block>, oN) not satisfied after
   reorient`. Note this is *not* caused by the `block_letter` / `block_digit` types: those
   subtypes (like the legacy `block_r_e` / `block_o_b` / `block_t_g`) are declared in
   `domain.pddl` but referenced by no action precondition and nowhere in `tmp3`, so they
   are inert labels. Reorient-only tasks on the same assets (`letters_orientation`) reach
   82%, so it is the reorient-plus-stack combination that attrites.
3. **`isolate` never plans.** All 256 instances exhaust the planner's search budget at
   ~470s each with no error emitted, so the family produced nothing to execute.
