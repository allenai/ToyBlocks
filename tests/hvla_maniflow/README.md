# ManiFlow + HVLA integration tests

Pytest modules here exercise ManiFlow serving paths that depend on a checked-out **`maniflow-tb`** submodule (torchvision crop path, etc.). Depth Z masking for eval is tested against vendored **`depth_ptmap_bounds`** in ToyBlocks (no HVLA import).

They run in **Core CI** (`ci-datagen.yaml`, step “ManiFlow + HVLA integration tests”) after `uv pip install -e .[dev,isaac,torch-cu128,maniflow]` and submodule init.

They are **not** collected by **`ci-unit.yaml`** (`tests/serving` stays submodule-free).
