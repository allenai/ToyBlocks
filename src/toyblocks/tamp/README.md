# TAMP 

<!-- TODO: maybe remove this readme or incorporate it into some documentation for how to add a new domain, predicates etc. -->

## Code Structure

`<domain>/template`: holds the template environment files for generating new environments for data generation and evaluation for a given `<domain>`
- `<domain>/template/template_env.py`: Base environment code, necessary functions should be replaced to load the right objects in from the asset database
- `<domain>/template/template_env_cfg.py`: Base environment config code. All possible randomizations? should be included here? And let pddl env generator decide which to keep?
`<domain>/problems`: holds all problem PDDL files. Each problem PDDL file corresponds with one data generator
`<domain>/predicates`: holds all predicates for a given `<domain>`

<!-- TODO: currently organized so that we / others can add new domains. Need to evaluate how complicated that is. Also need to evaluate
whether predicates can be shared between domains? -->

## Generate environment code from PDDL

```bash
python scripts/generate_task_from_pddl.py --headless \
    --pddl src/toyblocks/tamp/blocks/problems/pnp_on_table.pddl \
    --domain_path src/toyblocks/tamp/blocks \
    --output_dir outputs \
    --num-reset-states 16

# add --visualize to open viser to view the generated instances/reset states of the tasks
# env code saved to outputs/pnp_on_table/
```

To import and make the generated environment, run the following

```python
from toyblocks.cli.isaac_app import launch_app
simulation_app = launch_app()
import gymnasium as gym
import sys
sys.path.append("outputs")
import pnp_on_table

env = gym.make(id="pnp_on_table-v0")
```