# Setting up a new policy server

ToyBlocks provides a base class and a WebSocket server so any model can be evaluated
with the same pipeline. The model implementation lives in its own repository;
ToyBlocks only provides the interface.

**1. Subclass `BasePolicy`** in your model's package:

```python
from blocksuite.serving import BasePolicy

class MyPolicy(BasePolicy):
    def __init__(self):
        self.model = load_my_model()

    def infer(self, obs: dict) -> dict:
        action = self.model.predict(obs)
        return {"actions": action}  # np.ndarray, shape (action_dim,) or (T, action_dim)
```

The `infer` method receives a dict of numpy arrays and must return a dict with at
least an `"actions"` key. Any extra ndarray keys returned alongside it are made
available to the eval loop on `PolicyClientAdapter.last_aux` (this is how
`action_pd_mode` is passed through).

**2. Serve it** using `blocksuite.serving.PolicyServer`:

```python
from blocksuite.serving import PolicyServer
server = PolicyServer(MyPolicy(), host="0.0.0.0", port=8000)
server.serve_forever()
```

There is no generic launcher script; wrap these lines in your own entry point.
`src/blocksuite/serving/serve_maniflow_policy.py` (run as
`python -m blocksuite.serving.serve_maniflow_policy`) is a complete example: a `tyro` CLI that
loads the ManiFlow checkpoint into a `BasePolicy` and serves it with `PolicyServer`.

Point [`evaluate.py`](../evals/evaluate.md) at the server with `--policy_host` and
`--policy_port`. `launch_eval.py` starts only the ManiFlow server (`--policy maniflow` or
`molmo2_maniflow`), so run your own server alongside `evaluate.py` instead.

**3. (Optional) Add an observation transform** if your model expects a different
observation format than the ToyBlocks env produces. Subclass `BaseObsTransform` in
`src/blocksuite/serving/obs_transforms/` and register it in `_lazy_registry`, then use
`--obs_transform <name>` at eval time. Transforms can also override
`action_postprocess` to rewrite the policy response before the action chunk broker.
