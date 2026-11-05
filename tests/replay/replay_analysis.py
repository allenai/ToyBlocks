import sys
import json

thres = {"final": 1e-2, "max": 2e-1, "mean": 3e-2}

with open("test_output/replay_flip_pnp_r_on_red/replay_results.json") as f:
    results = json.load(f)

num_checks = 0
for ep, ep_results in enumerate(results):
    for k, v in ep_results.items():
        for thres_key, thres_val in thres.items():
            if k.endswith(f"_error_{thres_key}"):
                num_checks += 1
                if abs(v) > thres_val:
                    print(f"Failed {ep=} with {k} = {v:.3f} > {thres_val:.3f}")
                    sys.exit(1)

print(f"Replay analysis: {num_checks} checks across {len(results)} episodes and {'/'.join(list(thres.keys()))} per-episode errors [SUCCESS]")
