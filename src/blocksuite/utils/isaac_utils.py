from typing import Any


def replace_env_regex_ns(cfg: Any, replace: str = "/World/envs/env_.*"):
    # go through every entry possible and if there is a property prim_path with {ENV_REGEX_NS} replace it with /World/envs/env_.*
    if hasattr(cfg, "__dict__"):
        for k, v in cfg.__dict__.items():
            if hasattr(v, "__dict__") or isinstance(v, dict) or isinstance(v, list):
                replace_env_regex_ns(v, replace)
            elif k == "prim_path":
                cfg.__dict__[k] = v.format(ENV_REGEX_NS=replace)
    elif isinstance(cfg, dict):
        for k, v in cfg.items():
            if hasattr(v, "__dict__") or isinstance(v, dict) or isinstance(v, list):
                replace_env_regex_ns(v, replace)
            elif k == "prim_path":
                cfg[k] = v.format(ENV_REGEX_NS=replace)
    elif isinstance(cfg, list):
        for v in cfg:
            replace_env_regex_ns(v, replace)