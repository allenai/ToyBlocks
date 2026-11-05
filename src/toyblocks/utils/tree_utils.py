from typing import Any, Callable
import typing
import torch
def slice(x, i):
    out = dict()
    if isinstance(x, dict):
        for k, v in x.items():
            out[k] = slice(v, i)
    else:
        return x[i]
    return out

def cat(xs: list[dict], dim: int = 0):
    out = dict()
    def _cat(x, y):
        _out = dict()
        if isinstance(y, dict):
            for k, v in y.items():
                if k in x:
                    _out[k] = _cat(x[k], v)
                else:
                    _out[k] = v
        else:
            return torch.cat([x, y], dim=dim)
        return _out
    for x in xs:
        out = _cat(out, x)
    return out

def reshape(x, shape):
    if isinstance(x, dict):
        return {k: reshape(v, shape) for k, v in x.items()}
    else:
        return x.reshape(shape)

def shape(x, first_only=False):
    """
    Get the shape of leaf items in a tree. If first_only is True, return the shape of the first item only
    """
    if isinstance(x, dict):
        if first_only:
            return shape(next(iter(x.values())), first_only)
        return {k: shape(v, first_only) for k, v in x.items()}
    else:
        return x.shape

@typing.overload
def apply(x, f: Callable[[Any], Any]) -> Any: ...
def apply(x, y, f: Callable[[Any, Any], Any]) -> Any:
    if callable(y):
        if isinstance(x, dict):
            return {k: apply(v, f) for k, v in x.items()}
        else:
            return f(x)
    else:
        if isinstance(x, dict):
            return {k: apply(v, y[k], f) for k, v in x.items()}
        else:
            return f(x, y)