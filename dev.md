# Development Setup (wip)


stao: I can't get isaac lab automatically discovered by python language servers so I add this to .vscode/settings.json

```
{
    "cursorpyright.analysis.extraPaths": [
        "IsaacLab/source/isaaclab"
    ],
    "python.autoComplete.extraPaths": [
        "IsaacLab/source/isaaclab"
    ]
}
```


## Notes

`src/blocksuite/cli/base_args.py` contains a base arguments dataclass for use with tyro to replace any instance of argparse usage in Isaac. This means we have nicer and typed CLIs. See `scripts/random_actions.py` for example usage of tyro and typed CLI args.
