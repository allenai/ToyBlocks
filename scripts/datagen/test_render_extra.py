"""    python scripts/datagen/test_render_extra.py

--render-extra must carry a BARE flag, not just flag+value.

This passthrough is the only way the render sidecars (--extra_rgb / --extra_depth) and any
other render.py flag reach the render from datagen. It has been wrong twice:

  1. The help once advertised an UNQUOTED form (--render-extra --extra_depth rg8_224) that
     argparse rejects outright.
  2. The quoted form was then appended to the render argv as ONE argv element, which
     render.py rejected. Fixed by shlex-splitting.

Neither fix covered the remaining case: argparse classifies a token starting with '-' as an
OPTION rather than a value unless it contains a space (argparse._parse_optional returns None
when ' ' in arg_string). So --render-extra "--extra_rgb 224" happens to work because of the
space, while --render-extra "--skip_failure_episodes" is rejected. Only the '=' form is
correct for every case, which is why the documented form uses it.
"""
import argparse
import re
import shlex
import sys
from pathlib import Path

_failures = []


def _check(name, got, want):
    if got == want:
        print(f"  PASS  {name}")
        return
    print(f"  FAIL  {name}\n        got : {got}\n        want: {want}")
    _failures.append(name)


def _parser_kwargs():
    """The --render-extra declaration AS WRITTEN in run_datagen_pipeline.py.

    Re-typing nargs="*" here would test argparse, not the script; parse the real call so a change
    to the declaration is picked up.
    """
    src = (Path(__file__).resolve().parent / "run_datagen_pipeline.py").read_text()
    m = re.search(r'add_argument\(\s*"--render-extra",\s*(.*?)\n\s*\)', src, re.S)
    if not m:
        raise SystemExit("FAIL: could not find the --render-extra declaration")
    body = m.group(1)
    kw = {}
    if 'nargs="*"' in body:
        kw["nargs"] = "*"
    kw["default"] = [] if "nargs" in kw else ""
    return kw


def _run(kw, argv):
    p = argparse.ArgumentParser()
    p.add_argument("--render-extra", **kw)
    try:
        ns = p.parse_args(argv)
    except SystemExit:
        return None
    v = ns.render_extra
    return [t for c in (v if isinstance(v, list) else [v]) if c for t in shlex.split(c)]


def main():
    kw = _parser_kwargs()
    print(f"A: the '=' form carries every shape (declaration: {kw})")
    for argv, want, label in (
        (["--render-extra=--extra_rgb 224"], ["--extra_rgb", "224"], "flag + value"),
        (["--render-extra=--skip_failure_episodes"], ["--skip_failure_episodes"], "BARE flag"),
        (["--render-extra=--extra_rgb 224 --extra_depth rg8_224 --depth_gop 1000"],
         ["--extra_rgb", "224", "--extra_depth", "rg8_224", "--depth_gop", "1000"], "multi"),
    ):
        _check(f"= form carries a {label}", _run(kw, argv), want)

    print("\nB: the space-separated form silently fails on a bare flag")
    _check("space form, flag+value  -> works (the space saves it)",
           _run(kw, ["--render-extra", "--extra_rgb 224"]), ["--extra_rgb", "224"])
    _check("space form, BARE flag   -> rejected (this is why '=' is documented)",
           _run(kw, ["--render-extra", "--skip_failure_episodes"]), None)

    print("\nC: the docs teach the form that always works")
    src = (Path(__file__).resolve().parent / "run_datagen_pipeline.py").read_text()
    _check("docstring sample uses --render-extra=", "--render-extra=\"" in src, True)
    _check("help text tells the reader '=' is required for a bare flag",
           bool(re.search(r"= FORM|'=' is not optional", src)), True)
    # The space form may still appear as an explicit counter-example ("... is REJECTED"), but the
    # USAGE block must not teach it -- that block is what people copy.
    usage = src.split('"""', 2)[1]
    usage = usage[usage.index("Usage:"):] if "Usage:" in usage else usage
    _check("the Usage: block never shows the space form",
           bool(re.search(r'--render-extra "--', usage)), False)
    stray = [ln.strip() for ln in src.splitlines()
             if re.search(r'--render-extra "--', ln)
             and not re.search(r"REJECTED|FAILS|rejected|ACCIDENT", ln)]
    _check("every remaining space-form mention is labelled as failing", stray, [])

    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S): {_failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
