"""One timing cell, run in the target interpreter: python worker.py '<json spec>'.

Standalone on purpose (stdlib + torch + triton only), so the target venv needs
nothing from lab installed. Prints one line: LAB_RESULT <json>.
"""
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    spec = json.loads(sys.argv[1])
    if spec.get("build_python"):
        # An editable install's finder outranks sys.path and would pin the installed checkout.
        sys.meta_path[:] = [f for f in sys.meta_path
                            if "triton_apple_backend" not in getattr(f, "known_source_files", {})]
        sys.path.insert(0, spec["build_python"])

    import triton
    import triton_apple_backend
    import torch

    plugin = str(triton_apple_backend.PLUGIN_LIBRARY)
    if spec.get("build_python"):
        want = os.path.realpath(spec["build_python"])
        got = os.path.realpath(os.path.dirname(triton_apple_backend.__file__))
        if not got.startswith(want):
            sys.exit(f"lab worker: loaded triton_apple_backend from {got}, not {want}")

    case_spec = importlib.util.spec_from_file_location("lab_case", spec["case_path"])
    case = importlib.util.module_from_spec(case_spec)
    case_spec.loader.exec_module(case)

    launch, max_rel_err = case.setup(num_stages=spec["num_stages"])
    launch()
    torch.mps.synchronize()
    err = max_rel_err()
    for _ in range(spec["warmup"]):
        launch()
    torch.mps.synchronize()
    ms = []
    for _ in range(spec["samples"]):
        t = time.perf_counter()
        launch()
        torch.mps.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)

    out = {
        "samples": [round(v, 4) for v in ms],
        "median": round(statistics.median(ms), 4),
        "max_rel_err": err,
        "plugin": {"path": plugin, "sha256": sha256(plugin)},
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "triton": triton.__version__,
            "triton_sha": git_sha(os.path.dirname(triton.__file__)),
        },
    }
    print("LAB_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
