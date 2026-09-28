"""Build the AppleGPU plugin at a triton-ext commit, against the Triton and torch of the timing venv.

Each commit gets a worktree in ~/lab-builds/<sha8> of lab's own clone (or of
LAB_TRITON_EXT_SRC, an existing clone), built once and reused.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

BUILDS = Path(os.environ.get("LAB_BUILDS", Path.home() / "lab-builds"))
REMOTE = os.environ.get("LAB_TRITON_EXT_REMOTE", "https://github.com/imperatormk/triton-ext.git")
PKG = Path("backend/AppleGPU/python/triton_apple_backend")

PROBE = """
import json, os, sys, torch, triton, torch.utils
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(triton.__file__))))
info = os.path.join(root, "cmake", "llvm-info.json")
print(json.dumps({"python": sys.executable, "triton_root": root, "triton_file": triton.__file__,
                  "llvm_info": json.load(open(info)) if os.path.exists(info) else None,
                  "torch_dir": os.path.join(torch.utils.cmake_prefix_path, "Torch")}))
"""


def _run(cmd, log=None, **kw):
    if log:
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(map(str, cmd))}\n")
            f.flush()
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, **kw)
        if r.returncode:
            tail = "".join(open(log).readlines()[-25:])
            raise SystemExit(f"{cmd[0]} failed (full log: {log}):\n{tail}")
        return ""
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise SystemExit(f"{' '.join(map(str, cmd[:4]))} failed:\n{r.stderr.strip()}")
    return r.stdout.strip()


def source():
    if os.environ.get("LAB_TRITON_EXT_SRC"):
        return Path(os.environ["LAB_TRITON_EXT_SRC"]).expanduser()
    src = BUILDS / "_src"
    if not (src / ".git").exists():
        BUILDS.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {REMOTE} -> {src}")
        _run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", REMOTE, str(src)])
    return src


def resolve(src, rev):
    def lookup():
        r = subprocess.run(["git", "-C", str(src), "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    sha = lookup()
    if not sha:
        _run(["git", "-C", str(src), "fetch", "--quiet", REMOTE, "+refs/heads/*:refs/remotes/lab/*"])
        sha = lookup()
    if not sha:
        raise SystemExit(f"commit {rev} not found in {REMOTE}")
    return sha


def target_env(python):
    env = json.loads(_run([python, "-c", PROBE]).splitlines()[-1])
    llvm = os.environ.get("LLVM_INSTALL_DIR")
    if not llvm and env["llvm_info"]:
        i = env["llvm_info"]
        llvm = str(Path.home() / ".triton" / "llvm" / f"llvm-{i['llvm_hash'][:8]}-macos-arm64-{i['build_number']}")
    if not llvm or not Path(llvm).is_dir():
        raise SystemExit(f"can't find the LLVM your Triton was built with (tried {llvm}); set LLVM_INSTALL_DIR")
    env["llvm"] = llvm
    env["triton_sha"] = subprocess.run(["git", "-C", env["triton_root"], "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip() or None
    return env


def ensure(rev, python):
    """Return (checkout of triton-ext at rev, whether it was built just now)."""
    for tool in ("cmake", "ninja"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found; pip install cmake ninja")
    src = source()
    sha = resolve(src, rev)
    tree = BUILDS / sha[:8]
    stamp = tree / ".lab-build.json"
    env = target_env(python)
    want = {"sha": sha, "python": env["python"], "triton_sha": env["triton_sha"], "llvm": env["llvm"],
            "torch_dir": env["torch_dir"]}
    if stamp.exists() and json.loads(stamp.read_text()) == want and (tree / PKG / "libapplegpu_backend.dylib").exists():
        return tree, False

    if not tree.exists():
        _run(["git", "-C", str(src), "worktree", "add", "--quiet", "--detach", str(tree), sha])
    backend = tree / "backend" / "AppleGPU"
    build = backend / "build"
    shutil.rmtree(build, ignore_errors=True)
    stamp.unlink(missing_ok=True)
    log = tree / "lab-build.log"
    print(f"  building plugin at {sha[:8]} (log: {log})")
    cenv = {**os.environ, "LLVM_INSTALL_DIR": env["llvm"]}
    _run(["cmake", "-S", str(backend), "-B", str(build), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
          f"-DPython_EXECUTABLE={env['python']}", f"-DTorch_DIR={env['torch_dir']}"], log, env=cenv)
    _run(["ninja", "-C", str(build), "libapplegpu_backend.dylib"], log, env=cenv)

    pkg = tree / PKG
    targets = [build / "lib" / "libapplegpu_backend.dylib", *build.glob("**/metal_native*.so"),
               *build.glob("**/metal_torch*.so")]
    for t in targets:
        link = pkg / t.name
        link.unlink(missing_ok=True)
        link.symlink_to(os.path.relpath(t, pkg))
    if not (pkg / "libapplegpu_backend.dylib").exists():
        raise SystemExit(f"build finished but no libapplegpu_backend.dylib under {build}/lib")
    stamp.write_text(json.dumps(want, indent=2) + "\n")
    return tree, True
