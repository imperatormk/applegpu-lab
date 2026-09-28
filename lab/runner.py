import ast
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import machine

SCHEMA = "lab/v1"
CASES = Path(__file__).parent / "cases"
WORKER = Path(__file__).parent / "worker.py"
RUNS = Path(os.environ.get("LAB_RUNS", Path.home() / "lab-runs"))
DUMP_EXTS = (".msl", ".metal", ".metallib", ".ttgir", ".ttir", ".json")
NOISE = 0.03


def case_path(case_id):
    for p in sorted(CASES.glob("*.py")):
        if p.stem == case_id or case_meta(p)[0]["id"] == case_id:
            return p
    if Path(case_id).is_file():
        return Path(case_id)
    raise SystemExit(f"unknown case {case_id!r}; have: {', '.join(list_cases())}")


def list_cases():
    return [p.stem for p in sorted(CASES.glob("*.py")) if not p.stem.startswith("_")]


def case_meta(path):
    src = path.read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CASE" for t in node.targets):
            return ast.literal_eval(node.value), hashlib.sha256(src.encode()).hexdigest()
    raise SystemExit(f"{path}: no CASE = {{...}} literal")


def _git(path, *args):
    r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def describe_build(path):
    if path is None:
        return {"label": "installed", "path": None, "python": None, "git": None}
    path = Path(path).expanduser().resolve()
    py = path / "backend" / "AppleGPU" / "python"
    if not (py / "triton_apple_backend").is_dir():
        raise SystemExit(f"{path} is not a triton-ext checkout (no backend/AppleGPU/python)")
    sha = _git(path, "rev-parse", "HEAD")
    dirty = bool(_git(path, "status", "--porcelain", "--untracked-files=no", "--", "backend/AppleGPU"))
    remote = _git(path, "remote", "get-url", "origin") or ""
    repo = remote.removesuffix(".git").split("github.com")[-1].lstrip(":/") or None
    return {
        "label": (sha or "nogit")[:8] + ("+dirty" if dirty else ""),
        "path": str(path),
        "python": str(py),
        "git": {"repo": repo, "sha": sha, "dirty": dirty},
    }


def _cell(build, ns, run_dir, case_file, warmup, samples, python):
    cache = run_dir / "cache" / f"{build['label']}-ns{ns}"
    env = {k: v for k, v in os.environ.items() if k not in ("TRITON_PLUGIN_PATHS", "TRITON_PASS_PLUGIN_PATH")}
    env.update(TRITON_CACHE_DIR=str(cache), TRITON_APPLE_NO_AUTOREGISTER="1")
    spec = {"case_path": str(case_file), "num_stages": ns, "warmup": warmup, "samples": samples,
            "build_python": build["python"]}
    r = subprocess.run([python, str(WORKER), json.dumps(spec)], capture_output=True, text=True, env=env)
    for line in r.stdout.splitlines():
        if line.startswith("LAB_RESULT "):
            return json.loads(line[len("LAB_RESULT "):]), cache
    tail = "\n".join(l for l in (r.stderr + r.stdout).splitlines() if "leaked" not in l)[-3000:]
    raise SystemExit(f"worker failed for {build['label']} ns={ns}:\n{tail}")


def _collect_dumps(cache, dest):
    hashes = {}
    for f in cache.rglob("*"):
        if f.suffix in DUMP_EXTS and not f.name.startswith("__grp__"):
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
            if f.suffix in (".msl", ".metal", ".metallib"):
                hashes[f.suffix.lstrip(".") + "_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes


def run(case_id, builds, ns_list, rounds, warmup, samples, python, issue=None, log=print):
    case_file = case_path(case_id)
    case, case_sha = case_meta(case_file)
    ns_list = ns_list or case.get("num_stages", [1])
    builds = [describe_build(b) for b in builds] or [describe_build(None)]
    info = machine.load()
    stamp = datetime.now(timezone.utc)
    run_id = f"{stamp:%Y%m%dT%H%M%S}-{info['id']}-{case['id']}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True)

    cells = {(b["label"], ns): {"rounds": [], "cache": None} for b in builds for ns in ns_list}
    order = [(b, ns) for b in builds for ns in ns_list]
    for r in range(rounds):
        for b, ns in (order if r % 2 == 0 else order[::-1]):
            try:
                res, cache = _cell(b, ns, run_dir, case_file, warmup, samples, python)
            except SystemExit:
                shutil.rmtree(run_dir, ignore_errors=True)
                raise
            c = cells[(b["label"], ns)]
            c["rounds"].append(res)
            c["cache"] = cache
            log(f"  round {r + 1}/{rounds}  {b['label']:>16}  ns={ns}  median {res['median']:.2f} ms"
                f"  err {res['max_rel_err']:.1e}")

    results, env = [], None
    for b in builds:
        for ns in ns_list:
            c = cells[(b["label"], ns)]
            first = c["rounds"][0]
            env = env or first["env"]
            flat = [s for rr in c["rounds"] for s in rr["samples"]]
            err = max(rr["max_rel_err"] for rr in c["rounds"])
            artifacts = _collect_dumps(c["cache"], run_dir / "dumps" / f"{b['label']}-ns{ns}")
            results.append({
                "build": {k: v for k, v in b.items() if k != "python"} | {"plugin": first["plugin"]},
                "config": case.get("config", {}) | {"num_stages": ns},
                "ms": {
                    "median": round(statistics.median(rr["median"] for rr in c["rounds"]), 3),
                    "min": min(flat), "max": max(flat),
                    "rounds": [rr["median"] for rr in c["rounds"]],
                    "spread": round(spread({"rounds": [rr["median"] for rr in c["rounds"]]}), 4),
                    "samples": [rr["samples"] for rr in c["rounds"]],
                },
                "correctness": {"max_rel_err": err, "tol": case.get("tol"), "ok": err <= case.get("tol", 1e-3)},
                "artifacts": artifacts,
            })

    envelope = {
        "schema": SCHEMA,
        "id": run_id,
        "created": stamp.isoformat(timespec="seconds"),
        "case": {"id": case["id"], "desc": case.get("desc"), "sha256": case_sha},
        "machine": info,
        "env": env,
        "protocol": {"warmup": warmup, "samples": samples, "rounds": rounds, "interleaved": True,
                     "order": "alternating per round", "sync": "torch.mps.synchronize per launch",
                     "timer": "time.perf_counter", "cache": "fresh TRITON_CACHE_DIR per build x num_stages",
                     "noise_limit": NOISE},
        "request": {"issue": issue} if issue else None,
        "results": results,
    }
    (run_dir / "result.json").write_text(json.dumps(envelope, indent=2) + "\n")
    shutil.rmtree(run_dir / "cache", ignore_errors=True)
    return run_dir, envelope


REQUIRED = ("schema", "id", "case", "machine", "env", "protocol", "results")


def load(run):
    p = Path(run)
    if not p.exists():
        p = RUNS / run
    if p.is_dir():
        p = p / "result.json"
    env = json.loads(p.read_text())
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing or env["schema"] != SCHEMA:
        raise SystemExit(f"{p}: not a {SCHEMA} envelope (missing {missing})")
    return p.parent, env


def latest():
    runs = sorted(d for d in RUNS.glob("*") if (d / "result.json").exists()) if RUNS.exists() else []
    if not runs:
        raise SystemExit(f"no runs in {RUNS}")
    return runs[-1]


def spread(ms):
    return max(ms["rounds"]) / min(ms["rounds"]) - 1


def noisy(env):
    limit = env["protocol"].get("noise_limit", NOISE)
    return [f"{r['build']['label']} ns={r['config']['num_stages']}" for r in env["results"]
            if spread(r["ms"]) > limit]


def table(env):
    rows = env["results"]
    first = rows[0]["build"]["label"]
    base = {r["config"]["num_stages"]: r["ms"]["median"] for r in rows if r["build"]["label"] == first}
    m = env["machine"]
    out = [f"**{env['case']['id']}** on `{m['id']}` ({m['chip']}, {m['gpu_cores']}-core GPU, macOS {m['macos']})",
           "",
           "| build | num_stages | median ms | rounds | min–max | Δ vs first | max rel err |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        ns, ms = r["config"]["num_stages"], r["ms"]
        b = base.get(ns)
        delta = "—" if r["build"]["label"] == first or not b else f"{(ms['median'] / b - 1) * 100:+.1f} %"
        ok = "" if r["correctness"]["ok"] else " ✗"
        noise = " ⚠" if spread(ms) > env["protocol"].get("noise_limit", NOISE) else ""
        out.append(f"| `{r['build']['label']}` | {ns} | {ms['median']:.2f} | "
                   f"{', '.join(f'{x:.2f}' for x in ms['rounds'])}{noise} | {ms['min']:.2f}–{ms['max']:.2f} | {delta} | "
                   f"{r['correctness']['max_rel_err']:.1e}{ok} |")
    return "\n".join(out)
