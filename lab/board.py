import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .runner import table

BOARD = os.environ.get("LAB_BOARD", "imperatormk/applegpu-lab")
RELEASE = "runs"
COMMENT_LIMIT = 60000

# Headings match .github/ISSUE_TEMPLATE/request.yml, so CLI and web-form requests parse the same.
FIELDS = {"Source PR": "pr", "Build SHA": "sha", "Base SHA": "base", "Cases": "cases",
          "num_stages": "num_stages", "Note": "note"}


def gh(*args, input=None, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, input=input)
    if check and r.returncode:
        raise SystemExit(f"gh {' '.join(args[:3])} failed: {r.stderr.strip()}")
    return r


def parse_request(body):
    req = {}
    for m in re.finditer(r"^###\s+(.+?)\s*\n+(.*?)(?=^###\s|\Z)", body or "", re.S | re.M):
        key, val = FIELDS.get(m.group(1)), m.group(2).strip()
        if key and val and val != "_No response_":
            req[key] = val
    for key in ("cases", "num_stages"):
        if key in req:
            req[key] = [v.strip() for v in re.split(r"[,\s]+", req[key]) if v.strip()]
    if "num_stages" in req:
        req["num_stages"] = [int(v) for v in req["num_stages"]]
    return req


def request_body(req):
    parts = []
    for title, key in FIELDS.items():
        val = req.get(key)
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        parts.append(f"### {title}\n\n{val or '_No response_'}")
    return "\n\n".join(parts) + "\n"


def create_request(title, req, board=BOARD):
    gh("label", "create", "request", "-R", board, "--color", "1d76db", "--force")
    r = gh("issue", "create", "-R", board, "--title", title, "--label", "request", "--body-file", "-",
           input=request_body(req))
    return r.stdout.strip()


def read_request(issue, board=BOARD):
    data = json.loads(gh("issue", "view", str(issue), "-R", board, "--json", "title,body,url,state").stdout)
    return data, parse_request(data["body"])


def _refusals(env, allow_dirty):
    why = []
    for r in env["results"]:
        g = r["build"]["git"]
        if not g or not g.get("sha"):
            why.append(f"build {r['build']['label']} has no git SHA (run with --build <triton-ext checkout>)")
        elif g["dirty"] and not allow_dirty:
            why.append(f"build {r['build']['label']} has uncommitted changes in backend/AppleGPU")
    return sorted(set(why))


def publish(run_dir, env, issue, board=BOARD, allow_dirty=False):
    why = _refusals(env, allow_dirty)
    if why:
        raise SystemExit("refusing to publish:\n  " + "\n  ".join(why))

    with tempfile.TemporaryDirectory() as tmp:
        archive = shutil.make_archive(str(Path(tmp) / env["id"]), "zip", run_dir)
        if gh("release", "view", RELEASE, "-R", board, check=False).returncode:
            gh("release", "create", RELEASE, "-R", board, "--title", "Run archives",
               "--notes", "Zips of lab runs (result.json, MSL/metallib dumps). One asset per run.")
        gh("release", "upload", RELEASE, archive, "-R", board, "--clobber")
    url = f"https://github.com/{board}/releases/download/{RELEASE}/{env['id']}.zip"

    env = env | {"archive": url}
    blob = json.dumps(env, separators=(",", ":"))
    if len(blob) > COMMENT_LIMIT:
        env = env | {"results": [r | {"ms": {k: v for k, v in r["ms"].items() if k != "samples"}}
                                 for r in env["results"]]}
        blob = json.dumps(env, separators=(",", ":"))
    body = (f"{table(env)}\n\n[run archive]({url}) · protocol: {env['protocol']['rounds']} interleaved rounds × "
            f"{env['protocol']['samples']} samples, warmup {env['protocol']['warmup']}\n\n"
            f"<details><summary>envelope</summary>\n\n```json lab-result\n{blob}\n```\n</details>\n")
    return gh("issue", "comment", str(issue), "-R", board, "--body-file", "-", input=body).stdout.strip()
