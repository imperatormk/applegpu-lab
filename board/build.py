"""Collect requests and results from the board repo's issues into site/board.json.

Runs in the Pages workflow. Requests come from any issue labelled `request`;
results only from comments by the repo's owner, members or collaborators, so
nobody outside the tester list can put numbers on the board.

A run's archive is linked only if it is this repo's `runs` release asset for
that run id, uploaded by the commenter, and its contents check out (see
check_archive). Otherwise the run is shown without a download.
"""
import hashlib
import io
import json
import os
import re
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.board import RELEASE, parse_request  # noqa: E402

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ.get("GITHUB_TOKEN")
TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}
RESULT = re.compile(r"```json lab-result\s*\n(.*?)\n```", re.S)
OUT = Path(__file__).resolve().parents[1] / "site" / "board.json"

RUN_ID = re.compile(r"[A-Za-z0-9._-]{1,200}")
MEMBER = re.compile(r"(result\.json|dumps/|dumps/[A-Za-z0-9._+-]+/"
                    r"|dumps/[A-Za-z0-9._+-]+/[A-Za-z0-9._-]+\.(msl|metal|ttgir|ttir|json|metallib))")
MAX_ZIP = 50 << 20
MAX_UNPACKED = 200 << 20
HASHED = {"msl_sha256": ".msl", "metal_sha256": ".metal", "metallib_sha256": ".metallib"}


def fetch(url, accept="application/vnd.github+json"):
    if not url.startswith("https://"):
        url = f"https://api.github.com/repos/{REPO}/{url}"
    req = urllib.request.Request(url, headers={"Accept": accept,
                                               **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})})
    return urllib.request.urlopen(req)


def api(path):
    items, url = [], path
    while url:
        with fetch(url) as r:
            items += json.load(r)
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m and m.group(1)
    return items


def release_assets():
    try:
        with fetch(f"releases/tags/{RELEASE}") as r:
            return {a["name"]: a for a in json.load(r)["assets"]}
    except urllib.error.HTTPError:
        return {}


def _comparable(env):
    keep = {k: env.get(k) for k in ("schema", "id", "case", "machine", "env", "protocol")}
    keep["results"] = [r | {"ms": {k: v for k, v in r["ms"].items() if k != "samples"}} for r in env["results"]]
    return json.dumps(keep, sort_keys=True)


def check_archive(env, author, assets):
    """Return None if the run's archive is sound, else why not."""
    run_id = env.get("id", "")
    if not RUN_ID.fullmatch(run_id):
        return "bad run id"
    name = f"{run_id}.zip"
    if env.get("archive") != f"https://github.com/{REPO}/releases/download/{RELEASE}/{name}":
        return "archive link is not this board's release asset"
    asset = assets.get(name)
    if not asset:
        return "release asset missing"
    if asset["uploader"]["login"] != author:
        return f"asset uploaded by {asset['uploader']['login']}, not {author}"
    if asset["size"] > MAX_ZIP:
        return "archive too large"

    with fetch(asset["url"], accept="application/octet-stream") as r:
        blob = r.read(MAX_ZIP + 1)
    if len(blob) > MAX_ZIP:
        return "archive too large"
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return "not a zip"

    infos = z.infolist()
    for i in infos:
        if not MEMBER.fullmatch(i.filename):
            return f"unexpected file {i.filename!r}"
        if i.flag_bits & 0x1:
            return "encrypted member"
        if stat.S_IFMT(i.external_attr >> 16) not in (0, stat.S_IFREG, stat.S_IFDIR):
            return f"{i.filename!r} is not a regular file"
    if sum(i.file_size for i in infos) > MAX_UNPACKED:
        return "archive unpacks too large"

    try:
        inner = json.loads(z.read("result.json"))
    except (KeyError, ValueError):
        return "no readable result.json"
    if _comparable(inner) != _comparable(env):
        return "result.json in the archive differs from the posted envelope"

    names = set(z.namelist())
    for res in env["results"]:
        folder = f"dumps/{res['build']['label']}-ns{res['config']['num_stages']}/"
        for key, suffix in HASHED.items():
            want = res.get("artifacts", {}).get(key)
            if not want:
                continue
            files = [n for n in names if n.startswith(folder) and n.endswith(suffix)]
            if not any(hashlib.sha256(z.read(n)).hexdigest() == want for n in files):
                return f"{folder}*{suffix} does not match {key}"
    return None


def results(issue, assets):
    out = []
    for c in api(f"issues/{issue['number']}/comments?per_page=100"):
        if c["author_association"] not in TRUSTED:
            continue
        for block in RESULT.findall(c["body"] or ""):
            try:
                env = json.loads(block)
            except ValueError:
                continue
            if env.get("schema") != "lab/v1" or not isinstance(env.get("results"), list):
                continue
            author = c["user"]["login"]
            try:
                problem = check_archive(env, author, assets)
            except (KeyError, TypeError, AttributeError, urllib.error.URLError) as e:
                problem = f"could not check archive ({type(e).__name__})"
            env["archive_check"] = problem or "ok"
            if problem:
                env["archive"] = None
                print(f"#{issue['number']} {env.get('id')}: archive rejected: {problem}")
            env["posted"] = {"by": author, "at": c["created_at"], "url": c["html_url"]}
            out.append(env)
    return out


def main():
    assets = release_assets()
    issues = []
    for i in api("issues?state=all&per_page=100"):
        if "pull_request" in i:
            continue
        labels = [l["name"] for l in i["labels"]]
        runs = results(i, assets)
        if "request" not in labels and not runs:
            continue
        issues.append({
            "number": i["number"], "title": i["title"], "state": i["state"], "url": i["html_url"],
            "author": i["user"]["login"], "updated": i["updated_at"], "labels": labels,
            "request": parse_request(i["body"]) if "request" in labels else None,
            "runs": runs,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"repo": REPO, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "issues": issues}, indent=1))
    print(f"{len(issues)} issues, {sum(len(i['runs']) for i in issues)} runs -> {OUT}")


if __name__ == "__main__":
    main()
