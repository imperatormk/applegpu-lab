"""Collect requests and results from the board repo's issues into site/board.json.

Runs in the Pages workflow. Requests come from any issue labelled `request`;
results only from comments by the repo's owner, members or collaborators, so
nobody outside the tester list can put numbers on the board.
"""
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.board import parse_request  # noqa: E402

REPO = os.environ["GITHUB_REPOSITORY"]
TOKEN = os.environ.get("GITHUB_TOKEN")
TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}
RESULT = re.compile(r"```json lab-result\s*\n(.*?)\n```", re.S)
OUT = Path(__file__).resolve().parents[1] / "site" / "board.json"


def api(path):
    items, url = [], f"https://api.github.com/repos/{REPO}/{path}"
    while url:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   **({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})})
        with urllib.request.urlopen(req) as r:
            items += json.load(r)
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
            url = m and m.group(1)
    return items


def results(issue):
    out = []
    for c in api(f"issues/{issue['number']}/comments?per_page=100"):
        if c["author_association"] not in TRUSTED:
            continue
        for block in RESULT.findall(c["body"] or ""):
            try:
                env = json.loads(block)
            except ValueError:
                continue
            if env.get("schema") != "lab/v1":
                continue
            env["posted"] = {"by": c["user"]["login"], "at": c["created_at"], "url": c["html_url"]}
            out.append(env)
    return out


def main():
    issues = []
    for i in api("issues?state=all&per_page=100"):
        if "pull_request" in i:
            continue
        labels = [l["name"] for l in i["labels"]]
        runs = results(i)
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
