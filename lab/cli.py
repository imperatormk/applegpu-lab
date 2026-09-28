import argparse
import json
import sys
import time
from pathlib import Path

from . import board, builds, machine, runner


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def _checkout(ref, python):
    p = Path(ref).expanduser()
    if p.is_dir():
        return str(p), False
    tree, built = builds.ensure(ref, python)
    return str(tree), built


def cmd_probe(a):
    print(json.dumps(machine.load(a.id), indent=2))


def cmd_cases(a):
    for c in runner.list_cases():
        meta, _ = runner.case_meta(runner.case_path(c))
        print(f"{meta['id']:<28} {meta.get('desc', '')}")


def cmd_run(a):
    cases, ns, refs = a.case, a.ns, a.build
    if a.issue:
        _, req = board.read_request(a.issue)
        cases = cases or req.get("cases", [])
        ns = ns or req.get("num_stages")
        refs = refs or [r for r in (req.get("base"), req.get("sha")) if r]
    if not cases:
        sys.exit("no case given (and none in the request)")
    resolved = [_checkout(r, a.python) for r in refs]
    checkouts = [c for c, _ in resolved]
    if any(built for _, built in resolved) and a.cooldown:
        print(f"  cooling down {a.cooldown}s after the build")
        time.sleep(a.cooldown)
    for case in cases:
        print(f"{case}: {len(checkouts) or 1} build(s), {a.rounds} rounds")
        run_dir, env = runner.run(case, checkouts, ns, a.rounds, a.warmup, a.samples, a.python, a.issue)
        print(f"\n{runner.table(env)}\n\n{run_dir}")
        noisy = runner.noisy(env)
        if noisy:
            print(f"\nnoisy: round medians differ by more than {runner.NOISE:.0%} in {', '.join(noisy)}. "
                  f"Something else was using the machine; rerun, or add --rounds 5.")
        if a.publish:
            print(board.publish(run_dir, env, a.issue, allow_dirty=a.allow_dirty, force=a.force))


def cmd_build(a):
    for rev in a.rev:
        print(builds.ensure(rev, a.python)[0])


def cmd_show(a):
    _, env = runner.load(a.run or runner.latest())
    print(runner.table(env))


def cmd_publish(a):
    run_dir, env = runner.load(a.run or runner.latest())
    issue = a.issue or (env.get("request") or {}).get("issue")
    if not issue:
        sys.exit("which issue? pass --issue N")
    print(board.publish(run_dir, env, issue, allow_dirty=a.allow_dirty, force=a.force))


def cmd_request(a):
    req = {"pr": a.pr, "sha": a.sha, "base": a.base, "cases": a.cases.split(","),
           "num_stages": _ints(a.ns) if a.ns else None, "note": a.note}
    print(board.create_request(a.title, req))


def cmd_pull(a):
    data, req = board.read_request(a.issue)
    print(f"#{a.issue} {data['title']} ({data['state'].lower()})\n{data['url']}\n")
    for k, v in req.items():
        print(f"  {k:<11} {', '.join(map(str, v)) if isinstance(v, list) else v}")
    print(f"\nto answer it (builds {' and '.join(r for r in (req.get('base'), req.get('sha')) if r)} if needed):\n")
    print(f"  lab run --issue {a.issue} --publish")


def main(argv=None):
    p = argparse.ArgumentParser(prog="lab", description="Time Triton kernels on Apple GPUs; publish to the board.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("probe", help="show this machine's record; --id names it")
    s.add_argument("--id")
    s.set_defaults(fn=cmd_probe)

    sub.add_parser("cases", help="list cases").set_defaults(fn=cmd_cases)

    s = sub.add_parser("run", help="time cases, interleaving builds")
    s.add_argument("case", nargs="*")
    s.add_argument("--build", action="append", default=[], metavar="CHECKOUT_OR_SHA",
                   help="triton-ext checkout or commit to time (commits are built on demand); repeat to A/B "
                        "(default: the request's base and sha with --issue, else the installed plugin)")
    s.add_argument("--ns", type=_ints, help="num_stages list, e.g. 1,2 (default: the case's)")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--warmup", type=int, default=5)
    s.add_argument("--samples", type=int, default=20)
    s.add_argument("--python", default=sys.executable, help="interpreter with torch + triton")
    s.add_argument("--issue", type=int, help="board request to answer (cases/ns default from it)")
    s.add_argument("--cooldown", type=int, default=60, help="seconds to wait after building before timing")
    s.add_argument("--publish", action="store_true")
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--force", action="store_true", help="publish even if noisy")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("build", help="build the plugin at triton-ext commits into ~/lab-builds")
    s.add_argument("rev", nargs="+")
    s.add_argument("--python", default=sys.executable, help="interpreter with torch + triton")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("show", help="print a run's table (default: latest)")
    s.add_argument("run", nargs="?")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("publish", help="upload a run and comment it on a board issue")
    s.add_argument("run", nargs="?")
    s.add_argument("--issue", type=int)
    s.add_argument("--allow-dirty", action="store_true")
    s.add_argument("--force", action="store_true", help="publish even if noisy")
    s.set_defaults(fn=cmd_publish)

    s = sub.add_parser("request", help="open a timing request on the board")
    s.add_argument("title")
    s.add_argument("--cases", required=True)
    s.add_argument("--sha", required=True)
    s.add_argument("--base")
    s.add_argument("--pr")
    s.add_argument("--ns")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_request)

    s = sub.add_parser("pull", help="show a request and the command that answers it")
    s.add_argument("issue", type=int)
    s.set_defaults(fn=cmd_pull)

    a = p.parse_args(argv)
    a.fn(a)
