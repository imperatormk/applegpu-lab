# applegpu-lab

Timings of Triton kernels on real Apple GPUs, with results you can compare, kept out of
source-repo PR threads.

- **Requests** are issues here (the *Timing request* form or `lab request`).
- **Results** are comments on those issues. Each comment has a table and a `lab/v1` envelope in a
  fenced code block tagged `json lab-result`.
- **Run archives** (result.json, MSL / metallib / TTGIR dumps) are assets on the `runs` release.
- **The page** is built from the issues by `.github/workflows/board.yml` and deployed to Pages.
  Nothing is committed per run.

Only comments by the repo's owner, members or collaborators count as results. To post
numbers, ask to be added as a collaborator. Anyone can open a request.

## Install

On the Mac with the GPU, into the venv that has torch and your Triton:

```sh
pip install -e .
lab probe --id <short-name>        # e.g. darko-m1pro; saved to ~/.config/lab/machine.json
```

## Answer a request

```sh
lab pull 12                  # what's asked
lab run --issue 12           # builds the request's base + sha if needed, then an interleaved A/B
lab show                     # the table of the latest run
lab publish --issue 12       # zip -> release asset, table + envelope -> comment
```

(`lab run --issue 12 --publish` does both.)

Builds go to `~/lab-builds/<sha8>`. `lab` keeps its own clone of the fork there, builds against
the Triton, LLVM and torch of the venv running `lab`, and reuses a build until any of those
change. The first build of a commit takes about a minute. Needs `cmake` and `ninja`; set
`LLVM_INSTALL_DIR` if your Triton isn't a source checkout. `LAB_TRITON_EXT_SRC=<clone>` reuses a
clone you already have.

`--build` takes a commit or a path to a triton-ext checkout with its plugin built; repeat it to
time any set of builds. With neither `--issue` nor `--build`, the installed plugin is timed. You
can look at that run locally, but it can't be published because there is no SHA to pin it to.

`publish` refuses a build with uncommitted changes under `backend/AppleGPU` unless you pass
`--allow-dirty`.

## What a run measures

For every build × `num_stages`, in a fresh subprocess with its own `TRITON_CACHE_DIR`:
one launch, a correctness check against the case's reference, 5 warmup launches, then 20
launches with `torch.mps.synchronize()` around each. Rounds (default 3) alternate the build
order, forward then reversed. The worker records the path and sha256 of the plugin dylib
it actually loaded, and fails if that dylib is not from the requested checkout.

If a row's round medians differ by more than 3 %, something else was using the machine. The
row is flagged ⚠ and `publish` refuses the run unless you pass `--force`. After building a
plugin, `run` waits 60 s (`--cooldown`) before timing.

Two results are comparable only if they have the same `machine.id`, `case.id` + `case.sha256`,
`config`, and `protocol`. The Δ column only compares builds within one run, because only
those were interleaved.

## Cases

`lab/cases/<name>.py` defines a `CASE = {...}` literal (id, config, default num_stages, tolerance)
and `setup(num_stages) -> (launch, max_rel_err)`. `lab cases` lists them.
