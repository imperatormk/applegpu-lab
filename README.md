# applegpu-lab

Timings of Triton kernels on real Apple GPUs, with results you can compare, kept out of
source-repo PR threads.

- **Requests** are issues here (the *Timing request* form or `lab request`).
- **Results** are comments on those issues. Each comment has a table and a `lab/v1` envelope in a
  ```` ```json lab-result ```` block.
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
lab pull 12                                      # what's asked, and the command to run
lab run --issue 12 \
  --build ~/src/triton-ext-b9d5c06 \
  --build ~/src/triton-ext-14c74569              # interleaved A/B; cases + num_stages come from the issue
lab show                                         # the table of the latest run
lab publish --issue 12                           # zip -> release asset, table + envelope -> comment
```

Each `--build` is a triton-ext checkout whose plugin is built
(`backend/AppleGPU/python/triton_apple_backend/libapplegpu_backend.dylib` present). With no
`--build`, the installed plugin is timed. You can look at that run locally, but it can't be
published because there is no SHA to pin it to.

`publish` refuses a build with uncommitted changes under `backend/AppleGPU` unless you pass
`--allow-dirty`.

## What a run measures

For every build × `num_stages`, in a fresh subprocess with its own `TRITON_CACHE_DIR`:
one launch, a correctness check against the case's reference, 5 warmup launches, then 20
launches with `torch.mps.synchronize()` around each. Rounds (default 3) alternate the build
order, forward then reversed. The worker records the path and sha256 of the plugin dylib
it actually loaded, and fails if that dylib is not from the requested checkout.

Two results are comparable only if they have the same `machine.id`, `case.id` + `case.sha256`,
`config`, and `protocol`. The Δ column only compares builds within one run, because only
those were interleaved.

## Cases

`lab/cases/<name>.py` defines a `CASE = {...}` literal (id, config, default num_stages, tolerance)
and `setup(num_stages) -> (launch, max_rel_err)`. `lab cases` lists them.
