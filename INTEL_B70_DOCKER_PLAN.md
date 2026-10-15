# Strata on an Intel Arc Pro B70 — container build & run plan

Working note, **not verified on hardware** — it is a plan derived from `docs/INTEL_ARC.md`,
`docs/INTEL.md`, `sycl/tools/Dockerfile`, `sycl/tools/build.sh`, `sycl/serve/strata-sycl.sh`,
`sycl/setup_intel.py`, `Dockerfile` and `docker-entrypoint.sh`. Nothing here has been run on a B70.
Every number quoted is from the docs and is labelled with where it came from.

---

## 0. What the Intel path looks like today (so the delta is clear)

Three pieces, and the container is nested:

| Piece | What it is |
|---|---|
| `strata-sycl-dev` (from `sycl/tools/Dockerfile`) | oneAPI 2026.1 `icpx` + oneMKL + Level Zero, built on the community image `ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable`, plus dpct/ninja/unzip/pip/venv/rsync. `ENV ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1`, `ENTRYPOINT ["/bin/bash","-lc"]`. Docs: ~6 min build, ~13 GB image. |
| the engine build | a `docker run` of that image against the **host** checkout bind-mounted at `/work`, running `sycl/tools/build.sh strata`. Output: `build-sycl-aot/strata`. |
| `setup_intel.py` → config | writes `"exe": "<repo>/sycl/serve/strata-sycl.sh"`, and that script **nests another `docker run`** (`--device /dev/dri -v $root:/work`) to start the binary. |

Two consequences that drive the design:

1. **The root `Dockerfile` is CUDA-only** — `FROM nvidia/cuda:13.0.0-devel-ubuntu24.04`, builds with
   `-DSTRATA_ENABLE_CUDA=ON`, and its entrypoint calls `setup.py` (CUDA/HIP path). Nothing in it is reusable
   except the *shape* (engine compiled at build time, config + models on a volume, env vars read by the entrypoint).
2. **Running the stock Intel path inside a container means Docker-in-Docker** (`/var/lib/docker.sock` + the docker
   CLI in the image, and the nested `-v ...:/work` would name *host* paths). `docs/INTEL_ARC.md` ("Without Docker")
   already names the answer: run `build-sycl-aot/strata` or `sycl/serve/server_intel.py` **with an `exe` that
   sources `setvars.sh` and exports `SYCL_CACHE_PERSISTENT=0`, `ZES_ENABLE_SYSMAN=1`,
   `STRATA_VERIFY_DEVICE_PLAN=1`**. Inside our image that `exe` is just the binary — the container *is* the oneAPI
   runtime image. So: one image, no nesting.

Target layout inside the image — chosen so `setup_intel.sycl_path()`'s `/work` mapping is an **identity**
(it maps paths relative to `STRATA_SYCL_ROOT`, default `<repo>/..`, onto `/work/...`; it hard-fails on
anything outside that root):

```
/work/                 == STRATA_SYCL_ROOT (the default: repo's parent)
/work/strata/          the checkout + .venv + build-sycl-aot/strata   (from the image)
/work/data/            the volume: models/, packs/, mtp/, logs/, config/
```

---

## 1. Host prerequisites (cannot be containerised)

The kernel driver is a host thing; the image only carries the userspace runtime.

```sh
# Ubuntu 24.04+; xe (B-series) or i915 (A-series) driver + compute runtime
sudo apt install -y intel-opencl-icd libze1 libze-intel-gpu1     # docs/INTEL_ARC.md "What you need"
sudo apt install -y docker.io git
ls /dev/dri                                                     # want: renderD128
lspci -nn | grep -i 'vga\|display'                              # Pro B70 = [8086:e223]; B60 = e211/e221
```

Disk / RAM budget (docs/INTEL.md, docs/INTEL_ARC.md):

* image: ~13 GB (Stage A) + the engine build;
* model data on the volume: ~68–70 GB download for the full Flash-Next set (IQ2_XS "needs the 68 GB download";
  the Coder's shard 2 is byte-identical to IQ2_XS's, so a hard link saves the copy);
* host RAM: the B70 path keeps experts in VRAM (`--stream-experts`) and puts the misses in a **pinned host
  mirror** (`STRATA_MIRROR_MIB`, default free RAM − 4 GiB) — so don't cap the container's memory tightly.

---

## 2. Build the image — Stage A: the oneAPI dev image (upstream, unchanged)

From the Strata checkout:

```sh
cd Strata
docker build -t strata-sycl-dev sycl/tools          # once; ~6 min, ~13 GB (docs/INTEL_ARC.md)
```

Re-run this after pulling a Strata update that changes `sycl/tools/Dockerfile` (docs/INTEL.md, "How to build it").

Two probes before going further — both are cheap and both have obvious failure modes:

```sh
docker run --rm --device /dev/dri strata-sycl-dev "sycl-ls"
#   want: a level_zero: "Intel(R) Arc(TM) Pro B70 Graphics" line.
#   Only a CPU device => /dev/dri permission (host render group) or a missing xe runtime.
docker run --rm strata-sycl-dev "ocloc ids bmg-g31"
#   AOT needs ocloc. docs/INTEL.md refers to "ocloc ids bmg-g21 in the image", so it is expected present;
#   if it is not, add `intel-ocloc` to Stage A (or build Stage B's engine JIT: AOT=).
```

---

## 3. Build the image — Stage B: the Strata B70 server image

`Dockerfile.b70`, repo root (created; untracked). It does at build time exactly what `docs/INTEL_ARC.md`'s
"Build with Docker" does at run time, and stops there — setup/model download stay at first start
(`setup_intel.intel_gpus()` needs `/sys/class/drm`, and the model is ~70 GB).

```dockerfile
# syntax=docker/dockerfile:1
# Strata on an Intel Arc (B70 / B60 / B580 / A-series): the SYCL port, one container, no nesting.
# Stage A first:  docker build -t strata-sycl-dev sycl/tools
FROM strata-sycl-dev

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 \
    # the three the port's own start script pins (sycl/serve/strata-sycl.sh, docs/INTEL.md "Things that matter")
    ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1 \
    STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 STRATA_STAGER_THREADS=12

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip rsync \
    && rm -rf /var/lib/apt/lists/*

# /work is what strata-sycl.sh mounts; keeping the repo one level under it makes every
# path setup_intel writes (/work/strata/..., /work/data/...) valid inside the container as-is.
WORKDIR /work/strata
COPY . .

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint-sycl.sh sycl/serve/*.sh

# the engine, ahead-of-time for the card. -S sycl (not -S .) so the binary lands at
# build-sycl-aot/strata, which is the first path setup_intel.sycl_engine() looks for.
# Flags copied from sycl/tools/build.sh: -fp-model=precise and
# -cl-fp32-correctly-rounded-divide-sqrt are load-bearing (docs/INTEL.md, "How it was made").
ARG AOT=bmg-g31                  # Pro B70. bmg-g21: B580/B570/Pro B60. empty: SPIR-V JIT (~47 s first start)
ARG JOBS=12
RUN set +u; . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u; \
    cmake -S /work/strata/sycl -B /work/strata/build-sycl-aot -G Ninja \
          -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
          -DSTRATA_SYCL_AOT="$AOT" -DSTRATA_SYCL_PARITY=OFF \
    && cmake --build /work/strata/build-sycl-aot -j "$JOBS" --target strata -- -k 0 \
    && test -x /work/strata/build-sycl-aot/strata

VOLUME ["/work/data"]
EXPOSE 8080
# /health answers 503 while the model loads (docs/INTEL.md); a 30 GB model loads in ~2 min, and a
# first start also downloads ~70 GB. A short start-period marks a loading container unhealthy and a
# restart policy kills it mid-download (same reasoning as the CUDA Dockerfile).
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint-sycl.sh"]
```

Build commands (one image per card family — AOT is baked in):

```sh
docker build -f Dockerfile.b70 -t strata-b70 .                       # Arc Pro B70: bmg-g31 (default)
docker build -f Dockerfile.b70 --build-arg AOT=bmg-g21 -t strata-b70-b60 .   # B580 / B570 / Pro B60
docker build -f Dockerfile.b70 --build-arg AOT= -t strata-b70-jit .  # A-series / unknown card: JIT
```

Build-time facts worth knowing:

* **Needs network**: `sycl/CMakeLists.txt` `FetchContent`s llama.cpp at the pinned commit
  (`3cf03257f219afbe7334045ff7c6a06ac68c627d`) because `build.sh`/this Dockerfile don't pass
  `STRATA_GGML_DIR`, and `.dockerignore` excludes `third_party/llama.cpp/` and `/build*/`.
* **Build it on the PC it runs on**: `sycl/CMakeLists.txt` forces `GGML_NATIVE ON`, so ggml-cpu is compiled for
  the build host's CPU. Strata's own AVX-512 / AVX2 / AVX1 kernels are runtime-dispatched
  (`strata_resolve(...)` per source file), so an AVX2 host is fine, but an AVX-512-built image on an
  AVX2-only host is the same trap the CUDA image warns about.
* **No GPU needed for the build** — AOT cross-compiles for the named device via `ocloc`.
* `sycl/tools/build.sh` always exits with its last `grep`'s status, so if you use it instead of the direct
  cmake above, guard the layer: `RUN ... bash sycl/tools/build.sh strata && grep -q '^BUILD EXIT 0$' build-sycl-aot/build.log`.
* `-DSTRATA_SYCL_PARITY=OFF` skips the ~157 kernel-test targets. To re-enable them in the image, build with
  parity ON and run `ctest --test-dir build-sycl-aot` **on the card** (docs/INTEL_ARC.md).
* **Keep `build-sycl-aot/` in the image** (the CUDA image deletes its cmake trees — don't copy that here):
  `sycl_engine()` only accepts `build-sycl-aot/strata` or `build-sycl/strata`. To shrink, delete
  `_deps`/`CMakeFiles`/ggml build dirs but leave the binary, or move it and set `STRATA_SYCL_BIN`.
* Build cost: docs quote ~3 min for the engine, ~10 min for the whole first pass; `--build-arg JOBS=` to fit RAM.

---

## 4. The one new runtime file: a native `exe` (no Docker-in-Docker)

`setup_intel.to_sycl()` always writes `"exe": "<repo>/sycl/serve/strata-sycl.sh"` and has no switch for it.
That script `docker run`s the image again. Inside our container, replace it with a direct `exe`, which is the
"run it natively" path `docs/INTEL_ARC.md` describes.

`sycl/serve/strata-native.sh` (created; untracked) — the full file is the one below, plus a
`cd "$here"` before the exec (the server already sets `cwd` from the config, so it is a no-op there;
it makes the script work standalone):

```sh
#!/usr/bin/env bash
# strata-native.sh: the SYCL engine as an in-container `exe` for serve/server.py.
# The container is already the oneAPI runtime image, so no nested docker run.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
# oneAPI 2026.1's setvars.sh dies under `set -u` at OCL_ICD_FILENAMES: unbound variable (docs/INTEL_ARC.md)
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
export SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
export STRATA_VERIFY_DEVICE_PLAN=${STRATA_VERIFY_DEVICE_PLAN:-1}
export STRATA_VERIFY_NO_HOST=${STRATA_VERIFY_NO_HOST:-1}
export STRATA_STAGER_THREADS=${STRATA_STAGER_THREADS:-12}
cd "$here"
exec "${STRATA_SYCL_BIN:-$here/build-sycl-aot/strata}" "$@"
```

and in the entrypoint, after setup writes the config, rewrite the one key (`serve/server.py` reads
`cfg["exe"]`, `cfg["args"]`, and merges `cfg["env"]` into the child environment via `child_env()`):

```sh
.venv/bin/python - "$cfg" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); c = json.loads(p.read_text(encoding="utf-8-sig"))
c["exe"] = "/work/strata/sycl/serve/strata-native.sh"
c.pop("sycl_root", None)                      # only meaningful to strata-sycl.sh
p.write_text(json.dumps(c, indent=1))
PY
```

Do this on **every** start, not only after a setup pass: a config carried over on the volume from a
host-side `setup.sh --backend sycl` names `strata-sycl.sh`. Keep the config's `"env"` block — it carries
`ONEAPI_DEVICE_SELECTOR`, and on an A-series card `STRATA_VERIFY_NO_HOST=0` and the FP64 emulation keys.

*Fallback if you don't want a new `exe`:* install the docker CLI in the image and pass
`-v /var/lib/docker.sock:/var/lib/docker.sock`, and make the container's `/work` bind match a **host** path
(the nested `-v "$root:/work"` resolves on the host). Works, but it is the nesting this design removes.

---

## 5. Run it with config from environment variables — `docker-entrypoint-sycl.sh`

Created at the repo root (untracked); the full contents are that file. Same shape as
`docker-entrypoint.sh`, but it drives `setup.py --backend sycl` (which re-execs
`sycl/setup_intel.py` with `--backend` stripped) and execs `sycl/serve/server_intel.py` (the Intel server:
xe Monitor readings + logprobs) instead of `serve/server.py`. The skeleton below is what it does:

Verified details it relies on: setup writes the config to the **repo root** as `strata-<tag>.json`
(`cfg_path = ROOT / f"strata-{tag.lower()}.json"`) via `os.replace`, which clobbers a symlink — so the
entrypoint re-establishes the volume→repo symlink after every setup pass (the CUDA entrypoint's #1244
rule, done in the same run). `server.py` requires `--engine strata` (default `mock`) and spawns the
engine as `[exe, "--serve", *args]` with `cwd=cfg["cwd"]`.

Env vars to read — the CUDA set minus the NVIDIA-only ones, plus the Intel ones:

| Var | Default | Maps to |
|---|---|---|
| `FAMILY` | `qwen` | `--family` |
| `MODEL` | `IQ2_XS` | `--model` |
| `CONTEXT` | `32768` | `--context` |
| `KV` | empty → setup's `int8` | `--kv` |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | `--host` / `--port` |
| `API_KEY` | empty | `--api-key` (set it before exposing the port) |
| `GGUF_DIR` | empty | `--gguf-dir` (e.g. `/work/gguf`, no download) |
| `MODELS_DIR` | `/work/data/models` | `--models-dir` |
| `REINSTALL` | `0` | force a setup pass to change settings |
| `CONFIG` | empty | start this config instead of `MODEL`'s |
| `VRAM_RESERVE_MIB` | empty → setup's pick | `--vram-reserve-mib` |
| `DRAFT_VOCAB` | empty | `--draft-vocab` (setup forces `en` under 12 GB) |
| `GPU` / `GPUS` / `LAYER_SPLIT` | empty | `--gpu` / `--gpus` / `--layer-split` (two Arcs) |
| `KV_STREAMING` | empty → `auto` | `--kv-streaming` |
| `STRATA_*`, `ONEAPI_*`, `UR_*`, `IGC_*`, `SYCL_*`, `ZES_*` | — | pass straight to the engine (see below) |

Deliberately **not** exposed: `VISION` (images are not wired on Intel — `setup_intel` forces `--vision none`),
`LOW_RAM` / `RESIDENT_BUDGET_GIB` (`setup_intel` forces `--low-ram off` and drops
`--resident-experts`/`--mmap-experts`; the port streams experts into VRAM instead).

Skeleton:

```sh
#!/bin/sh
set -e
cd /work/strata || exit 1
STRATA_DATA="${STRATA_DATA:-/work/data}"
FAMILY="${FAMILY:-qwen}"; MODEL="${MODEL:-IQ2_XS}"; CONTEXT="${CONTEXT:-32768}"
HOST="${HOST:-0.0.0.0}"; PORT="${PORT:-8080}"; API_KEY="${API_KEY:-}"
mkdir -p "$STRATA_DATA/config" "$STRATA_DATA/models"

# same tag rule as docker-entrypoint.sh (qwen has an empty family tag); setup looks for
# strata-*.json in the repo folder, so the volume copy is linked in, not copied.
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="${prefix}$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
link="/work/strata/strata-$tag.json"

if [ -n "$CONFIG" ]; then ln -sfn "$CONFIG" "$link"
elif [ "${REINSTALL:-0}" != "1" ] && [ -f "$cfg" ]; then ln -sfn "$cfg" "$link"
else
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision none \
         --data-dir "$STRATA_DATA" --models-dir "$STRATA_DATA/models" \
         --host "$HOST" --port "$PORT" --api-key "$API_KEY" --yes --setup --no-start
  [ -n "$KV" ]                 && set -- "$@" --kv "$KV"
  [ -n "$GGUF_DIR" ]           && set -- "$@" --gguf-dir "$GGUF_DIR"
  [ -n "$GPU" ]                && set -- "$@" --gpu "$GPU"
  [ -n "$GPUS" ]               && set -- "$@" --gpus "$GPUS"
  [ -n "$LAYER_SPLIT" ]        && set -- "$@" --layer-split "$LAYER_SPLIT"
  [ -n "$KV_STREAMING" ]       && set -- "$@" --kv-streaming "$KV_STREAMING"
  [ -n "$VRAM_RESERVE_MIB" ]   && set -- "$@" --vram-reserve-mib "$VRAM_RESERVE_MIB"
  [ -n "$DRAFT_VOCAB" ]        && set -- "$@" --draft-vocab "$DRAFT_VOCAB"
  .venv/bin/python setup.py --backend sycl "$@"
  [ -e "$link" ] && cp -f "$link" "$cfg"
fi

# ... the exe rewrite from §4, idempotently, on $cfg
exec .venv/bin/python sycl/serve/server_intel.py --engine strata --config "$link" --port "$PORT"
```

Notes on that last line:

* `exec` keeps the server as PID 1, so `docker stop`'s SIGTERM reaches it (the CUDA image does the same with
  `STRATA_EXECV=1`). The port's engine exits directly once its requests are done (docs/INTEL.md, "Speed depends
  on the text"), and `strata-sycl.sh` uses `--stop-timeout 30` — mirror it on the container.
* No `--open` in a container (setup's run script would add it for a browser).
* The alternative `exec ./run-$tag.sh` also works: `setup_intel.write_run_script` already rewrites that script
  to start `server_intel.py`, and it `exec`s, so PID 1 stays right.

**How engine env vars reach the engine.** With the native `exe` there are two paths and they agree:
`serve/server.py`'s `child_env()` starts from `os.environ` and overlays the config's `"env"` block, so a plain
`-e STRATA_MIRROR_MIB=0` / `-e IGC_EnableDPEmulation=1` / `-e ONEAPI_DEVICE_SELECTOR=level_zero:*` on
`docker run` reaches the engine. The config's `"env"` block wins over `-e` for the same key. `strata-sycl.sh`
drops `0`/empty values because the engine tests switches by presence; with the native `exe` the value is passed
verbatim, and the engine itself reads `0` as "off" (docs/INTEL.md, "Why `STRATA_VERIFY_NO_HOST=0`").

---

## 6. Run command

```sh
docker run -d --name strata-b70 \
  --device /dev/dri \
  --ulimit memlock=-1 \
  --stop-timeout 60 \
  --oom-score-adj 1000 \
  -p 8080:8080 \
  -v strata-b70-data:/work/data \
  -e MODEL=IQ2_XS -e CONTEXT=32768 -e HOST=0.0.0.0 -e API_KEY=<secret> \
  strata-b70
docker logs -f strata-b70
```

Why each flag (all of these are what `sycl/serve/strata-sycl.sh` already does, moved out one level):

* `--device /dev/dri` — the GPU nodes. Not `--gpus` (that is the NVIDIA runtime hook).
  `/sys/class/drm` is read-only in a container but readable, and that is how
  `setup_intel.intel_gpus()` finds the card (vendor `0x8086` + driver `xe`/`i915`, VRAM from the PCI id table
  or BAR 2). Verify rather than assume:
  `docker run --rm strata-b70 -lc 'for f in /sys/class/drm/card*/device/vendor; do echo $f $(cat $f); done'`.
* `--ulimit memlock=-1` — the pinned host mirror (`STRATA_MIRROR_MIB`) and the pinned PLE/stager buffers.
  **Exception:** an Alchemist on `i915` (A750) should *not* get it — docs/INTEL.md's A750 section says the locked
  n-gram table plus the RAM arena then don't fit and the PC swaps.
* `--oom-score-adj 1000`, `--stop-timeout` — as in `strata-sycl.sh`.
* Don't set a low `--memory` cap: setup's KV-streaming decision uses real host RAM from `/proc/meminfo`
  (the same container-vs-host trap `docker-entrypoint.sh` documents for `LOW_RAM`), and the pinned mirror is
  charged to the cgroup. If you must cap it, set `-e KV_STREAMING=off`.
* Run as root (simplest: `/dev/dri` nodes are `root:render`). As the host user instead:
  `-u $(id -u):$(id -g) --group-add "$(getent group render | cut -d: -f3)"` — which is why
  docs/INTEL.md tells hosts to `usermod -aG docker,render`.
* Two Arcs: `-e ONEAPI_DEVICE_SELECTOR=level_zero:*` plus setup's `--gpus 0,1 --layer-split`
  (`setup_intel` also pins `STRATA_STAGE_TRIM=1`, #1980).
* A one-off command instead of the server (the Stage A image's `["/bin/bash","-lc"]` entrypoint is replaced here):
  `docker run --rm -it --device /dev/dri --entrypoint /bin/bash strata-b70 -lc 'sycl-ls'`.

---

## 7. Order to verify it in (cheap → expensive)

1. `docker run --rm --device /dev/dri strata-sycl-dev "sycl-ls"` → the B70 under `level_zero:`.
2. `docker run --rm strata-sycl-dev "ocloc ids bmg-g31"` → AOT target present.
3. `docker run --rm strata-b70 -lc 'test -x /work/strata/build-sycl-aot/strata && echo engine ok'`.
4. setup's own check, no download:
   `docker run --rm --device /dev/dri -v strata-b70-data:/work/data strata-b70 -lc '.venv/bin/python setup.py --backend sycl --check'`
   → the card named at 32 GB with a per-model fit verdict.
5. First real start with a small model (`-e MODEL=IQ1_M` for the Coder, which fits a 32 GB card outright) —
   watch for `loading the model (the first start takes a minute or two)` and then a 200 from `/health`.
6. API smoke: `curl -H "authorization: Bearer <key>" http://127.0.0.1:8080/v1/chat/completions ...`.
   Docs' served numbers on a B70: 48–72 tok/s on a ~190-token answer, prompt included; the first request after a
   start is the slower one.
7. Optional (Monitor tab VRAM/load): run `sycl/tools/gpustat.py` as root inside the container writing
   `/run/gpustat.json` (xe reports per-client VRAM only in `/proc/*/fdinfo`, root-only). Without it the tab
   still shows temperature, power and the PCIe link from sysfs.

---

## 8. Risks and things this plan cannot settle from the docs

* **Nothing here has been run on an Arc.** The maintainers state they have no Intel GPU, and the images are
  untested on Intel; the container path adds its own unknowns (sysfs visibility, `render` gid mapping,
  `ocloc` presence in the base image, `memlock` under a cgroup). Step 1–4 above exist to fail fast.
* **`exe` rewriting is a private edit of setup's output.** If a future `setup_intel.py` changes the config
  shape, the rewrite silently keeps an old `exe`. Guard: assert the config's `"backend"` is `"sycl"` and that
  its args still contain `--pack`/`--native` before exec'ing.
* **VRAM over-subscription is not a clean failure.** On `xe`, an allocation past VRAM evicts buffers to host RAM
  and the machine livelocks until the watchdog resets it (docs/INTEL.md: a 262K request did exactly that on
  2026-09-29). Measured headroom on a 32 GB B70: 32K → 3.5 GB free, 131K → 1.5 GB ("the practical ceiling"),
  163K → ~0.9 GB (not attempted), 262K → does not fit. Keep `CONTEXT` ≤ 131072 and leave ~1.5 GB.
* **AOT is per-card.** `bmg-g31` = Pro B70; `bmg-g21` = B580/B570/Pro B60; A-series has no known AOT build →
  build `AOT=` (JIT, ~47 s of compiling on the first start of each process).
* **A `-e CONTEXT=` change needs a setup pass** (`REINSTALL=1`), and the `--prefill`/`--vram-reserve-mib`
  setup_intel picks move at the 32K and 24 GB boundaries.
* **`SYCL_CACHE_PERSISTENT` stays 0** (the persistent JIT cache segfaulted on Xe2). If you build a JIT image and
  want the 15 MB cache reused across starts, docs/INTEL.md's `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` is
  recorded but flagged as the thing that crashed — opt in knowingly.
* **Images/vision are not wired on Intel** — keep `VISION` out of the env surface.
* **Update flow:** rebuild Stage A when `sycl/tools/Dockerfile` changes; rebuild Stage B after every engine
  update (setup says when the engine is older than it needs); the volume keeps models and configs, so a new
  image with an old volume is a normal start.
