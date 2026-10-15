#!/bin/sh
# docker-entrypoint-sycl.sh: entry point for the Strata Intel Arc (SYCL port) container.
#
# The oneAPI runtime and the AOT engine are already in the image, so the first start only downloads
# the model and runs setup; later starts go straight to serving. The install config is kept on the
# /work/data volume so a recreated container skips the setup pass.
#
# The one difference from docker-entrypoint.sh (CUDA): setup_intel.py names sycl/serve/strata-sycl.sh
# as the config's "exe", and that script nests a `docker run`. Inside this container that would be
# Docker-in-Docker, so the entry point rewrites "exe" to sycl/serve/strata-native.sh (the engine
# binary run directly - the container IS the oneAPI runtime image) and starts sycl/serve/server_intel.py
# (serve/server.py plus the Intel Monitor readings and logprobs), as PID 1.
set -e
cd /work/strata || exit 1

# --- env vars -> setup choices (the CUDA set minus the NVIDIA-only ones, plus the Intel ones).
# VISION and LOW_RAM are deliberately not exposed: sycl/setup_intel.py forces --vision none (images are
# not wired on Intel) and --low-ram off (the port streams experts into VRAM, so the CUDA RAM rule does
# not apply).
STRATA_DATA="${STRATA_DATA:-/work/data}"
FAMILY="${FAMILY:-qwen}"
MODEL="${MODEL:-IQ2_XS}"
CONTEXT="${CONTEXT:-32768}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
API_KEY="${API_KEY:-}"
KV="${KV:-}"                    # int8 | q4_0 | k8v4; empty: setup's own default (int8)
GGUF_DIR="${GGUF_DIR:-}"        # a mounted folder with GGUF files you already have: no download
MODELS_DIR="${MODELS_DIR:-$STRATA_DATA/models}"
GPU="${GPU:-}"                  # one card, numbered as setup lists them
GPUS="${GPUS:-}"                # "0,1": one model across several cards (two Arcs, --layer-split)
LAYER_SPLIT="${LAYER_SPLIT:-}"  # with GPUS: where each later card's layers start
KV_STREAMING="${KV_STREAMING:-}" # auto | on | off; empty: setup's own default (auto)
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-}"   # empty: setup's pick (1024/2048 by context, 300 under 12 GB)
DRAFT_VOCAB="${DRAFT_VOCAB:-}"  # empty: setup's pick (en on a card under 12 GB)
CONFIG="${CONFIG:-}"            # a config file to start with (wins over MODEL's volume config)

# fail fast: the engine is part of the image; a missing one means the image was built wrong.
[ -x build-sycl-aot/strata ] || {
  echo "engine build-sycl-aot/strata is missing - rebuild the image (Dockerfile.b70)." >&2; exit 1; }

mkdir -p "$STRATA_DATA/config" "$STRATA_DATA/models"

# Same tag rule as docker-entrypoint.sh: qwen has an empty family tag, the rest take "<family>-".
# setup.py writes the config to the repo root as strata-<tag>.json; the volume copy is the one that
# counts, so the repo file is a link into $STRATA_DATA/config (the CUDA entry point's #1244 rule).
case "$FAMILY" in qwen) prefix="" ;; *) prefix="${FAMILY}-" ;; esac
tag="$(printf '%s' "${prefix}${MODEL}" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
link="/work/strata/strata-$tag.json"

if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || { echo "CONFIG=$CONFIG does not exist." >&2; exit 1; }
  ln -sfn "$CONFIG" "$link"
  cfg="$CONFIG"
  echo "Config: $cfg (from CONFIG)"
elif [ "${REINSTALL:-0}" != "1" ] && [ -f "$cfg" ]; then
  ln -sfn "$cfg" "$link"
  echo "Config: $cfg (existing)"
else
  # The engine is already in the image; this pass downloads the model (~70 GB the first time) and
  # writes the config. --backend sycl routes setup.py onto sycl/setup_intel.py (the Intel steps),
  # which re-runs this setup through its AMD path with the SYCL engine and container paths.
  if [ -n "$GGUF_DIR" ]; then
    echo "Setting up $tag from the GGUF files in $GGUF_DIR (the engine is already in the image)."
  else
    echo "Setting up $tag: downloading the model (~70 GB; the engine is already in the image)."
  fi
  set -- --family "$FAMILY" --model "$MODEL" --context "$CONTEXT" --vision none \
    --data-dir "$STRATA_DATA" --models-dir "$MODELS_DIR" \
    --host "$HOST" --port "$PORT" --api-key "$API_KEY" \
    --yes --setup --no-start
  if [ -n "$KV" ]; then set -- "$@" --kv "$KV"; fi
  if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$LAYER_SPLIT" ]; then set -- "$@" --layer-split "$LAYER_SPLIT"; fi
  if [ -n "$KV_STREAMING" ]; then set -- "$@" --kv-streaming "$KV_STREAMING"; fi
  if [ -n "$VRAM_RESERVE_MIB" ]; then set -- "$@" --vram-reserve-mib "$VRAM_RESERVE_MIB"; fi
  if [ -n "$DRAFT_VOCAB" ]; then set -- "$@" --draft-vocab "$DRAFT_VOCAB"; fi
  .venv/bin/python setup.py --backend sycl "$@"
  [ -f "$link" ] || { echo "setup did not write $link" >&2; exit 1; }
  # setup wrote the repo file (os.replace clobbers a link), so the volume copy and the link are
  # re-established here: the volume is the source of truth from this start on.
  cp -f "$link" "$cfg"
  ln -sfn "$cfg" "$link"
  echo "Config: $cfg (from MODEL $MODEL, just set up)"
fi

# Point the config at the in-container engine, not the nested-docker wrapper. Idempotent, and run on
# every start: a config carried over on the volume from a host-side `setup.sh --backend sycl` names
# strata-sycl.sh. The config's "env" block is kept (it carries ONEAPI_DEVICE_SELECTOR and, on an
# i915 Alchemist, STRATA_VERIFY_NO_HOST=0 and the FP64 emulation keys).
.venv/bin/python - "$cfg" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
c = json.loads(p.read_text(encoding="utf-8-sig"))
c["exe"] = "/work/strata/sycl/serve/strata-native.sh"
c.pop("sycl_root", None)                        # only meaningful to strata-sycl.sh
p.write_text(json.dumps(c, indent=1))
PYEOF

# The Intel server (xe Monitor readings + logprobs), as PID 1 so docker stop's SIGTERM reaches it.
# This is what run-<model>.sh does (setup_intel rewrites that script to start server_intel.py), minus
# the browser. The engine's environment: the container's STRATA_*/ONEAPI_*/UR_*/IGC_*/SYCL_*/ZES_*
# variables plus the config's "env" block (child_env in serve/server.py overlays the block on
# os.environ, so a -e on docker run reaches the engine and the config wins on a clash).
exec .venv/bin/python sycl/serve/server_intel.py --engine strata --config "$cfg" --port "$PORT"
