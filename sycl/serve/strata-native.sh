#!/usr/bin/env bash
# strata-native.sh: the SYCL-built engine as an in-container `exe` for serve/server.py (--engine strata).
#
# The port's binary needs the oneAPI runtime. sycl/serve/strata-sycl.sh provides it by nesting a
# `docker run` of the dev image - right on a bare host, Docker-in-Docker inside a container. This
# wrapper is the "run it natively" path of docs/INTEL_ARC.md ("Without Docker"): the container is
# already the oneAPI runtime image, so source setvars.sh and exec the binary directly, with the
# port's run-time switches. The serve protocol is lines on stdin/stdout (the server attaches them);
# stderr goes to the server's log.
#
#   STRATA_SYCL_BIN   the engine binary (default: <repo>/build-sycl-aot/strata)
#
# The server's environment (child_env in serve/server.py) is inherited: the container's STRATA_*,
# ONEAPI_*, UR_*, IGC_*, SYCL_* and ZES_* variables, overlaid by the config's "env" block. The
# defaults below are the port's own (strata-sycl.sh's); the engine reads a value of 0 (or empty) as
# "off", so a 0 inherited from the config is passed through as-is.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
# oneAPI 2026.1's setvars.sh stops at "OCL_ICD_FILENAMES: unbound variable" under `set -u`
# (docs/INTEL_ARC.md): source it with -u off, as sycl/tools/build.sh does.
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
export SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
export STRATA_VERIFY_DEVICE_PLAN=${STRATA_VERIFY_DEVICE_PLAN:-1}
export STRATA_VERIFY_NO_HOST=${STRATA_VERIFY_NO_HOST:-1}
export STRATA_STAGER_THREADS=${STRATA_STAGER_THREADS:-12}
cd "$here"
exec "${STRATA_SYCL_BIN:-$here/build-sycl-aot/strata}" "$@"
