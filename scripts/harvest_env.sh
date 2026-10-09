# Shell environment for the harvest stages (GPU + EGL). Source it, do not execute it:
#
#   source scripts/harvest_env.sh
#
# Run scripts/prefetch_assets.py once before the first harvest (with network access, without this
# file sourced): the settings below assume that every download is already in place.
#
# - MuJoCo and PyOpenGL render with EGL.
# - When the node has no system libEGL.so.1 (the GLVND dispatcher), the Debian/Ubuntu packages
#   libegl1, libegl-mesa0 and libglvnd0 are downloaded with apt-get and unpacked into
#   $VLAGEOM_EGL_DIR (default /tmp/libegl-$USER), which is put on LD_LIBRARY_PATH. This needs a
#   Debian or Ubuntu node with apt-get, dpkg-deb and flock; on other systems install libEGL
#   system-wide. The NVIDIA driver must provide its EGL vendor library (libEGL_nvidia.so.0).
# - JAX (imported by mujoco-mjx) stays on the CPU, so it does not reserve GPU memory.
# - Hugging Face runs offline (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1): the checkpoint and the
#   Qwen3 tokenizer must already be downloaded. Set HF_HUB_OFFLINE=0 before sourcing to allow
#   downloads.
# - VLAGEOM_NOOP_FILELOCK=1: the harvest processes only read the asset cache, so its file locks are
#   replaced by no-op locks (vlageom/sim/locks.py). Set VLAGEOM_NOOP_FILELOCK=0 before sourcing to
#   keep the real locks.

export PYTHONUNBUFFERED=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
export HF_HUB_DISABLE_TELEMETRY=1
export VLAGEOM_NOOP_FILELOCK="${VLAGEOM_NOOP_FILELOCK:-1}"

# ldconfig lives in /sbin or /usr/sbin, which are often not on a user's PATH. The lookups below
# end with `true`, so a caller running `set -e` (scripts/harvest_array.sbatch) reaches the messages.
_vlageom_ldconfig="$(command -v ldconfig || { for _c in /sbin/ldconfig /usr/sbin/ldconfig; do
    [ -x "${_c}" ] && echo "${_c}" && break; done; true; })"
if [ "${MUJOCO_GL}" = "egl" ] && ! { [ -n "${_vlageom_ldconfig}" ] \
        && "${_vlageom_ldconfig}" -p 2>/dev/null | grep 'libEGL\.so\.1 ' >/dev/null; }; then
    _egl_dir="${VLAGEOM_EGL_DIR:-/tmp/libegl-${USER:-user}}"
    mkdir -p "${_egl_dir}"
    # one unpack per node, also when several jobs start on it at once
    ( flock -w 600 9
      if [ ! -d "${_egl_dir}/root" ]; then
          cd "${_egl_dir}" \
            && apt-get download libegl1 libegl-mesa0 libglvnd0 \
            && for f in *.deb; do dpkg-deb -x "$f" ./root; done
      fi ) 9>"${_egl_dir}/.lock" \
      || echo "harvest_env.sh: could not unpack the EGL libraries into ${_egl_dir}" >&2
    _egl_lib="$(find "${_egl_dir}/root" -name 'libEGL.so.1' -print -quit 2>/dev/null || true)"
    if [ -n "${_egl_lib}" ]; then
        export LD_LIBRARY_PATH="$(dirname "${_egl_lib}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    else
        echo "harvest_env.sh: no libEGL.so.1 found; install libEGL (GLVND) system-wide" >&2
    fi
    unset _egl_dir _egl_lib
fi
unset _vlageom_ldconfig _c
