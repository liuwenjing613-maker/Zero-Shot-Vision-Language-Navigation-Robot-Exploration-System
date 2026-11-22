#!/usr/bin/env bash
# Install the minimal headless VLN simulation environment on Linux x86_64.
# Run with: bash scripts/install_vln_sim.sh
# Set VLN_ROOT to choose where the environment and package cache are stored.
# Set MAMBA_EXE to the Micromamba executable if it is outside PATH.
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf 'This Habitat-Sim build requires Linux x86_64.\n' >&2
    exit 1
fi

vln_root="${VLN_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/vln}"
if [[ "$vln_root" != /* ]]; then
    printf 'VLN_ROOT must be an absolute path.\n' >&2
    exit 1
fi
prefix="$vln_root/envs/vln-sim"

if [[ ! -e "$prefix" && ! -L "$prefix" ]]; then
    micromamba="${MAMBA_EXE:-micromamba}"
    if ! command -v "$micromamba" >/dev/null; then
        printf 'Install Micromamba or set MAMBA_EXE to its executable path.\n' >&2
        exit 1
    fi

    mkdir -p -- "$vln_root/envs" "$vln_root/mamba"
    "$micromamba" --no-rc --root-prefix "$vln_root/mamba" create --yes \
        --prefix "$prefix" --override-channels -c aihabitat -c conda-forge \
        --strict-channel-priority 'python=3.9' \
        'habitat-sim=0.3.3=py3.9_headless_linux_*' \
        'numpy=1.26' scipy py-opencv 'libopencv=*=headless*' \
        networkx matplotlib-base pyyaml
fi

if [[ ! -x "$prefix/bin/python" ]]; then
    printf 'No Python executable in %s. Choose a fresh VLN_ROOT or inspect the existing environment.\n' "$prefix" >&2
    exit 1
fi

# Verify dependencies without starting a GPU session or loading a dataset.
MPLBACKEND=Agg "$prefix/bin/python" -I - "$prefix" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

prefix = Path(sys.argv[1])
if sys.version_info[:2] != (3, 9) or not any(
    (prefix / "conda-meta").glob("habitat-sim-0.3.3-py3.9_headless_linux_*.json")
):
    raise SystemExit(f"Expected Python 3.9 and Habitat-Sim 0.3.3 Headless in {prefix}")

with tempfile.TemporaryDirectory(prefix="vln-mpl-") as mpl_config:
    os.environ["MPLCONFIGDIR"] = mpl_config
    import cv2
    import habitat_sim
    import magnum
    import matplotlib
    import networkx
    import numpy
    import scipy
    import yaml

    if habitat_sim.__version__ != "0.3.3" or not numpy.__version__.startswith("1.26."):
        raise SystemExit("Expected Habitat-Sim 0.3.3 and NumPy 1.26")
    print(f"Verified Python {sys.version.split()[0]}, Habitat-Sim {habitat_sim.__version__}, and simulation dependencies.")
PY

printf 'Environment: %s\n' "$prefix"
printf 'Activate with Conda or Micromamba using this path.\n'
