#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="mesh-prep"

echo "=== mesh-prep setup ==="

# ---------- Platform detection ----------
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="mac" ;;
    *)       echo "Unsupported platform: $OS"; exit 1 ;;
esac
echo "Platform: $PLATFORM ($ARCH)"

if ! command -v conda &>/dev/null; then
    echo "Error: conda is required. Install Miniconda/Anaconda first."
    exit 1
fi

eval "$(conda shell.bash hook)"

# ---------- Create env if needed ----------
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '$ENV_NAME' exists, activating..."
else
    if [ "$PLATFORM" = "mac" ] && [ "$ARCH" = "arm64" ]; then
        # Apple Silicon: force x86 via Rosetta (vmtk has no arm64 build)
        echo "Creating conda env '$ENV_NAME' (Python 3.9, x86 via Rosetta)..."
        CONDA_SUBDIR=osx-64 conda create -y -n "$ENV_NAME" python=3.9 -q
        conda activate "$ENV_NAME"
        conda config --env --set subdir osx-64
        echo "Env locked to osx-64 (Rosetta)"
    else
        # Linux or Intel Mac: native x86_64
        echo "Creating conda env '$ENV_NAME' (Python 3.9)..."
        conda create -y -n "$ENV_NAME" python=3.9 -q
    fi
fi

conda activate "$ENV_NAME"

# ---------- System packages (Ubuntu only) ----------
if [ "$PLATFORM" = "linux" ]; then
    echo "Installing system packages for Qt/GL..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            libgl1-mesa-glx libglib2.0-0 libxkbcommon-x11-0 \
            libfontconfig1 libdbus-1-3 libxcb-xinerama0 2>/dev/null || true
    fi
fi

# ---------- Install vmtk via conda ----------
echo ""
echo "Installing vmtk from conda-forge..."
conda install -y -c conda-forge vmtk -q

# ---------- Install mesh-prep + GUI deps via pip ----------
echo ""
echo "Installing pyvista, pyvistaqt, and mesh-prep..."
pip install pyvista qtpy PyQt5 -q
pip install pyvistaqt --no-deps -q
pip install -e "$SCRIPT_DIR" -q

# ---------- Verify ----------
echo ""
echo "Verifying dependencies..."
deps_ok=true

for dep in numpy pyvista vtk PyQt5 pyvistaqt vmtk; do
    mod=$dep
    case $dep in
        PyQt5) mod="PyQt5.QtWidgets" ;;
    esac
    if python -c "import $mod" 2>/dev/null; then
        ver=$(python -c "import $mod; print(getattr($mod, '__version__', 'OK'))" 2>/dev/null)
        echo "  $dep ... $ver"
    else
        echo "  $dep ... MISSING"
        deps_ok=false
    fi
done

if [ "$deps_ok" = false ]; then
    echo ""
    echo "Some dependencies failed. Check errors above."
    exit 1
fi

echo ""
if command -v mesh-prep &>/dev/null; then
    echo "mesh-prep installed at: $(which mesh-prep)"
fi

echo ""
echo "=== Setup complete ==="
echo "Run:  conda activate $ENV_NAME && mesh-prep"
