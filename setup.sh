#!/usr/bin/env bash
# =============================================================================
# One-shot environment setup for the MDO quadrotor project.
# Run from the repo root:  bash setup.sh
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mdo_quadrotor"
CCBLADE_PATH="$HOME/CCBlade.jl"
CCBLADE_OPENMDAO_PATH="$HOME/CCBladeOpenMDAOExamples"

echo "============================================================"
echo " MDO Quadrotor — environment setup"
echo " Repo:  $REPO_DIR"
echo "============================================================"

# ── Step 1: juliaup + Julia 1.9 ──────────────────────────────────────────────
echo ""
echo "[1/6] Installing Julia 1.9 via juliaup..."

if ! command -v juliaup &>/dev/null && [ ! -f "$HOME/.juliaup/bin/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes
fi
export PATH="$HOME/.juliaup/bin:$PATH"

if ! juliaup status 2>/dev/null | grep -q "1\.9"; then
    juliaup add 1.9
fi
juliaup default 1.9
echo "  Julia 1.9 ready."

# ── Step 2: conda environment ─────────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "  Environment '$ENV_NAME' already exists — skipping creation."
    echo "  To recreate it from scratch, run:  conda env remove -n $ENV_NAME"
else
    conda env create -f "$REPO_DIR/environment.yml"
    echo "  Environment created."
fi

# ── Step 3: Clone CCBlade.jl (reversal branch) and apply patches ─────────────
echo ""
echo "[3/6] Setting up CCBlade.jl..."

if [ ! -d "$CCBLADE_PATH" ]; then
    echo "  Cloning kanekosh/CCBlade.jl (reversal branch)..."
    git clone --branch reversal https://github.com/kanekosh/CCBlade.jl.git "$CCBLADE_PATH"
else
    echo "  $CCBLADE_PATH already exists — skipping clone."
fi

# Apply compatibility patches for newer Julia (fixes UndefRefError/segfault in broadcast)
python3 - <<'PYEOF'
import os

path = os.path.expanduser("~/CCBlade.jl/src/CCBlade.jl")
with open(path) as f:
    src = f.read()

already_patched = "sym === :r || sym === :chord" in src
if already_patched:
    print("  CCBlade.jl patch already applied.")
else:
    # Each patch: (sentinel to detect old code, old string, new string)
    patches = [
        # --- Section ---
        (
            "function Base.getproperty(obj::Vector{Section{TF1, TF2, TF3, TAF}}, sym::Symbol) where {TF1, TF2, TF3, TAF}\n"
            "    return getfield.(obj, sym)\n"
            "end # This is not always type stable b/c we don't know if the return type will be float or af function.",

            "function Base.getproperty(obj::Vector{Section{TF1, TF2, TF3, TAF}}, sym::Symbol) where {TF1, TF2, TF3, TAF}\n"
            "    if sym === :r || sym === :chord || sym === :theta || sym === :af\n"
            "        return getfield.(obj, sym)\n"
            "    end\n"
            "    return getfield(obj, sym)\n"
            "end # This is not always type stable b/c we don't know if the return type will be float or af function.\n"
            "\n"
            "function Base.dataids(obj::Vector{Section{TF1, TF2, TF3, TAF}}) where {TF1, TF2, TF3, TAF}\n"
            "    return (UInt(pointer(obj)),)\n"
            "end"
        ),
        # --- OperatingPoint ---
        (
            "function Base.getproperty(obj::Vector{OperatingPoint{TF1, TF2, TF3, TF4, TF5}}, sym::Symbol) where {TF1, TF2, TF3, TF4, TF5}\n"
            "    return getfield.(obj, sym)\n"
            "end",

            "function Base.getproperty(obj::Vector{OperatingPoint{TF1, TF2, TF3, TF4, TF5}}, sym::Symbol) where {TF1, TF2, TF3, TF4, TF5}\n"
            "    if sym === :Vx || sym === :Vy || sym === :rho || sym === :pitch || sym === :mu || sym === :asound\n"
            "        return getfield.(obj, sym)\n"
            "    end\n"
            "    return getfield(obj, sym)\n"
            "end"
        ),
        # --- Outputs ---
        (
            "function Base.getproperty(obj::Vector{Outputs{TF}}, sym::Symbol) where TF\n"
            "    return getfield.(obj, sym)\n"
            "end",

            "function Base.getproperty(obj::Vector{Outputs{TF}}, sym::Symbol) where TF\n"
            "    if sym === :Np || sym === :Tp || sym === :a || sym === :ap || sym === :u || sym === :v ||\n"
            "       sym === :phi || sym === :alpha || sym === :W || sym === :cl || sym === :cd ||\n"
            "       sym === :cn || sym === :ct || sym === :F || sym === :G\n"
            "        return getfield.(obj, sym)\n"
            "    end\n"
            "    return getfield(obj, sym)\n"
            "end"
        ),
    ]

    applied = 0
    for old, new in patches:
        if old in src:
            src = src.replace(old, new)
            applied += 1
        else:
            print(f"  WARNING: Could not find one expected getproperty block — patch may need manual review.")

    with open(path, "w") as f:
        f.write(src)
    print(f"  CCBlade.jl patch applied ({applied}/3 blocks patched).")
PYEOF

# ── Step 4: Clone CCBladeOpenMDAOExamples (pythoncall branch) ────────────────
echo ""
echo "[4/6] Setting up CCBladeOpenMDAOExamples..."

if [ ! -d "$CCBLADE_OPENMDAO_PATH" ]; then
    echo "  Cloning kanekosh/CCBladeOpenMDAOExamples (pythoncall branch)..."
    git clone --branch pythoncall https://github.com/kanekosh/CCBladeOpenMDAOExamples.git "$CCBLADE_OPENMDAO_PATH"
else
    echo "  $CCBLADE_OPENMDAO_PATH already exists — skipping clone."
fi

# ── Step 5: Install Python packages into conda env ───────────────────────────
echo ""
echo "[5/6] Installing Python packages into '$ENV_NAME'..."

conda run -n "$ENV_NAME" --no-capture-output \
    pip install -e "$CCBLADE_OPENMDAO_PATH" -q

conda run -n "$ENV_NAME" --no-capture-output \
    pip install -e "$REPO_DIR/code/dymos_quadrotor" -q

echo "  Python packages installed."

# ── Step 6: Register Julia packages ──────────────────────────────────────────
echo ""
echo "[6/6] Registering Julia packages (this may take several minutes)..."

conda run -n "$ENV_NAME" --no-capture-output \
    bash -c 'unset VIRTUAL_ENV && python '"$REPO_DIR"'/install_julia_packages.py'

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " To run scripts:"
echo "   conda activate $ENV_NAME"
echo "   unset VIRTUAL_ENV    # if you have a virtualenv active"
echo "   python code/dynOpt/quadrotor_openmdao_setup.py"
echo "============================================================"
