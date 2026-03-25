"""
Register Julia packages required by this project.

Run this script once after setting up the conda environment:
    conda activate mdo_quadrotor
    unset VIRTUAL_ENV
    python install_julia_packages.py

The CCBlade.jl repo must already be cloned to ~/CCBlade.jl (done by setup.sh).
"""
import os
import glob

# --- Find Julia 1.9 binary ---
# Do NOT add juliaup to PATH: juliapkg would then pick up juliaup's 'release'
# channel (latest Julia) instead of 1.9.  Point directly to the 1.9 binary.
julia_19_candidates = sorted(glob.glob(
    os.path.expanduser("~/.julia/juliaup/julia-1.9*linux*/bin")
))
if not julia_19_candidates:
    raise RuntimeError(
        "Julia 1.9 not found at ~/.julia/juliaup/julia-1.9*/bin.\n"
        "Install it with:  juliaup add 1.9"
    )
julia_19_bin = julia_19_candidates[0]
os.environ["PATH"] = julia_19_bin + os.pathsep + os.environ.get("PATH", "")

# --- CCBlade.jl path ---
ccblade_path = os.path.expanduser("~/CCBlade.jl")
if not os.path.isdir(ccblade_path):
    raise RuntimeError(
        f"CCBlade.jl not found at {ccblade_path}.\n"
        "Run setup.sh first to clone it."
    )

# --- Register packages ---
import juliapkg
juliapkg.add("CCBlade", "e1828068-15df-11e9-03e4-ef195ea46fa4", dev=True, path=ccblade_path)
juliapkg.add("ConcreteStructs", "2569d6c7-a4a2-43d3-a901-331e8e4be471")
juliapkg.add("ForwardDiff", "f6369f11-7733-5829-9624-2563aa707210")
juliapkg.add("ComponentArrays", "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66")
juliapkg.add("FLOWMath", "6cb5d3fb-0fe8-4cc2-bd89-9fe0b19a99d3")
juliapkg.resolve()
print("Done! Julia packages registered successfully.")
