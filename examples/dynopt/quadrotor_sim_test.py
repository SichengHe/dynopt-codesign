"""
Sensitivity of LQR-controlled quadrotor to linearization errors.

Loads two blade designs:
  - Baseline  : d_init — blade optimized for minimum hover power (sequential)
  - Optimized : d_opt  — blade co-designed with LQR (eps=0.03, from compare_FD_adjoint.py)

Sweeps a scale factor on the full initial-state deviation from hover equilibrium:

    x_init = scale * x_init_base

For scale=1 the perturbation matches x_init_base; larger scales stress-test the
LQR controller's robustness to linearization errors.

Run compare_FD_adjoint.py first to generate quadrotor_d_opt.npy.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../dymos_quadrotor'))

import numpy as np
import matplotlib.pyplot as plt
import CL
from quadrotor_settings import res_funcs_generator, funcs_generator

# ============================================================
#  Load designs
# ============================================================
# --- Baseline: sequential blade optimization ---
twist_init = np.array([28.625901569164, 25.25668202258955, 20.762409457456656, 17.05302822457691,
                       14.166807345926632, 12.050792115126653, 10.687482715596696, 9.623772899617274,
                       8.73787265213354, 7.830294208487746])
chord_by_R_init = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3102845037371853,
                             0.24447688356440495, 0.17144033633125266, 0.05])
d_init = np.concatenate((np.deg2rad(twist_init), chord_by_R_init))
num_cp = len(twist_init)

# --- Optimized: co-design (eps=0.03) from compare_FD_adjoint.py ---
d_opt_path = 'quadrotor_d_opt.npy'
if not os.path.isfile(d_opt_path):
    raise FileNotFoundError(
        f"Optimized design not found at {d_opt_path}.\n"
        "Run compare_FD_adjoint.py first to generate it."
    )
d_opt = np.load(d_opt_path)
print(f"Loaded optimized design from {d_opt_path}")

# ============================================================
#  Fixed simulation parameters
# ============================================================
Q = np.eye(6)
R = np.eye(2) * 0.01
dt = 0.05
T = 30.0
N = int(T / dt)
times = np.linspace(0, T, N)
theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.38, 472.38])

# Base initial state used to define the nominal deviation
x_init_base = np.array([1.0, 1.0, 0.1, 0.5, 0.3, 0.05])

# Failure thresholds
DIVERGE_THRESH = 50.0   # peak state-deviation norm
CONVERGE_TOL   = 0.5    # mean norm in final 20% of trajectory

# Scale-factor sweep: multiply the full deviation by these factors
scale_values = np.round(np.arange(15.0, 17.1, 0.1), 2)
# scale_values = [1]

# ============================================================
#  Setup residual functions (created once per design, reused
#  across all scale values to avoid redundant initialisation)
# ============================================================
print("Setting up residual functions for baseline design...")
res_base, res_reduced_base = res_funcs_generator(num_cp=num_cp)

print("Setting up residual functions for optimized design...")
res_opt, res_reduced_opt = res_funcs_generator(num_cp=num_cp)

# ============================================================
#  Hover equilibrium reference and base deviation
#
#  The state vector w = (x, y, theta, vx, vy, theta_dot).
#  At hover all components are zero; rotor speeds are control
#  variables (not part of w).  theta_0 is the initial guess
#  for the equilibrium solver, not a state.
# ============================================================
x_eq = np.zeros(6)           # hover equilibrium in state space

print(f"Base deviation: {x_init_base}  (norm = {np.linalg.norm(x_init_base):.4f})")

# ============================================================
#  Helper: simulate one design from a given x_init
# ============================================================
def simulate(d, res, res_reduced, x_init_i):
    """
    Returns
    -------
    state_dev : ndarray, shape (6, N)
        State deviation from hover equilibrium.
    w_tgt : ndarray, shape (6,)
        Hover equilibrium state vector.
    cost : float
        LQR cost.
    """
    cl = CL.CL(d, res, res_reduced, Q, R, T, N, x_init_i)
    imp = funcs_generator(d, cl, theta_0)
    imp.set_design(d)
    imp.solve(theta_0=theta_0)
    cost = imp.compute()
    state_dev = imp.w                    # shape (6, N), deviations
    w_tgt = imp.cl.w_tgt                 # shape (6,), hover equilibrium
    return state_dev, w_tgt, cost


def converged(state_dev):
    """
    Returns True only if the LQR controller genuinely drives the system to
    equilibrium.  Failure is declared when:

      1. Hard divergence  — the state-deviation norm ever exceeds DIVERGE_THRESH, OR
      2. Not converging   — the mean norm over the final 20% of the trajectory is
                            both above CONVERGE_TOL *and* not smaller than the mean
                            norm over the 20–40% window (i.e. the trajectory is not
                            heading toward zero by the end of the simulation).

    A system that is simply slow to converge will have a decreasing norm in the
    tail and will NOT be flagged as failed.
    """
    norms = np.linalg.norm(state_dev, axis=0)   # shape (N,)

    # 1. Hard divergence
    if np.max(norms) > DIVERGE_THRESH:
        return False

    n20 = max(N // 5, 1)
    final_mean = np.mean(norms[-n20:])              # last 20%
    early_mean = np.mean(norms[n20: 2 * n20])       # 20–40% window

    # 2. Final portion still large AND not decreasing toward zero
    if final_mean > CONVERGE_TOL and final_mean >= early_mean:
        return False

    return True


# ============================================================
#  State labels for plotting
# ============================================================
state_labels = [
    r'$x$ [m]',
    r'$y$ [m]',
    r'$\theta$ [rad]',
    r'$v_x$ [m/s]',
    r'$v_y$ [m/s]',
    r'$\dot{\theta}$ [rad/s]',
]

# ============================================================
#  Scale-factor sweep
# ============================================================
records = []   # list of dicts with performance data

print(f"\nBase deviation magnitude: {np.linalg.norm(x_init_base):.4f}")
print(f"Starting scale sweep: {list(scale_values)}\n")

for scale in scale_values:
    x_init_i = scale * x_init_base

    print(f"  scale = {scale:.1f}x  |  ", end='', flush=True)

    # --- Simulate ---
    state_dev_base, w_tgt_base, cost_base = simulate(d_init, res_base, res_reduced_base, x_init_i)
    state_dev_opt,  w_tgt_opt,  cost_opt  = simulate(d_opt,  res_opt,  res_reduced_opt,  x_init_i)

    # Absolute states
    state_abs_base = state_dev_base + w_tgt_base[:, None]
    state_abs_opt  = state_dev_opt  + w_tgt_opt[:, None]

    conv_base = converged(state_dev_base)
    conv_opt  = converged(state_dev_opt)

    final_dev_base = np.linalg.norm(state_dev_base[:, -1])
    final_dev_opt  = np.linalg.norm(state_dev_opt[:,  -1])

    print(f"baseline cost={cost_base:.3f} (conv={conv_base})  |  "
          f"optimized cost={cost_opt:.3f}  (conv={conv_opt})")

    records.append({
        'scale':          scale,
        'cost_base':      cost_base,
        'cost_opt':       cost_opt,
        'final_dev_base': final_dev_base,
        'final_dev_opt':  final_dev_opt,
        'conv_base':      conv_base,
        'conv_opt':       conv_opt,
    })

    # --------------------------------------------------------
    #  Plot state histories for this scale value
    # --------------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(11, 7), sharex=True)
    axes = axs.flatten()

    for k in range(6):
        eq_val = (w_tgt_base[k] + w_tgt_opt[k]) / 2   # equilibria should be identical
        axes[k].plot(times, state_abs_base[k, :], 'C0', lw=1.5, label='Baseline')
        axes[k].plot(times, state_abs_opt[k, :],  'C1', lw=1.5, label='Optimized')
        axes[k].axhline(eq_val, color='gray', lw=0.8, linestyle='--', label='Equilibrium')
        axes[k].set_ylabel(state_labels[k])
        if k == 0:
            axes[k].legend(fontsize=8)

    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 1].set_xlabel('Time [s]')

    conv_str_base = 'converged' if conv_base else 'FAILED'
    conv_str_opt  = 'converged' if conv_opt  else 'FAILED'
    fig.suptitle(
        rf'Scale $= {scale:.1f}\times$  |  '
        f'Baseline: cost={cost_base:.2f} ({conv_str_base})  |  '
        f'Optimized: cost={cost_opt:.2f} ({conv_str_opt})',
        fontsize=10
    )
    fig.tight_layout()
    plt.show()

    # Stop if both designs have failed
    if not conv_base and not conv_opt:
        print(f"\n  Both designs failed at scale={scale:.1f}x — stopping sweep.")
        break

# ============================================================
#  Summary table
# ============================================================
print("\n" + "="*70)
print(f"{'scale':>7}  {'cost_base':>10}  {'conv_base':>10}  "
      f"{'cost_opt':>10}  {'conv_opt':>10}")
print("-"*70)
for r in records:
    print(f"{r['scale']:7.1f}  {r['cost_base']:10.3f}  {str(r['conv_base']):>10}  "
          f"{r['cost_opt']:10.3f}  {str(r['conv_opt']):>10}")

# ============================================================
#  Summary performance plot
# ============================================================
scales  = [r['scale']        for r in records]
c_base  = [r['cost_base']    for r in records]
c_opt   = [r['cost_opt']     for r in records]
d_base  = [r['final_dev_base'] for r in records]
d_opt_v = [r['final_dev_opt']  for r in records]

fig, axs = plt.subplots(1, 2, figsize=(11, 4))

axs[0].plot(scales, c_base,  'C0o-', label='Baseline')
axs[0].plot(scales, c_opt,   'C1s-', label='Optimized')
axs[0].set_xlabel(r'Initial deviation scale factor')
axs[0].set_ylabel('LQR cost')
axs[0].set_title('LQR cost vs initial perturbation scale')
axs[0].legend()
axs[0].grid(True, alpha=0.4)

axs[1].plot(scales, d_base,  'C0o-', label='Baseline')
axs[1].plot(scales, d_opt_v, 'C1s-', label='Optimized')
axs[1].axhline(CONVERGE_TOL,   color='gray',  linestyle='--', lw=0.8, label=f'Conv tol = {CONVERGE_TOL}')
axs[1].axhline(DIVERGE_THRESH, color='red',   linestyle=':',  lw=0.8, label=f'Diverge = {DIVERGE_THRESH}')
axs[1].set_xlabel(r'Initial deviation scale factor')
axs[1].set_ylabel(r'$\|x_{final}\|$')
axs[1].set_title('Final state deviation vs initial perturbation scale')
axs[1].legend()
axs[1].grid(True, alpha=0.4)

fig.suptitle('Sensitivity to linearization error (LQR) — scale-factor sweep', fontsize=11)
fig.tight_layout()
plt.show()
