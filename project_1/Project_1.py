"""
Simulate a bead locked to a vertical circular wire of radius R.

Includes a constant propulsive force, linear drag, and friction proportional
to |F_N|. The bead cannot leave the wire, so the constraint and the |F_N|
friction law stay in force even if a coaster would have lost contact.

Running this file animates the default parameters, then finds the critical
propulsion at which the inward normal first hits zero on the opening loop
(the grazing-coaster diagnostic) and reports the asymptotic peak g-force.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# System parameters
m, r, g, c, mu = 1.0, 2.0, 9.81, 0.5, 0.2
f_p = 10
tau = np.sqrt(r/g)       # Characteristic time scale
eps = 1e-3/tau           # Small parameter to smooth the friction sign function

def dSdt(t, S, f_p=f_p):
    """
    Compute the state derivatives for the ODE solver.
    """
    theta, omega = S
  
    # Angular acceleration. 
    alpha = (f_p / (m*r) - (g/r)*np.sin(theta) - (c/m)*omega 
             - np.abs(mu*(omega**2 + (g/r)*np.cos(theta)))*np.tanh(omega/eps)) #np.tanh(omega/eps) smoothly approximates np.sign(omega)
    return [omega, alpha]

def normal_force(theta, omega):
    """Inward normal force of the wire on the bead."""
    return m*(r*omega**2 + g*np.cos(theta))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    out_dir = Path(__file__).resolve().parent

    # Default run: f_p = 10 N, rest at the bottom
    t_eval = np.linspace(0, 50*tau, 1001)
    theta0, omega0 = 0.0, 0.0
    sol = solve_ivp(dSdt, [t_eval[0], t_eval[-1]], [theta0, omega0], t_eval=t_eval)
    theta, omega = sol.y
    x, y = r*np.sin(theta), -r*np.cos(theta) # Convert polar/angular coordinates to Cartesian

    # 1. Make an animation of the motion
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1.3*r, 1.3*r)
    ax.set_ylim(-1.3*r, 1.3*r)
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), r, fill=False, linestyle="--", color="gray"))
    ball, = ax.plot([], [], "o", color="tab:red", markersize=10)

    def draw_frame(frame):
        """Update function for the animation."""
        ball.set_data([x[frame]], [y[frame]])
        return ball,

    ani = animation.FuncAnimation(fig, draw_frame, frames=range(0, len(sol.t), 4), interval=25, blit=True)
    ani.save(out_dir / "circular_motion.gif", writer=animation.PillowWriter(fps=15))

    # 2. Make plots of position vs time, speed vs time, and acceleration vs time
    v = r*omega
    a_t = r*dSdt(sol.t, sol.y)[1]
    a_n = r*omega**2

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    axes[0].plot(sol.t, x, label="x")
    axes[0].plot(sol.t, y, label="y")
    axes[0].set_ylabel("position (m)")
    axes[0].legend()
    axes[1].plot(sol.t, v)
    axes[1].set_ylabel("speed v (m/s)")
    axes[2].plot(sol.t, a_t, label="tangential")
    axes[2].plot(sol.t, a_n, label="centripetal")
    axes[2].set_ylabel("acceleration (m/s^2)")
    axes[2].legend()
    axes[2].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "kinematics.png", dpi=150)

    # 3. Make plots of all generalized forces (separately) vs time
    Q = r * np.array([
        np.full_like(sol.t, f_p),
        -m*g*np.sin(theta),
        -c*v,
        -mu*np.abs(normal_force(theta, omega))*np.tanh(omega/eps)
    ])
    labels = ["propulsion Q (N·m)", "gravity Q (N·m)", "linear drag Q (N·m)", "friction Q (N·m)"]

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    for ax, q, label in zip(axes, Q, labels):
        ax.plot(sol.t, q)
        ax.set_ylabel(label)
    axes[3].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out_dir / "forces.png", dpi=150)

# 4. Find a scenario where the object starts from rest at a vertical
#    section of track and makes it over the top of the loop with a
#    normal force of zero the first time around so that if it wasn't
#    locked to the track like a rollercoaster, that it would make it
#    around the loop without losing contact.
def passed_top(t, S, f_p):
    """triggered when the bead successfully makes it over the top."""
    return S[0] - 1.5*np.pi
passed_top.terminal = True
passed_top.direction = 1

def stalled(t, S, f_p):
    """triggered if the bead falls back or stalls."""
    return S[1] - 10*eps
stalled.terminal = True
stalled.direction = -1

def lowest_normal_force(f_p):
    """
    Run simulation to see if the bead makes the loop.
    Returns the minimum inward normal force over the top half of the first loop.
    """
    # Start at rest on the left vertical section of the wire (theta = -pi/2)
    run = solve_ivp(dSdt, [0, 100*tau], [-np.pi/2, 0.0], args=(f_p,), events=[passed_top, stalled],
                    dense_output=True, rtol=1e-8, atol=1e-8)
    
    # If the 'passed_top' event didn't trigger, the bead stalled/fell
    if run.t_events[0].size == 0:
        return -np.inf
        
    theta_run, omega_run = run.sol(np.linspace(0, run.t_events[0][0], 1001))
  
    # Filter for the upper portion of the loop and find the minimum normal force
    return normal_force(theta_run, omega_run)[theta_run >= np.pi/2].min()


def critical_propulsion():
    """
    Return F_p,crit: propulsion at which min F_N over the top half of the first
    loop is zero, starting from rest at the left vertical.
    """
    # Bracket the root by doubling f_p until the bead clears the loop with a positive normal force
    f_p_high = m*g
    while lowest_normal_force(f_p_high) < 0:
        f_p_high *= 2

    # Find the exact f_p where the minimum normal force is exactly 0
    return brentq(lowest_normal_force, 0.0, f_p_high)

if __name__ == "__main__":
    f_p_crit = critical_propulsion()

# Given those same conditions and being allowed to continue with the
# same propulsive force, what is the largest g-force the vehicle will
# attain once in an asymptotic condition where loop by loop it is no
# longer gaining speed?
def asymptotic_g_force(f_p):
    """
    Return F_N/(m g) sampled along a 1000*tau integration at this propulsion.
    """
    if c == 0 and mu == 0:
        raise ValueError("no dissipation: speed grows forever, so there is no asymptotic g-force")

    # Run a long simulation with the critical f_p to reach a steady state
    long_run = solve_ivp(dSdt, [0, 1000*tau], [-np.pi/2, 0.0], args=(f_p,),
                         dense_output=True, rtol=1e-8, atol=1e-8)
    t_sample = np.linspace(0, 1000*tau, 100001)
    return normal_force(*long_run.sol(t_sample)) / (m*g)

if __name__ == "__main__":
    g_force = asymptotic_g_force(f_p_crit)
    print("max g-force = {:.2f} g".format(g_force[-len(g_force)//16:].max()))
