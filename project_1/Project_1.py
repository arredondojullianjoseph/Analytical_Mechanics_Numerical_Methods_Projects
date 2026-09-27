"""
Simulate and animate a vehicle moving on a circular track (like a rollercoaster loop).
Includes a constant force of propulsion, a resistive force that's linear with speed, and a normal-force-dependent friction force
Calculates the critical propulsive force needed to just clear the loop without losing contact,
and the resulting asymptotic maximum g-force.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    
# Time array and initial conditions
t_eval = np.linspace(0, 50*tau, 1001)
theta0, omega0 = 0.0, 0.0
S0 = [theta0, omega0]

sol = solve_ivp(dSdt, [t_eval[0], max(t_eval)], S0, t_eval=t_eval)

# 1. Make an animation of the motion
theta, omega = sol.y
x, y = r*np.sin(theta), -r*np.cos(theta) # Convert polar/angular coordinates to Cartesian

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
ani.save("circular_motion.gif", writer=animation.PillowWriter(fps=15))

# 2. Make plots of position vs time, speed vs time, and acceleration vs time
v = r*omega
a = r*dSdt(sol.t, sol.y)[1]

fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
axes[0].plot(sol.t, x, label="x")
axes[0].plot(sol.t, y, label="y")
axes[0].set_ylabel("position (m)")
axes[0].legend()
axes[1].plot(sol.t, v)
axes[1].set_ylabel("speed v (m/s)")
axes[2].plot(sol.t, a)
axes[2].set_ylabel("acceleration a (m/s^2)")
axes[2].set_xlabel("t (s)")
fig.tight_layout()
fig.savefig("kinematics.png", dpi=150)

# 3. Make plots of all generalized forces (separately) vs time
# Calculate each force component acting on the system
def normal_force(theta, omega):
    """Calculate the normal force exerted by the track on the vehicle."""
    return m*(r*omega**2 + g*np.cos(theta))
  
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
fig.savefig("forces.png", dpi=150)

# 4. Find the critical propulsive force and asymptotic g-force
if c == 0 and mu == 0:
    raise ValueError("no dissipation: speed grows forever, so there is no asymptotic g-force")

def passed_top(t, S, f_p):
    """triggered when the vehicle successfully makes it over the top."""
    return S[0] - 1.5*np.pi
passed_top.terminal = True
passed_top.direction = 1

def stalled(t, S, f_p):
    """triggered if the vehicle falls back or stalls."""
    return S[1] - 10*eps
stalled.terminal = True
stalled.direction = -1

def lowest_normal_force(f_p):
    """
    Run simulation to see if the vehicle makes the loop. 
    Returns the minimum normal force experienced over the top half.
    """
    # Start at rest on the left vertical section of the track (theta = -pi/2)
    run = solve_ivp(dSdt, [0, 1e6*tau], [-np.pi/2, 0.0], args=(f_p,), events=[passed_top, stalled],
                    dense_output=True, rtol=1e-10, atol=1e-10)
    
    # If the 'passed_top' event didn't trigger, the vehicle stalled/fell
    if run.t_events[0].size == 0:
        return -10.0
        
    theta_run, omega_run = run.sol(np.linspace(0, run.t_events[0][0], 20001))
  
    # Filter for the upper portion of the loop and find the minimum normal force
    return normal_force(theta_run, omega_run)[theta_run >= np.pi/2].min()

# Bracket the root by doubling f_p until the vehicle clears the loop with a positive normal force
f_p_high = m*g
while lowest_normal_force(f_p_high) < 0:
    f_p_high *= 2

# Find the exact f_p where the minimum normal force is exactly 0
f_p_crit = brentq(lowest_normal_force, 0.0, f_p_high)

# Run a long simulation with the critical f_p to reach a steady state
long_run = solve_ivp(dSdt, [0, 1000*tau], [-np.pi/2, 0.0], args=(f_p_crit,),
                     t_eval=np.linspace(0, 1000*tau, 100001), rtol=1e-10, atol=1e-10)

# Calculate g-forces for the entire run
g_force = normal_force(*long_run.y) / (m*g)

# Print the max g-force from the last sixteenth of the run (assuming transients have died out)
print("max g-force = {:.2f} g".format(g_force[-len(g_force)//16:].max()))
