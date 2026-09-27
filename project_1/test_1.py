import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

import Project_1 as cm

def energy(theta, omega):
    return 0.5*cm.m*(cm.R*omega)**2 - cm.m*cm.g*cm.R*np.cos(theta)
  
def test_normal_force_zero_at_top_at_minimum_speed():
    # At the top, gravity alone supplies the centripetal force when R*omega^2 = g.
    omega = np.sqrt(cm.g/cm.R)
    assert abs(cm.normal_force(np.pi, omega)) < 1e-12
  
def test_work_energy_with_all_forces():
    # Change in energy must equal the work done by propulsion, drag, and friction.
    t = np.linspace(0, 50*cm.tau, 50001)
    sol = solve_ivp(cm.dSdt, [0, t[-1]], [0.0, 0.0], t_eval=t, rtol=1e-10, atol=1e-10)
    theta, omega = sol.y
    v = cm.R*omega
    friction = cm.mu*np.abs(cm.normal_force(theta, omega))*np.tanh(omega/cm.eps)
    power = (cm.F_p - cm.c*v - friction)*v
    work = cumulative_trapezoid(power, t, initial=0)
    dE = energy(theta, omega) - energy(theta[0], omega[0])
    assert np.max(np.abs(dE - work)) < 1e-5*np.max(np.abs(work))

def test_asymptotic_g_force_has_converged():
    # Once speed stops growing lap to lap, two back-to-back windows reach the same peak g-force.
    n = len(cm.g_force)
    earlier = cm.g_force[-2*n//16:-n//16].max()
    later = cm.g_force[-n//16:].max()
    assert abs(later - earlier) < 1e-4*later
