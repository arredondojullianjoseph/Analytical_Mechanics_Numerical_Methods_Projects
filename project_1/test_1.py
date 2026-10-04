import numpy as np
import pytest
from scipy.integrate import solve_ivp, cumulative_trapezoid

import Project_1 as cm

def energy(theta, omega):
    return 0.5*cm.m*(cm.r*omega)**2 - cm.m*cm.g*cm.r*np.cos(theta)
  
def test_normal_force_zero_at_top_at_minimum_speed():
    # At the top, gravity alone supplies the centripetal force when R*omega^2 = g.
    omega = np.sqrt(cm.g/cm.r)
    assert abs(cm.normal_force(np.pi, omega)) < 1e-12
  
def test_work_energy_with_all_forces():
    # Change in energy must equal the work done by propulsion, drag, and friction.
    t = np.linspace(0, 50*cm.tau, 5001)
    sol = solve_ivp(cm.dSdt, [0, t[-1]], [0.0, 0.0], t_eval=t, rtol=1e-8, atol=1e-8)
    theta, omega = sol.y
    v = cm.r*omega
    friction = cm.mu*np.abs(cm.normal_force(theta, omega))*np.tanh(omega/cm.eps)
    power = (cm.f_p - cm.c*v - friction)*v
    work = cumulative_trapezoid(power, t, initial=0)
    dE = energy(theta, omega) - energy(theta[0], omega[0])
    assert np.max(np.abs(dE - work)) < 1e-5*np.max(np.abs(work))

@pytest.fixture(scope="module")
def critical_case():
    f_p_crit = cm.critical_propulsion()
    return f_p_crit, cm.asymptotic_g_force(f_p_crit)

def test_asymptotic_g_force_has_converged(critical_case):
    # Once speed stops growing lap to lap, consecutive laps reach the same peak g-force.
    _, peaks = critical_case
    assert abs(peaks[-1] - peaks[-2]) < 1e-4*peaks[-1]

def test_first_loop_normal_force_nonnegative_at_critical(critical_case):
    # F_p,crit is first-loop grazing: min F_N on the top half is zero, not negative.
    f_p_crit, _ = critical_case
    n_min = cm.lowest_normal_force(f_p_crit)
    assert n_min >= -1e-8
    assert abs(n_min) < 1e-8
