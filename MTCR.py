import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from "Thalamocortical Relay Fidelity Varies Across Subthalamic Nucleus Deep \
# Brain Stimulation Protocols in a Data-Driven Computational Model"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 3 # maximum conductances (mS/cm^2)
g_K  = 5
g_T = 5
g_l  = 0.05
E_Na = 50 # reversal potentials (mV)
E_K  = -90
E_l  = -70

# equations governing opening / closing rates
def m_inf(V): return 1 / (1 + np.exp(-(V + 37) / 7))
def h_inf(V): return 1 / (1 + np.exp(V + 41) / 4)
def p_inf(V): return 1 / (1 + np.exp(-(V + 60) / 6.2))
def r_inf(V): return 1 / (1 + np.exp((V + 84) / 4))

def a_1(V): return 0.128 * np.exp(-(V + 46) / 18)
def b_1(V): return 4 / (1 + np.exp(-(V + 23) / 5))
def tau_h(V): return 1 / (a_1(V) + b_1(V))
def tau_r(V): return 0.4 * (28 + np.exp(-(V + 25) / 10.5))

# define the currents in the model
def I_Na(V, h): return g_Na * (m_inf(V)**3) * h * (V - E_Na)
def I_K(V, h): return g_K * ((0.75 * (1 - h))**4) * (V - E_K)
def I_T(V, r): return g_T * (p_inf(V)**2) * r * V
def I_l(V): return g_l * (V - E_l)

def f(t, y):
    # set external current: make sure it is 0 for large times.
    if 0 < t <= 10:
        I_e = 0
    elif 10 < t <= 40:
        I_e = 3
    else:
        I_e = 0
    # define variables that are to be integrated
    V, h, r = y

    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (I_K(V, h) + I_Na(V, h) + I_T(V, r) + I_l(V)))
    # enter the equations controlling the gating variables.
    h_dot = (h_inf(V) - h) / tau_h(V)
    r_dot = (r_inf(V) - r) / tau_r(V)
    return [V_dot, h_dot, r_dot]

# enter intial values for V, n, m, h
V_0 = -65
h_0 = h_inf(V_0)
r_0 = r_inf(V_0)
y_0 = [V_0, h_0, r_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 300.0)
numpoints = 5000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(f, t_interval, y_0, 'BDF')

V = soln.y[0, :]
h = soln.y[1, :]
r = soln.y[2, :]

# plot & show solution
plt.plot(soln.t, soln.y[0, :], 'b')
plt.ylabel('Voltage (mv)')
plt.xlabel('Time (ms)')
plt.show()
