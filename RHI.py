import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from "Gamma Oscillation by Synaptic Inhibition in a Hippocampal Interneuronal Network Model"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 35 # maximum conductances (mS/cm^2)
g_K  = 9
g_l  = 0.1
E_Na = 55 # reversal potentials (mV)
E_K  = -90
E_l  = -65

# equations governing opening/closing rates
def a_m(V): return -0.1 * (V + 35) / (np.exp(-0.1*(V + 35)) - 1)
def b_m(V): return 4 * np.exp(-(V + 60)/18)
def a_h(V): return 0.07 * np.exp(-(V + 58)/20)
def b_h(V): return (np.exp(-0.1 * (V + 28)) + 1)**(-1)
def a_n(V): return -0.01 * (V + 34) / (np.exp(-0.1 * (V + 34)) - 1)
def b_n(V): return 0.125 * np.exp(-(V + 44)/80)

def m_inf(V): return a_m(V) / (a_m(V) + b_m(V))

# define the currents in the model
def I_Na(V, h): return g_Na * (m_inf(V)**3) * h * (V - E_Na)
def I_K(V, n): return g_K * (n**4) * (V - E_K)
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
    V, n, h = y

    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (I_K(V, n) + I_Na(V, h) + I_l(V)))
    # enter the equations controlling the gating variables.
    n_dot = 5 * (a_n(V)*(1 - n) - b_n(V) * n)
    h_dot = 5 * (a_h(V)*(1 - h) - b_h(V) * h)
    return [V_dot, n_dot, h_dot]

# enter intial values for V, n, m, h
V_0 = -65
n_0 = a_n(V_0) / (a_n(V_0) + b_n(V_0))
h_0 = a_h(V_0) / (a_h(V_0) + b_h(V_0))
y_0 = [V_0, n_0, h_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 100.0)
numpoints = 5000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(f, t_interval, y_0, 'BDF')

V = soln.y[0, :]
n = soln.y[1, :]
h = soln.y[2, :]

# plot & show solution
plt.plot(soln.t, soln.y[0, :], 'b')
plt.ylabel('Voltage (mv)')
plt.xlabel('Time (ms)')
plt.show()
