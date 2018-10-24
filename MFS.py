import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from the paper: "Mechanisms of Firing Patterns in Fast-Spiking Cortical Interneurons"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 112.5 # maximum conductances (mS/cm^2)
g_K  = 225
g_l  = 0.25
g_d  = 0.39
theta_m = -24   # half-maximum, a parameter in the original paper
                # NB: theta_m is not given by parameters from supplemental information in paper
E_Na = 50 # reversal potentials (mV)
E_K  = -90
E_l  = -70

#  More parameters:
sigma_m = 11.5
theta_h = -58.3
sigma_h = -6.7
theta_n = -12.4
sigma_n = 6.8
theta_t_h = -60
sigma_t_h = -12
theta_tna = -14.6
sigma_tna = -8.6
theta_tnb = 1.3
sigma_tnb = 18.7
theta_a = -50
sigma_a = 20
theta_b = -70
sigma_b = -6

# Define an auxiliary function
def gamma_f(V, theta, sigma): return 1.0 / (1.0 + np.exp(-(V - theta)/sigma))

# equations governing opening/closing rates.
def m_inf(V): return gamma_f(V, theta_m, sigma_m)
def h_inf(V): return gamma_f(V, theta_h, sigma_h)
def tau_h(V): return 0.5 + 14.0 * gamma_f(V, theta_t_h, sigma_t_h)
def n_inf(V): return gamma_f(V, theta_n, sigma_n)
def tau_n(V): return ((0.087 + 11.4 * gamma_f(V, theta_tna, sigma_tna)) *
                        (0.087 + 11.4 * gamma_f(V, theta_tnb, sigma_tnb)))
def a_inf(V): return gamma_f(V, theta_a, sigma_a)
def b_inf(V): return gamma_f(V, theta_b, sigma_b)
tau_a = 2
tau_b = 150

# define the currents of the model
def I_K(V, n): return g_K * (n**2) * (V - E_K)
def I_Na(V, h): return (m_inf(V)**3) * h * (V - E_Na)
def I_d(V, a, b): return g_d * (a**3) * b * (V - E_K)
def I_l(V): return g_l * (V - E_l)

# define function that is time derivative of voltage: f = dV/dt
def f(t, y):
    # set external current: make sure it is 0 for large times.
    if 0 < t <= 5:
        I_e = 50
    else:
        I_e = 50
    # set the variables that are to be integrated
    V, n, h, a, b = y

    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (I_K(V, n) + I_Na(V, h) + I_d(V, a, b) + I_l(V)))

    # enter the equations controlling the gating variables.
    n_dot = (n_inf(V) - n) / tau_n(V)
    h_dot = (h_inf(V) - h) / tau_h(V)
    a_dot = (a_inf(V) - a) / tau_a
    b_dot = (b_inf(V) - b) / tau_b

    # since y = [V, n, h, a, b] we return [V_dot, n_dot, h_dot, a_dot, b_dot]
    return [V_dot, n_dot, h_dot, a_dot, b_dot]

# enter intial values for V, n, m, h, a, b
V_0 = -70.038
n_0 = 0.000208
h_0 = 0.8522
a_0 = 0.2686
b_0 = 0.5016
y_0 = [V_0, n_0, h_0, a_0, b_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 50.0)
numpoints = 10**3
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(f, t_interval, y_0, 'BDF')

# plot & show solution
plt.plot(soln.t, soln.y[0, :], 'b')
plt.show()
