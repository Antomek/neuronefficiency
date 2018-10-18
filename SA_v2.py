import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 120 # maximum conductances (mS/cm^2)
g_K  = 36
g_l  = 0.3
E_Na = 50 # reversal potentials (mV)
E_K  = -77
E_l  = -55

# equations governing opening/closing rates.
def tau_m(V): return -1/(0.1*(V+40)/(1-np.exp(-(V+40)/10))+4*np.exp(-(V+65)/18))
def m_inf(V): return -(0.1*(V+40)/(1-np.exp(-(V+40)/10)))*tau_m(V)
def tau_n(V): return -1/(0.01*(V+55)/(1-np.exp(-(V+55)/10))+0.125*np.exp(-(V+65)/80))
def n_inf(V): return -(0.01*(V+55)/(1-np.exp(-(V+55)/10)))*tau_n(V)
def tau_h(V): return -1/(0.07*np.exp(-(V+65)/20)+(1/(1+np.exp(-(V+35)/10))))
def h_inf(V): return -0.07*np.exp(-(V+65)/20)*tau_h(V)

# define function that will return time derivatives for integration
def f(t, y):
    # set external current: make sure it is 0 for large times.
    if t <= 1:
        I_e = 10
    else:
        I_e = 0
    # set the variables that are to be integrated
    V, n, m, h = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (g_K * (n**4) * (V - E_K) + g_Na * (m**3)* h *
        (V - E_Na) + g_l * (V - E_l)))
    # enter the equations controlling the gating variables.
    n_dot = (n - n_inf(V)) / tau_n(V)
    m_dot = (m - m_inf(V)) / tau_m(V)
    h_dot = (h - h_inf(V)) / tau_h(V)
    return [V_dot, n_dot, m_dot, h_dot]

# enter intial values for V, n, m, h
V_0 = -65
n_0 = 0.317
m_0 = 0.052
h_0 = 0.596
y_0 = [V_0, n_0, m_0, h_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 40.0)
numpoints = 1000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(f(t, y), t_interval, y_0, 'RK45', t_points)

# plot & show solution
plt.plot(soln.t, soln.y[0, :])
plt.show()
