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
g_ext = 1.5*g_Na # large value so that V_dot will follow I_ext quickly
E_Na = 50 # reversal potentials (mV)
E_K  = -77
E_l  = -55
# E_command = 20 # define a desired value of the voltage at which the 'clamp' will keep the 'membrane' fixed

# equations governing opening/closing rates.
def tau_m(V): return -1/(0.1*(V+40)/(1-np.exp(-(V+40)/10))+4*np.exp(-(V+65)/18))
def m_inf(V): return -(0.1*(V+40)/(1-np.exp(-(V+40)/10)))*tau_m(V)
def tau_n(V): return -1/(0.01*(V+55)/(1-np.exp(-(V+55)/10))+0.125*np.exp(-(V+65)/80))
def n_inf(V): return -(0.01*(V+55)/(1-np.exp(-(V+55)/10)))*tau_n(V)
def tau_h(V): return -1/(0.07*np.exp(-(V+65)/20)+(1/(1+np.exp(-(V+35)/10))))
def h_inf(V): return -0.07*np.exp(-(V+65)/20)*tau_h(V)

# define the currents in the model
def I_Na(V, m, h): return g_Na * (m**3)* h * (V - E_Na)
def I_K(V, n): return g_K * (n**4) * (V - E_K)
def I_l(V): return g_l * (V - E_l)

# def I_ext(V): return g_ext * (E_command - V)

I_ext_vals = []

# define function that will return time derivatives for integration
def f(t, y):
    if 0 <= t <= 20:
        E_command = lambda t: -65
    elif 20 < t <= 25:
        E_command = lambda t: 17 * (t - 20) - 65
    else:
        E_command = lambda t: 20
    # set the variables that are to be integrated
    V, n, m, h = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (g_ext * (E_command(t) - V) - (I_K(V, n) + I_Na(V, m, h) + I_l(V)))
    # enter the equations controlling the gating variables.
    n_dot = (n - n_inf(V)) / tau_n(V)
    m_dot = (m - m_inf(V)) / tau_m(V)
    h_dot = (h - h_inf(V)) / tau_h(V)
    I_ext_vals.append(g_ext * (E_command(t) - V))
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
soln = sp_int.solve_ivp(f, t_interval, y_0, 'LSODA', max_step=10**(-3))

V = soln.y[0, :]
n = soln.y[1, :]
m = soln.y[2, :]
h = soln.y[3, :]

# plot & show solution
plt.figure(1)
plt.plot(soln.t, V, 'g') # a V, t plot
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
f, (ax1, ax2) = plt.subplots(2, sharex = True)
ax1.plot(np.linspace(t_interval[0], t_interval[1], len(I_ext_vals)), I_ext_vals, 'b', label='External current')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('I_ext (uA)')
ax2.plot(soln.t, [I_Na(i,j,k) + I_K(i,l) + I_l(i) for i,j,k,l in zip(V, m, h, n)], 'g', label='Ionic current')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('I_ion (uA)')
plt.figure(3)
plt.plot([I_Na(i,j,k) + I_K(i,l) + I_l(i) for i,j,k,l in zip(V, m, h, n)], V, 'r')

plt.show()
