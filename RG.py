import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Model after paper Synchronization of Golgi and Granule Cell Firing in a Detailed Network Model of the Cerebellar Granule Cell Layer"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 120 # maximum conductances (mS/cm^2)
g_K  = 36
g_l  = 0.3
E_Na = 50 # reversal potentials (mV)
E_K  = -77
E_l  = -55

# auxiliary equations as defined in paper
def exp(V, A, B, C, V_0): return A * np.exp(B * (V - V_0)) + C
def sig(V, A, B, V_0): return A / (1 + np.exp(B * (V - V_0)))
def lin(V, A, B, V_0): return A * (V - V_0) / (np.exp(B * (V - V_0)) - 1)
def KC_alpha(V, A, B, C, con): return A / (1 + C * np.exp(B * V) / con) # con stands for Ca2+ concentration
def KC_beta(V, A, B, C, con): return A / (1 + con / (C * np.exp(B * V)))

# Parameter values of voltage-gated channels (cf. Table 1 in paper). 'am' is alpha fn for activation function m, 'bh' is beta function for inactivation function h etc.
A_NaF_am, B_NaF_am, V0_NaF_am = [7.5, 0.081, -39]
A_NaF_bm, B_NaF_bm, V0_NaF_bm = [7.5, -0.066, -39]
A_NaF_ah, B_NaF_ah, V0_NaF_ah = [0.6, -0.089, -50]
A_NaF_bh, B_NaF_bh, V0_NaF_bh = [0.6, 0.089, -50]
A_NaF_am, B_NaF_am, V0_NaF_am = [7.5, 0.081, -39]
A_Kdr_am, B_Kdr_am, V0_Kdr_am = [0.85, 0.073, -38]
A_Kdr_bm, B_Kdr_bm, V0_Kdr_bm = [0.85, -0.018, -38]
A_Kdr_ah, B_Kdr_ah, C_Kdr_ah, V0_Kdr_ah = [3 * (10**(-4)), -0.08, 35 * (10**(-4)), -46]
A_Kdr_bh, B_Kdr_bh, V0_Kdr_bh = [5.5 * (10**(-3)), -0.0807, -44]
A_CaL_am, B_CaL_am, V0_CaL_am = [8, -0.072, 5]
A_CaL_bm, B_CaL_bm, V0_CaL_bm = [0.1, 0.2, -8.9]
A_CaL_ah, B_CaL_ah, V0_CaL_ah = [0.025, -0.05, -60]
A_CaL_bh, B_CaL_bh, C_Cal_bh, V0_CaL_bh = [-0.025, -0.05, 0.025, -60]


# define the currents in the model
def I_Na(V, m, h): return g_Na * (m**3)* h * (V - E_Na)
def I_K(V, n): return g_K * (n**4) * (V - E_K)
def I_l(V): return g_l * (V - E_l)

# define function that will return time derivatives for integration
def f(t, y):
    # set external current: make sure it is 0 for large times.
    if t <= 1:
        I_e = 20
    else:
        I_e = 0
    # set the variables that are to be integrated
    V, n, m, h = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (I_K(V, n) + I_Na(V, m, h) + I_l(V)))
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
soln = sp_int.solve_ivp(f, t_interval, y_0, 'BDF')

V = soln.y[0, :]
n = soln.y[1, :]
m = soln.y[2, :]
h = soln.y[3, :]

# plot & show solution
plt.figure(1)
plt.plot(soln.t, V, 'g') # a V, t plot
f, (ax1, ax2, ax3) = plt.subplots(3, sharex = True)
# plt.plot(soln.t, [I_Na(i, j, k) for i,j,k in zip(V, m, h)], 'blue', label='sodium current')
# plt.plot(soln.t, [I_K(i, j) for i,j in zip(V, n)], 'red', label='potassium current')
# plt.plot(soln.t, [I_l(i) for i in V], 'green', label='leak current')
t_i = 0
t_f = 50
ax1.plot(soln.t[t_i:t_f], [I_Na(i,j,k) + I_K(i,l) + I_l(i) for i,j,k,l in zip(V[t_i:t_f],m[t_i:t_f],h[t_i:t_f],n[t_i:t_f])], 'g', label='Ion currents')
ax1.set_ylabel('Current (uA)')
ax1.set_xlabel('Time (ms)')
ax2.plot(soln.t[t_i:t_f], m[t_i:t_f], 'b', label='m')
ax2.plot(soln.t[t_i:t_f], h[t_i:t_f], 'y', label='h')
ax2.plot(soln.t[t_i:t_f], n[t_i:t_f], 'r', label='n')
ax2.set_ylabel('Probability')
ax2.set_xlabel('Time (ms)')
ax3.plot(soln.t[t_i:t_f], [-m_i + ((1 - h_i) + n_i) for m_i, h_i, n_i in zip(m[t_i:t_f], h[t_i:t_f], n[t_i:t_f])], label = 'inactivation - activation')
plt.legend()
plt.show()
