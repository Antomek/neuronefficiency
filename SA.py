import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# define function that is time derivative of voltage: f = dV/dt
def f(t, y, x):
    # set the relevant constants
    C_m, g_K, g_Na, g_l, V_K, V_Na, V_l = x
    # set external current: make sure it is 0 for large times.
    if t <= 5:
        I_e = 10
    else:
        I_e = 0
    # set the variables that are to be integrated
    V, n, m, h = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (g_K * (n**4) * (V - V_K) + g_Na * (m**3)* h *
        (V - V_Na) + g_l * (V - V_l)))

    # equations governing opening/closing rates.
    a_n = 0.01 * (10 - V) / (np.exp((10 - V)/10) - 1)
    b_n = 0.125 * np.exp(-V/80)
    a_m = 0.1 * (25 - V) / (np.exp((25 - V)/10) - 1)
    b_m = 4 * np.exp(-V / 18)
    a_h = 0.07 * np.exp(-V / 20)
    b_h = 1 / (np.exp((30 - V)/10) + 1)

    tau_n = 1 / (a_n + b_n)
    n_inf = a_n / (a_n + b_n)
    tau_m = 1 / (a_m + b_m)
    m_inf = a_m / (a_m + b_m)
    tau_h = 1 / (a_h + b_h)
    h_inf = a_h / (a_h + b_h)

    # enter the equations controlling the gating variables.
    n_dot = (n_inf - n) / tau_n
    m_dot = (m_inf - m) / tau_m
    h_dot = (h_inf - h) / tau_h
    # since y = [V, n, m, h] we return [V_dot, n_dot, m_dot, h_dot]
    return [V_dot, n_dot, m_dot, h_dot]

# enter the values of the constants. Values taken from Table 3 in `Membrane Current In Nerve`
# values are entered like: conts = [C_m, g_K, g_Na, g_l, V_K, V_Na, V_l]
conts = [1, 36, 120, 0.3, -77, +50, -55]

# define n_inf etc. for proper initialisation of the gating variables
def n_inf(V):
    a_n = 0.01 * (10 - V) / (np.exp((10 - V)/10) - 1)
    b_n = 0.125 * np.exp(-V/80)
    return a_n / (a_n + b_n)
def m_inf(V):
    a_m = 0.1 * (25 - V) / (np.exp((25 - V)/10) - 1)
    b_m = 4 * np.exp(-V / 18)
    return a_m / (a_m + b_m)
def h_inf(V):
    a_h = 0.07 * np.exp(-V / 20)
    b_h = 1 / (np.exp((30 - V)/10) + 1)
    return a_h / (a_h + b_h)

# enter intial values for V, n, m, h
V_0 = -65
n_0 = n_inf(V_0)
m_0 = m_inf(V_0)
h_0 = h_inf(V_0)
y_0 = [V_0, n_0, m_0, h_0]
print('n_0 = {},  m_0 = {},  h_0 = {}'.format(n_0, m_0, h_0))

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 40.0)
numpoints = 1000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(lambda t, y: f(t, y, conts), t_interval, y_0, 'RK45', t_points)

# plot & show solution
plt.plot(soln.t, soln.y[0, :])
plt.show()
