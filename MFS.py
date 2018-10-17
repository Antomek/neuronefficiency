import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from the paper: "Mechanisms of Firing Patterns in Fast-Spiking Cortical Interneurons"

# define function that is time derivative of voltage: f = dV/dt
def f(t, y, x):
    # set the relevant constants
    C_m, g_K, g_Na, g_l, g_d, theta_m, V_K, V_Na, V_l = x
    # set external current: make sure it is 0 for large times.
    if t <= 1:
        I_e = 100
    else:
        I_e = 0
    # set noise current in input
    D = 0
    I_noise = np.sqrt(2*D*np.random.normal(0, 1))
    # set the variables that are to be integrated
    V, n, h, a, b = y

    # equations governing opening/closing rates.
    m_inf = (1 + np.exp(-(V - theta_m)/11.5))**(-1)
    h_inf = (1 + np.exp(-(V + 58.3)/-6.7))
    tau_h = 0.5 + 14*((1 + np.exp(-(V + 60)/-12))**(-1))
    n_inf = (1 + np.exp(-(V + 12.4)/6.8))**(-1)
    tau_n = (0.087 + 11.4 * (1 + np.exp(V + 14.6)/8.6)**(-1)) *
            (0.087 + 11.4 * (1 + np.exp(-(V - 1.3)/18.7))**(-1))
    a_inf = (1 + np.exp(-(V + 50)/20))**(-1)
    b_inf = (1 + np.exp(-(V + 70)/-6))**(-1)
    tau_a = 2
    tau_b = 150

    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e + I_noise - (g_K * (n**2) * (V - V_K) + g_Na * (m_inf**3) * h *
        (V - V_Na) + g_d * (a**3) * b * (V - V_K) + g_l * (V - V_l)))

    # enter the equations controlling the gating variables.
    n_dot = (n_inf - n) / tau_n
    h_dot = (h_inf - h) / tau_h
    a_dot = (a_inf - a) / tau_a
    b_dot = (b_inf - b) / tau_b

    # since w = [V, n, m, h] we return [V_dot, n_dot, m_dot, h_dot]
    return [V_dot, n_dot, h_dot, a_dot, b_dot]

# values are entered like: conts = [C_m, g_K, g_Na, g_l, g_d, theta_m, V_K, V_Na, V_l]
# NB: theta_m and g_d are varied throughout the paper
conts = [1, 36, 120, 0.3, -77, +50, -55]
# enter intial values for V, n, m, h
V_0 = -55
n_0 = 0
m_0 = 0
h_0 = 0
y_0 = [V_0, n_0, m_0, h_0]

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