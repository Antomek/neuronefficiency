import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from paper: "NEURAL REPETITIVE FIRING MODIFICATIONS OF THE HODGKIN-HUXLEY AXON SUGGESTED BY EXPERIMENTAL RESULTS FROM CRUSTACEAN AXONS"
# define function that is time derivative of voltage: f = dV/dt
def f(t, y, x):
    # set the relevant constants
    C_m, g_K, g_Na, g_A, g_l, V_K, V_Na, V_A, V_l = x
    # set external current: make sure it is 0 for large times.
    if t <= 1:
        I_e = 10
    else:
        I_e = 0
    # set the variables that are to be integrated
    V, n, m, h, A, B = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (g_K * (n**4) * (V - V_K) + g_Na * (m**3)*h *
        (V - V_Na) + g_A * (A**3) * B * (V - V_A) + g_l * (V - V_l)))

    # equations governing opening/closing rates.
    a_n = -0.01 * (V + 50 - 4.3) / (np.exp(-(V + 50 - 4.3)/10) - 1)
    b_n = 0.125 * np.exp(-(V + 60 - 4.3)/80)
    a_m = -0.1 * (V + 35 - 5.3) / (np.exp(-(V + 35 - 5.3)/10 - 1))
    b_m = 4 * np.exp(-(V + 60 - 5.3)/18)
    a_h = 0.07 * np.exp(-(V + 60 - 12)/20)
    b_h = 1 / (np.exp(-(V + 30 - 12)/10 + 1))

    # equations that determine the gating variables differential equations
    tau_n = (2/3.8) / (a_n + b_n)
    n_inf = a_n / (a_n + b_n)
    tau_m = (1/3.8) / (a_m + b_m)
    m_inf = a_m / (a_m + b_m)
    tau_h = (1/3.8) / (a_h + b_h)
    h_inf = a_h / (a_h + b_h)
    A_inf = (0.0761 * np.exp((V + 94.22)/31.84) / (1 + np.exp((V + 1.17) / 28.93)))**(1/3)
    tau_A = 0.3632 + 1.158 / (1 + np.exp((V + 55.96) / 20.21))
    B_inf = 1 / ((1 + np.exp((V + 53.3) / 14.54))**4)
    tau_B = 1.24 + 2.678 / (1 + np.exp((V + 50) / 16.027))

    # enter the equations controlling the gating variables.
    n_dot = (1 / tau_n) * (n_inf - n)
    m_dot = (1 / tau_m) * (m_inf - m)
    h_dot = (1 / tau_h) * (h_inf - h)
    A_dot = (1 / tau_A) * (A_inf - A)
    B_dot = (1 / tau_B) * (B_inf - B)
    # since w = [V, n, m, h] we return [V_dot, n_dot, m_dot, h_dot, A_dot, B_dot]
    return [V_dot, n_dot, m_dot, h_dot, A_dot, B_dot]

# values are entered like: conts = [C_m, g_K, g_Na, g_A, g_l, V_K, V_Na, V_A, V_l]
conts = [1, 20, 120, 47.7, 0.3, -72, +55, -75, -17]
# enter intial values for V, n, m, h
V_0 = -50
n_0 = 0
m_0 = 0
h_0 = 0
A_0 = 0
B_0 = 0
y_0 = [V_0, n_0, m_0, h_0, A_0, B_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 30.0)
numpoints = 1000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(lambda t, y: f(t, y, conts), t_interval, y_0, 'RK45', t_points)

# plot & show solution
plt.plot(soln.t, soln.y[0, :])
plt.show()
