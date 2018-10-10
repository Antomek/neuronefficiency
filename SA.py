import numpy as np
import scipy as sp

# define function that is time derivative of voltage: f = dV/dt
def f(V, x, y):
    # set the relevant constants
    I_e, C_m, g_K, g_Na, g_l, V_K, V_Na, V_l = x
    # set the gating variables
    n, m, h = y
    # define DV/dt
    f = 1/C_m * (I_e - (g_K * (N**4) * (V - V_K) + g_Na * (M**3)*h) *
        (V - V_Na) + g_l * (V - V_l))
    return f

# enter the values of the constants. Values taken from Table 3 in `Membrane Current In Nerve`
x = [I_e, 1.0, 36, 120, 0.3, 12, -115, -10.613]

# enter the equations controlling the gating variables.
# 'g_1' is dn/dt, 'g_2' is dm/dt, 'g_3' is dh/dt
def g_1(n, a, b):
    return a * (a - n) - b*n

def g_2(m, a, b):
    return a * (1 - m) - b*m

def g_3(h, a, b):
    return a * (1-h) - b*h

# equations governing the opening/closing rates.
def a_n(V):
    return 0.01 * (V + 10) / ( np.exp((V+10)/10) - 1)
def b_n(V):
    return 0.125 * np.exp(V/80)
def a_m(V):
    return 0.1 * (V + 25) / (np.exp((V + 25)/10) - 1)
def b_m(V):
    return 4 * np.exp(V / 18)
def a_h(V):
    0.07 * np.exp(V / 20)
def b_h(V):
    1 / (np.exp((V + 30)/10) + 1)
