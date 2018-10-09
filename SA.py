import numpy

# define function that is time derivative of voltage: f = dV/dt
def f(V, x, y):
    # set the relevant constants
    I_e, g_K, g_Na, V_K, V_Na, V_l = x
    # set the 
    n, m, h = y
