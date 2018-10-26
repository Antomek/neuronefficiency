import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Equations come from "Current- and Voltage-Clamp Recordings and Computer Simulations of Kenyon Cells in the Honeybee"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_Na = 140 # maximum conductances (mS/cm^2)
g_NaF = 140
g_NaS = 12
g_KV  = 6
g_KA  = 58.1
g_l  = 0.39
E_Na = 58 # reversal potentials (mV)
E_K  = -81
E_l  = -65

# define auxiliary equations. "inf(v_h, V, s)" describes m_inf, h_inf etc. Tau describes the timescale.
def inf(V, s): return 1 / (1 + np.exp((V-h - V) / s))
def tau_1(tau_max, tau_min, V, V_h, s):
    return (tau_max - tau_min) / (1 + np.exp((V - V_h) / s)) + tau_min
def tau_2(tau_max, tau_min, V, V_h1, V_h2, s_1, s_2):
    return (tau_max - tau_min) / ((1 + np.exp((V - V_h1) / s_1)) * (1 + np.exp((V - V_h2) / s_2))) + tau_min
