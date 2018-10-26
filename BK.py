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
g_KST = 8.11
g_l  = 0.39
E_Na = 58 # reversal potentials (mV)
E_K  = -81
E_l  = -65

# define auxiliary equations. "inf(v_h, V, s)" describes m_inf, h_inf etc. Tau describes the timescale.
def inf(V, V_h, s): return 1 / (1 + np.exp((V_h - V) / s))
def tau_1(tau_max, tau_min, V, V_h, s):
    return (tau_max - tau_min) / (1 + np.exp((V - V_h) / s)) + tau_min
def tau_2(tau_max, tau_min, V, V_h1, V_h2, s_1, s_2):
    return (tau_max - tau_min) / ((1 + np.exp((V - V_h1) / s_1)) * (1 + np.exp((V - V_h2) / s_2))) + tau_min

# Parameters used in simulation of the Kenyon cell (cf. table 2 of paper)
# for Na:
Vh_Na = -30.1
s_Na = 6.65
Vh1_Na = -20.3
s1_Na = 6.45
tau_max_Na = 0.83
tau_min_Na = 0.093
# for NaF:
Vh_NaF = -51.4
s_NaF = 5.9
Vh1_NaF = -8.03
s1_Naf = 8.69
tau_max_NaF = 1.66
tau_min_NaF = 0.12
# for NaS:
Vh_NaS = -51.4
s_NaS = 5.9
Vh1_NaS = -32.6
s1_NaS = 8
tau_max_NaS = 12.24
tau_min_NaS = 1.9
# for K,V:
Vh_KV = -37.6
s_K = 27.24
Vh1_KV = 45
s1_KV = 13.71
tau_max_KV = 3.53
tau_min_KV = 1.85
# for KA (m for activation, h for inactivation variable):
Vh_KAm = -20.1
s_KAm = 16.1
Vh1_KAm = -70
s1_KAm = -4
Vh2_KAm = -20
s2_KAm = 12
tau_max_KAm = 1.65
tau_min_KAm = 0.35
Vh_KAh = -74.7
s_KAh = 7
Vh1_KAh = -60
s1_KAh =  -25
Vh2_KAh = -62
s2_KAh = 16
tau_max_KAh = 90
tau_min_KAh = 2.5
# for KST:
Vh_KSTm = -20.1
s_KSTm = 16.1
Vh1_KSTm = 20
s1_KSTm = 20
tau_max_KSTm = 5
tau_min_KSTm = 0.5
Vh_KSTh = -74.7
s_KSTh = 7
tau_max_KSTh = 200
tau_min_KSTh = 150
Vh1_KSTh = 52
s1_KSTh = 15

# equations governing opening / closing rates
def m_inf_Na(V): return inf(V, Vh_Na, s_Na)
def tau_m_Na(V): return tau_1(tau_max_Na, tau_min_Na, V, Vh1_Na, s1_Na)
def h_inf_NaF(V): return inf(V, Vh_NaF, s_NaF)
def tau_h_NaF(V): return tau_1(tau_max_NaF, tau_min_NaF, V, Vh1_NaF, s1_NaF)
def h_inf_NaS(V): return inf(V, Vh_NaF, s_NaF)
def tau_h_NaS(V): return tau_1(tau_max_NaS, tau_min_NaS, V, Vh1_NaS, s1_NaS)
def m_inf_Kv(V): return inf(V, Vh_KV, s_KV)
def tau_m_KV(V): return tau_1(tau_max_KV, tau_min_KV, V, Vh1_KV, s1_KV)
def m_inf_KA(V): return inf(V, Vh_KAm, s_KAm)
def tau_m_KA(V): return tau_2(tau_max_KAm, tau_min_KAm, V, Vh1_KAm, Vh2_KAm, s1_KAm, s2_KAm)
def h_inf_KA(V): return inf(V, Vh_KAh, s_KAh)
def tau_h_KA(V): return tau_2(tau_max_KAh, tau_min_KAh, V, Vh1_KAh, Vh2_KAm, s1_KAh, s2_KAh)
def m_inf_KST(V): return inf(V, Vh_KSTm, s_KSTm)
def tau_m_KST(V): return tau_1(tau_max_KSTm, tau_min_KSTm, V, Vh1_KSTm, s1_KSTm)
def h_inf_KST(V): return inf(V, Vh_KSTh, s_KSTh)
def tau_h_KST(V): return tau_1(tau_max_KSTh, tau_min_KSTh, V, Vh1_KSTh, s1_KSTh)

# define the currents in the model
def I_Na1(V, m_Na, h_NaF): return g_Na * (m_Na**3) * h_NaF * (V - E_Na)
def I_Na2(V, m_Na, h_NaF): return g_NaF * (m_Na**3) * h_NaF * (V - E_Na)
def I_KaF(V, m_KA, h_KA): return g_KA * (m_KA**3) * h_KA * (V - E_K)
def I_KaS(V, m_KST, h_KST): return g_KST * (m_KST**3) * h_KST * (V - E_K)
def I_KV(V, m_KV): return g_KV * (m_KV**4) * (V - E_K)
def I_l(V): return g_l * (V - E_l)
