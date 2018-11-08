import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import matplotlib.pyplot as plt
import matplotlib

# Model after paper Synchronization of Golgi and Granule Cell Firing in a Detailed Network Model of the Cerebellar Granule Cell Layer"

# Set up the constants
C_m  = 1 # membrane capacitance (uF/cm^2)
g_NaF = 40 # maximum conductances (mS/cm^2). NB: in paper cells are modelled as spheres, conductances given in nS.
g_CaL = 0.831
g_Kdr = 6.79
g_KA = 0.523
g_KC = 0.573
g_H = 0.172
g_l = 1.167
E_NaF = 55 # reversal potentials (mV)
E_Kdr  = -90
E_CaL  = 80
E_H = -42
E_KA = -90
E_KC = -90
E_l = -65

# auxiliary equations as defined in paper
def exp(V, A, B, C, V_0): return A * np.exp(B * (V - 10 - V_0)) + C
def sig(V, A, B, V_0): return A / (1 + np.exp(B * (V - 10 - V_0)))
def lin(V, A, B, V_0): return A * (V - V_0) / (np.exp(B * (V - 10 - V_0)) - 1)
def KC_alpha(V, A, B, C, con): return A / (1 + C * np.exp(B * (V - 10) / con)) # con stands for Ca2+ concentration
def KC_beta(V, A, B, C, con): return A / (1 + con / (C * np.exp(B * (V - 10))))

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
A_CaL_bh, B_CaL_bh, C_CaL_bh, V0_CaL_bh = [-0.025, -0.05, 0.025, -60]
A_H_am, B_H_am, V0_H_am = [4 * (10**(-3)), -0.0909, -75]
A_H_bm, B_H_bm, V0_H_bm = [4 * (10**(-3)), 0.0909, -75]
A_KA_taum, B_KA_taum, C_KA_taum, V0_KA_taum = [0.41, -1/42.8, 0.167, -43.5]
A_KA_minf, B_KA_minf, V0_KA_minf = [1, -1/19.8, -46.7]
A_KA_hinf, B_KA_hinf, V0_KA_hinf = [1, 1/8.4, -78.8]
A_KC_am, B_KC_am, C_KC_am = [12.5, -0.085, 1.5 * 10**(-3)]
A_KC_bm, B_KC_bm, C_KC_bm = [7.5, -0.077, 150 * 10**(-6)]

# define functions that determine gating dynamics. 'am' is alpha fn for m, 'bh' is beta fn for h etc.
def am_NaF(V): return exp(V, A_NaF_am, B_NaF_am, 0, V0_NaF_am)
def bm_NaF(V): return exp(V, A_NaF_bm, B_NaF_bm, 0, V0_NaF_bm)
def ah_NaF(V): return exp(V, A_NaF_ah, B_NaF_ah, 0, V0_NaF_ah)
def bh_NaF(V): return exp(V, A_NaF_bh, B_NaF_bh, 0, V0_NaF_bh)
def am_Kdr(V): return exp(V, A_Kdr_am, B_Kdr_am, 0, V0_Kdr_am)
def bm_Kdr(V): return exp(V, A_Kdr_bm, B_Kdr_bm, 0, V0_Kdr_bm)
def ah_Kdr(V): return exp(V, A_Kdr_ah, B_Kdr_ah, C_Kdr_ah, V0_Kdr_ah)
def bh_Kdr(V): return sig(V, A_Kdr_bh, B_Kdr_bh, V0_Kdr_bh)
def am_CaL(V): return sig(V, A_CaL_am, B_CaL_am, V0_CaL_am)
def bm_CaL(V): return lin(V, A_CaL_bm, B_CaL_bm, V0_CaL_bm)
def ah_CaL(V): return exp(V, A_CaL_ah, B_CaL_ah, C_CaL_bh, V0_CaL_ah)
def bh_CaL(V): return exp(V, A_CaL_bh, B_CaL_bh, 0, V0_CaL_bh)
def am_H(V):   return exp(V, A_H_am, B_H_am, 0, V0_H_am)
def bm_H(V):    return exp(V, A_H_bm, B_H_bm, 0, V0_H_bm)
def taum_KA(V): return exp(V, A_KA_taum, B_KA_taum, C_KA_taum, V0_KA_taum)
def minf_KA(V): return sig(V, A_KA_minf, B_KA_minf, V0_KA_minf)
def tauh_KA(V): return 10.8 + 0.03*(V - 10) + 1 / (57.9 * np.exp(0.127*(V - 10)) + 0.000134*np.exp(-0.059*(V - 10)))
def hinf_KA(V): return sig(V, A_KA_hinf, B_KA_hinf, V0_KA_hinf)
def am_KC(V, con):   return KC_alpha(V, A_KC_am, B_KC_am, C_KC_am, con)
def bm_KC(V, con):   return KC_beta(V, A_KC_bm, B_KC_bm, C_KC_bm, con)

# define the currents in the model
def I_NaF(V, m, h): return g_NaF * (m**3) * h * (V - E_NaF)
def I_Kdr(V, m, h): return g_Kdr * (m**4) * h * (V - E_Kdr)
def I_CaL(V, m, h): return g_CaL * (m**2) * h * (V - E_CaL)
def I_H(V, m): return g_H * m * (V - E_H)
def I_KA(V, m, h): return g_KA * (m**3) * h * (V - E_KA)
def I_KC(V, m): return g_KC * m * (V - E_KC)
def I_l(V): return g_l * (V - E_l)

# define function that will return time derivatives for integration
def f(t, y):
    # set external current: make sure it is 0 for large times.
    if 10 <= t <= 12:
        I_e = 10
    else:
        I_e = 0
    # set the variables that are to be integrated
    V, m_NaF, h_NaF, m_Kdr, h_Kdr, m_CaL, h_CaL, m_H, m_KA, h_Ka, m_KC, CA_con = y
    # define DV/dt ('_dot' denotes time differentiation)
    V_dot = 1/C_m * (I_e - (I_NaF(V, m_NaF, h_NaF) + I_Kdr(V, m_Kdr, h_Kdr)
            + I_CaL(V, m_CaL, h_CaL) + I_H(V, m_H) + I_KA(V, m_KA, h_KA)
            + I_KC(V, m_KC) + I_l(V)))
    # enter the equations controlling the gating variables.
    m_NaF_dot = am_NaF(V) * (1 - m_NaF) - bm_NaF(V) * m_NaF
    h_NaF_dot = ah_NaF(V) * (1 - h_NaF) - bh_NaF(V) * h_NaF
    m_Kdr_dot = am_Kdr(V) * (1 - m_Kdr) - bm_Kdr(V) * m_Kdr
    h_Kdr_dot = ah_Kdr(V) * (1 - h_Kdr) - bh_Kdr(V) * h_Kdr
    m_CaL_dot = am_CaL(V) * (1 - m_CaL) - bm_CaL(V) * m_CaL
    h_CaL_dot = ah_CaL(V) * (1 - h_CaL) - bh_CaL(V) * h_CaL
    m_H_dot = am_H(V) * (1 - m_H) - bm_H(V) * h_H
    m_KA_dot = (minf_KA(V) - m_KA) / taum_KA(V)
    h_KA_dot = (hinf_KA(V) - h_KA) / tauh_KA(V)
    m_KC_dot = am_KC(V, CA_con) * (1 - m_KC) - bm_KC(V, CA_con) * m_KC
    CA_con_dot = (I_CaL(V, m_CaL, h_CaL) / (2*96,494*0.084*(10**(-6))) ) - (CA_con - 75.5*(10**(-6))) / 10
    return [V_dot, m_NaF_dot, h_NaF_dot, m_Kdr_dot, h_Kdr_dot, m_CaL_dot, h_CaL_dot, m_H_dot, m_KA_dot, h_KA_dot, m_KC_dot, CA_con_dot]

# enter intial values for V, m's, h's etc. (using initialisation m_0 = m_inf(V_0) )
V_0 = -68
m_NaF_0 = 1 / (am_NaF(V_0)  + bm_NaF(V_0))
h_NaF_0 = 1 / (ah_NaF(V_0)  + bh_NaF(V_0))
m_Kdr_0 = 1 / (am_Kdr(V_0)  + bm_Kdr(V_0))
h_Kdr_0 = 1 / (ah_Kdr(V_0)  + bh_Kdr(V_0))
m_CaL_0 = 1 / (am_CaL(V_0)  + bm_CaL(V_0))
h_CaL_0 = 1 / (ah_CaL(V_0)  + bh_CaL(V_0))
m_H_0   = 1 / (am_H(V_0)  + bm_H(V_0))
m_KA_0  = minf_KA(V_0)
h_KA_0  = hinf_KA(V_0)
CA_con_0 = 75.5*(10**(-6))
m_KC_0  = 1 / (am_KC(V_0, CA_con_0)  + bm_KC(V_0, CA_con_0))
y_0 = [V_0, m_NaF_0, h_NaF_0, m_Kdr_0, h_Kdr_0, m_CaL_0, h_CaL_0, m_H_0, m_KA_0, h_KA_0, m_KC_0, CA_con_0]

# create timescale. t_interval is the time interval in which to calculate the solution.
# t_points are the points at which the solution is stored.
t_interval = (0.0, 50.0)
numpoints = 1000
t_points = np.linspace(t_interval[0], t_interval[1], numpoints)

# solve coupled ODEs with scipy's solver
soln = sp_int.solve_ivp(f, t_interval, y_0, 'BDF')

V = soln.y[0, :]

# plot & show solution
plt.figure(1)
plt.plot(soln.t, V, 'b') # a V, t plot
