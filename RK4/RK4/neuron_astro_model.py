import numpy as np
from numba import jit


C = 1
gNa = 40
gK = 35
gLeak = 0.3
ENa = 55
EK = -77
ELeak = -65

c0 = 2.0  # μM
c1 = 0.185
v1 = 6.0  # s^−1
v2 = 0.11  # μM/s
v3 = 2.2  # μM/s
v5 = 0.025  # μM/s
v6 = 0.2  # μM/s
k1 = 0.5  # s^−1
k2 = 1.0  # μM
k3 = 0.1
k4 = 1.1  # μM/s
a2 = 0.14  # μM/s
d1 = 0.13  # μM
d2 = 1.049  # μM
d3 = 0.9434  # μM
d5 = 0.082  # μM
# v4 = 0.5  # μM/s
alphaG = 25.0
betaG = 500.0
alpha = 0.8
IP3s = 7.143  # s
IP3_0 = 0.16  # μM
alpha_G = 25.0
beta_G = 500.0
alpha_Glu = 2
d_Ca = 0.001
d_IP3 = 0.12



@jit(nopython=True, cache=True, nogil=True)
def func_m(m, V):
  return 1000 * (alpha_m(V) * (1 - m) - beta_m(V) * m)

@jit(nopython=True, cache=True, nogil=True)
def func_n(n, V):
  return 1000 * (alpha_n(V) * (1 - n) - beta_n(V) * n)

@jit(nopython=True, cache=True, nogil=True)
def func_h(h, V):
  return 1000 * (alpha_h(V) * (1 - h) - beta_h(V) * h)
   
@jit(nopython=True, cache=True, nogil=True)
def func_V(m, n, h, V, iapp):
  return 1000 * ((gNa * m**3 * h * (ENa - V) + gK * n * (EK - V) + gLeak * (ELeak - V) + iapp)) / C

@jit(nopython=True, cache=True, nogil=True)
def func_V2(m, n, h, V1, V2, iapp, gsyn, Esyn):
  return 1000 * ((gNa * m**3 * h * (ENa - V2) + gK * n * (EK - V2) + gLeak * (ELeak - V2) + iapp + Isyn(gsyn, Esyn, V1, V2))) / C

@jit(nopython=True, cache=True, nogil=True)
def func_Ca(Ca, IP3, z, v4):
  return c1 * v1 * IP3**3 * Ca**3 * z**3 * (c0 / c1 - (1 + 1 / c1) * Ca) / ((IP3 + d1) * (Ca + d5))**3 - v3 * Ca**2 / (k3**2 + Ca**2) + c1 * v2 * (c0 / c1 - (1 + 1 / c1) * Ca) + v5 + v6 * IP3**2 / (k2**2 + IP3**2) - k1 * Ca

@jit(nopython=True, cache=True, nogil=True)
def func_IP3(Ca, IP3, z, v4, G):
  return (IP3_0 - IP3) / IP3s + v4 * (Ca + (1 - alpha) * k4) / (Ca + k4) + alpha_Glu / (1 + np.exp(-(G - 0.25) / 0.01))

@jit(nopython=True, cache=True, nogil=True)
def func_z(Ca, IP3, z, v4):
  return a2 * (d2 * (IP3 + d1) / (IP3 + d3) * (1 - z) - Ca * z) 

@jit(nopython=True, cache=True, nogil=True)
def func_G(G, V):
  return -alpha_G * G + beta_G * (1 / (1 + np.exp(-V / 0.5)))

@jit(nopython=True, cache=True, nogil=True)
def J_channel(Ca, IP3, z):
  return c1 * v1 * IP3**3 * Ca**3 * z**3 * (c0 / c1 - (1 + 1/c1) * Ca) / ((IP3 + d1) * (Ca + d5))**3

@jit(nopython=True, cache=True, nogil=True)
def J_pump(Ca):
  return v3 * Ca**2 / (k3**2 + Ca**2)

@jit(nopython=True, cache=True, nogil=True)
def J_leak(Ca):
  return c1 * v2 * (c0 / c1 - (1 + 1/c1) * Ca)

@jit(nopython=True, cache=True, nogil=True)
def J_in(IP3):
  return v5 + v6 * IP3**2 / (k2**2 + IP3**2)

@jit(nopython=True, cache=True, nogil=True)
def J_out(Ca):
  return k1 * Ca

@jit(nopython=True, cache=True, nogil=True)
def J_PLC(Ca, v4):
  return v4 * (Ca + (1 - alpha) * k4) / (Ca + k4)

@jit(nopython=True, cache=True, nogil=True)
def J_Glu(G):
  return alpha_Glu / (1 + np.exp((0.25 - G) * 100))

@jit(nopython=True, cache=True, nogil=True)
def Isyn(gsyn, Esyn, V1, V2):
   return gsyn * (Esyn - V2) / (1 + np.exp(-1 * V1 / 0.2))

@jit(nopython=True, cache=True, nogil=True)
def alpha_m(V):
  x = V + 35.0
  # Особая точка V = -35: числитель и знаменатель одновременно равны нулю.
  # Используем предел, чтобы не получать ZeroDivisionError при RK4 на сетках параметров.
  if abs(x) < 1e-10:
    return 0.182 * 9.0
  return 0.182 * x / (1.0 - np.exp(-x / 9.0))

@jit(nopython=True, cache=True, nogil=True)
def beta_m(V):
  x = V + 35.0
  if abs(x) < 1e-10:
    return 0.124 * 9.0
  return -0.124 * x / (1.0 - np.exp(x / 9.0))

@jit(nopython=True, cache=True, nogil=True)
def alpha_n(V):
  x = V - 25.0
  if abs(x) < 1e-10:
    return 0.02 * 9.0
  return 0.02 * x / (1.0 - np.exp(-x / 9.0))

@jit(nopython=True, cache=True, nogil=True)
def beta_n(V):
  x = V - 25.0
  if abs(x) < 1e-10:
    return 0.002 * 9.0
  return -0.002 * x / (1.0 - np.exp(x / 9.0))

@jit(nopython=True, cache=True, nogil=True)
def alpha_h(V):
  return 0.25 * np.exp(-1 * (V + 90) / 12)

@jit(nopython=True, cache=True, nogil=True)
def beta_h(V):
  return 0.25 * np.exp((V + 62) / 6) / np.exp((V + 90) / 12)

@jit(nopython=True, cache=True, nogil=True)
def gsyn_(gsyn, Ca, gastro):
  if Ca >= 0.3:
    return gsyn * (1 + gastro * Ca)
  else:
    return gsyn 


__all__ = [
    'C', 'gNa', 'gK', 'gLeak', 'ENa', 'EK', 'ELeak',
    'c0', 'c1', 'v1', 'v2', 'v3', 'v5', 'v6', 'k1', 'k2', 'k3', 'k4',
    'a2', 'd1', 'd2', 'd3', 'd5', 'alphaG', 'betaG', 'alpha', 'IP3s',
    'IP3_0', 'alpha_G', 'beta_G', 'alpha_Glu', 'd_Ca', 'd_IP3',
    'func_m', 'func_n', 'func_h', 'func_V', 'func_V2', 'func_Ca', 'func_IP3',
    'func_z', 'func_G', 'J_channel', 'J_pump', 'J_leak', 'J_in', 'J_out',
    'J_PLC', 'J_Glu', 'Isyn', 'alpha_m', 'beta_m', 'alpha_n', 'beta_n',
    'alpha_h', 'beta_h', 'gsyn_',
]
