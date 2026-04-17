from matplotlib import figure, pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from numba import jit
from tkinter import *
from tkinter import ttk
from tkinter.messagebox import showerror
import scipy



@jit(nopython=True)
def RK41(z, q, vv, mm, nn, hh, iapp_1, iapp_2, gsyn, Esyn, u4, gastro):
    t = np.zeros(z + 1)
    V1 = np.zeros(z + 1)
    m1 = np.zeros(z + 1)
    n1 = np.zeros(z + 1)
    h1 = np.zeros(z + 1)

    V2 = np.zeros(z + 1)
    m2 = np.zeros(z + 1)
    n2 = np.zeros(z + 1)
    h2 = np.zeros(z + 1)

    Ca1 = np.zeros(z + 1)
    IP31 = np.zeros(z + 1)
    z_01 = np.zeros(z + 1)
    G = np.zeros(z + 1)
      
    Box1 = np.zeros((5, z + 1))
    Box2 = np.zeros((5, z + 1))
    Box3 = np.zeros((4, z + 1))
    Box4 = np.zeros((2, z + 1))
    
    t[0] = 0
    V1[0] = V2[0] = vv
    m1[0] = m2[0] = mm
    n1[0] = n2[0] = nn
    h1[0] = h2[0] = hh
    Ca1[0] = 0.07
    IP31[0] = 0.16
    z_01[0] = 0.67
    G[0] = 0.0

    Box1[:, 0] = Box2[:, 0] = [0, vv, mm, nn, hh]
    Box3[:, 0] = [0, 0.07, 0.16, 0.67]
    Box4[:, 0] = [0, 0]
    
    for i in range(1, z + 1):
        Vm1, Vm2 = V1[i-1], V2[i-1]
        mm1, nn1, hh1 = m1[i-1], n1[i-1], h1[i-1]
        mm2, nn2, hh2 = m2[i-1], n2[i-1], h2[i-1]
        gsyn1 = gsyn_(gsyn, Ca1[i - 1], gastro)
        
        mk1 = q * func_m(mm1, Vm1)
        nk1 = q * func_n(nn1, Vm1)
        hk1 = q * func_h(hh1, Vm1)
        Vk1 = q * func_V(mm1, nn1, hh1, Vm1, iapp_1)
        
        mk12 = q * func_m(mm2, Vm2)
        nk12 = q * func_n(nn2, Vm2)
        hk12 = q * func_h(hh2, Vm2)
        Vk12 = q * func_V2(mm2, nn2, hh2, Vm1, Vm2, iapp_2, gsyn1, Esyn)
        
        Cak1 = q * func_Ca(Ca1[i-1], IP31[i-1], z_01[i-1], u4)
        IP3k1 = q * func_IP3(Ca1[i-1], IP31[i-1], z_01[i-1], u4, G[i-1])
        zk1 = q * func_z(Ca1[i-1], IP31[i-1], z_01[i-1], u4)
        Gk1 = q * func_G(G[i-1], V2[i-1])
        
        mk2 = q * func_m(mm1 + mk1/2, Vm1 + Vk1/2)
        nk2 = q * func_n(nn1 + nk1/2, Vm1 + Vk1/2)
        hk2 = q * func_h(hh1 + hk1/2, Vm1 + Vk1/2)
        Vk2 = q * func_V(mm1 + mk1/2, nn1 + nk1/2, hh1 + hk1/2, Vm1 + Vk1/2, iapp_1)
        
        mk22 = q * func_m(mm2 + mk12/2, Vm2 + Vk12/2)
        nk22 = q * func_n(nn2 + nk12/2, Vm2 + Vk12/2)
        hk22 = q * func_h(hh2 + hk12/2, Vm2 + Vk12/2)
        Vk22 = q * func_V2(mm2 + mk12/2, nn2 + nk12/2, hh2 + hk12/2, Vm1 + Vk1/2, Vm2 + Vk12/2, iapp_2, gsyn1, Esyn)
        
        Cak2 = q * func_Ca(Ca1[i-1] + Cak1/2, IP31[i-1] + IP3k1/2, z_01[i-1] + zk1/2, u4)
        IP3k2 = q * func_IP3(Ca1[i-1] + Cak1/2, IP31[i-1] + IP3k1/2, z_01[i-1] + zk1/2, u4, G[i-1] + Gk1/2)
        zk2 = q * func_z(Ca1[i-1] + Cak1/2, IP31[i-1] + IP3k1/2, z_01[i-1] + zk1/2, u4)
        Gk2 = q * func_G(G[i-1] + Gk1/2, V2[i-1] + Vk12/2)
        
        mk3 = q * func_m(mm1 + mk2/2, Vm1 + Vk2/2)
        nk3 = q * func_n(nn1 + nk2/2, Vm1 + Vk2/2)
        hk3 = q * func_h(hh1 + hk2/2, Vm1 + Vk2/2)
        Vk3 = q * func_V(mm1 + mk2/2, nn1 + nk2/2, hh1 + hk2/2, Vm1 + Vk2/2, iapp_1)
        
        mk32 = q * func_m(mm2 + mk22/2, Vm2 + Vk22/2)
        nk32 = q * func_n(nn2 + nk22/2, Vm2 + Vk22/2)
        hk32 = q * func_h(hh2 + hk22/2, Vm2 + Vk22/2)
        Vk32 = q * func_V2(mm2 + mk22/2, nn2 + nk22/2, hh2 + hk22/2, Vm1 + Vk2/2, Vm2 + Vk22/2, iapp_2, gsyn1, Esyn)
        
        Cak3 = q * func_Ca(Ca1[i-1] + Cak2/2, IP31[i-1] + IP3k2/2, z_01[i-1] + zk2/2, u4)
        IP3k3 = q * func_IP3(Ca1[i-1] + Cak2/2, IP31[i-1] + IP3k2/2, z_01[i-1] + zk2/2, u4, G[i-1] + Gk2/2)
        zk3 = q * func_z(Ca1[i-1] + Cak2/2, IP31[i-1] + IP3k2/2, z_01[i-1] + zk2/2, u4)
        Gk3 = q * func_G(G[i-1] + Gk2/2, V2[i-1] + Vk22/2)
        
        mk4 = q * func_m(mm1 + mk3, Vm1 + Vk3)
        nk4 = q * func_n(nn1 + nk3, Vm1 + Vk3)
        hk4 = q * func_h(hh1 + hk3, Vm1 + Vk3)
        Vk4 = q * func_V(mm1 + mk3, nn1 + nk3, hh1 + hk3, Vm1 + Vk3, iapp_1)
        
        mk42 = q * func_m(mm2 + mk32, Vm2 + Vk32)
        nk42 = q * func_n(nn2 + nk32, Vm2 + Vk32)
        hk42 = q * func_h(hh2 + hk32, Vm2 + Vk32)
        Vk42 = q * func_V2(mm2 + mk32, nn2 + nk32, hh2 + hk32, Vm1 + Vk3, Vm2 + Vk32, iapp_2, gsyn1, Esyn)
        
        Cak4 = q * func_Ca(Ca1[i-1] + Cak3, IP31[i-1] + IP3k3, z_01[i-1] + zk3, u4)
        IP3k4 = q * func_IP3(Ca1[i-1] + Cak3, IP31[i-1] + IP3k3, z_01[i-1] + zk3, u4, G[i-1] + Gk3)
        zk4 = q * func_z(Ca1[i-1] + Cak3, IP31[i-1] + IP3k3, z_01[i-1] + zk3, u4)
        Gk4 = q * func_G(G[i-1] + Gk3, V2[i-1] + Vk32)
        
        t[i] = i * q
        
        m1[i] = mm1 + (mk1 + 2*mk2 + 2*mk3 + mk4) / 6
        n1[i] = nn1 + (nk1 + 2*nk2 + 2*nk3 + nk4) / 6
        h1[i] = hh1 + (hk1 + 2*hk2 + 2*hk3 + hk4) / 6
        V1[i] = Vm1 + (Vk1 + 2*Vk2 + 2*Vk3 + Vk4) / 6
        
        m2[i] = mm2 + (mk12 + 2*mk22 + 2*mk32 + mk42) / 6
        n2[i] = nn2 + (nk12 + 2*nk22 + 2*nk32 + nk42) / 6
        h2[i] = hh2 + (hk12 + 2*hk22 + 2*hk32 + hk42) / 6
        V2[i] = Vm2 + (Vk12 + 2*Vk22 + 2*Vk32 + Vk42) / 6
        
        Ca1[i] = Ca1[i-1] + (Cak1 + 2 * Cak2 + 2 * Cak3 + Cak4) / 6
        IP31[i] = IP31[i-1] + (IP3k1 + 2 * IP3k2 + 2 * IP3k3 + IP3k4) / 6
        z_01[i] = z_01[i-1] + (zk1 + 2 * zk2 + 2 * zk3 + zk4) / 6
        G[i] = G[i-1] + (Gk1 + 2 * Gk2 + 2 * Gk3 + Gk4) / 6
        

        Box1[:, i] = [t[i], V1[i], m1[i], n1[i], h1[i]]
        Box2[:, i] = [t[i], V2[i], m2[i], n2[i], h2[i]]
        Box3[:, i] = [t[i], Ca1[i], IP31[i], z_01[i]]
        Box4[:, i] = [t[i], G[i]]

    return Box1, Box2, Box3, Box4

@jit(nopython=True)
def RK4(z, q, vv, mm, nn, hh, iapp):
  t = np.zeros(z + 1)
  V = np.zeros(z + 1)
  m = np.zeros(z + 1)
  n = np.zeros(z + 1)
  h = np.zeros(z + 1)
    
  Box = np.zeros((5, z + 1))
  
  t[0] = 0
  Box[0, 0] = t[0]
  V[0] = vv
  Box[1, 0] = V[0]
  m[0] = mm
  Box[2, 0] = m[0]
  n[0] = nn
  Box[3, 0] = n[0]
  h[0] = hh
  Box[4, 0] = h[0]

  for i in range(1, z + 1):
      
      mk1 = q * func_m(m[i-1], V[i-1])
      nk1 = q * func_n(n[i-1], V[i-1])
      hk1 = q * func_h(h[i-1], V[i-1])
      Vk1 = q * func_V(m[i-1], n[i-1], h[i-1], V[i-1], iapp)
      
      
      mk2 = q * func_m(m[i-1] + mk1/2, V[i-1] + Vk1/2)
      nk2 = q * func_n(n[i-1] + nk1/2, V[i-1] + Vk1/2)
      hk2 = q * func_h(h[i-1] + hk1/2, V[i-1] + Vk1/2)
      Vk2 = q * func_V(m[i-1]+ mk1/2, n[i-1] + nk1/2, h[i-1] + hk1/2, V[i-1] + Vk1/2, iapp)
      
      
      mk3 = q * func_m(m[i-1] + mk2/2, V[i-1] + Vk2/2)
      nk3 = q * func_n(n[i-1] + nk2/2, V[i-1] + Vk2/2)
      hk3 = q * func_h(h[i-1] + hk2/2, V[i-1] + Vk2/2)
      Vk3 = q * func_V(m[i-1] + mk2/2, n[i-1] + nk2/2, h[i-1] + hk2/2, V[i-1] + Vk2/2, iapp)
      
      
      mk4 = q * func_m(m[i-1] + mk3, V[i-1] + Vk3)
      nk4 = q * func_n(n[i-1] + nk3, V[i-1] + Vk3)
      hk4 = q * func_h(h[i-1] + hk3, V[i-1] + Vk3)
      Vk4 = q * func_V(m[i-1] + mk3, n[i-1] + nk3, h[i-1] + hk3, V[i-1] + Vk3, iapp)
        
      t[i] = t[i-1] + q
      Box[0, i] = (t[i])
      
      m[i] = m[i-1] + (mk1 + 2*mk2 + 2*mk3 + mk4) / 6
      n[i] = n[i-1] + (nk1 + 2*nk2 + 2*nk3 + nk4) / 6
      h[i] = h[i-1] + (hk1 + 2*hk2 + 2*hk3 + hk4) / 6
      V[i] = V[i-1] + (Vk1 + 2*Vk2 + 2*Vk3 + Vk4) / 6

      Box[1, i] = (V[i])
      Box[2, i] = (m[i])
      Box[3, i] = (n[i])
      Box[4, i] = (h[i])

  return Box
    
@jit(nopython=True)
def func_m(m, V):
  return 1000 * (alpha_m(V) * (1 - m) - beta_m(V) * m)

@jit(nopython=True)
def func_n(n, V):
  return 1000 * (alpha_n(V) * (1 - n) - beta_n(V) * n)

@jit(nopython=True)
def func_h(h, V):
  return 1000 * (alpha_h(V) * (1 - h) - beta_h(V) * h)
   
@jit(nopython=True)
def func_V(m, n, h, V, iapp):
  return 1000 * ((gNa * m**3 * h * (ENa - V) + gK * n * (EK - V) + gLeak * (ELeak - V) + iapp)) / C

@jit(nopython=True)
def func_V2(m, n, h, V1, V2, iapp, gsyn, Esyn):
  return 1000 * ((gNa * m**3 * h * (ENa - V2) + gK * n * (EK - V2) + gLeak * (ELeak - V2) + iapp + Isyn(gsyn, Esyn, V1, V2))) / C

@jit(nopython=True)
def func_Ca(Ca, IP3, z, v4):
  return c1 * v1 * IP3**3 * Ca**3 * z**3 * (c0 / c1 - (1 + 1 / c1) * Ca) / ((IP3 + d1) * (Ca + d5))**3 - v3 * Ca**2 / (k3**2 + Ca**2) + c1 * v2 * (c0 / c1 - (1 + 1 / c1) * Ca) + v5 + v6 * IP3**2 / (k2**2 + IP3**2) - k1 * Ca

@jit(nopython=True)
def func_IP3(Ca, IP3, z, v4, G):
  return (IP3_0 - IP3) / IP3s + v4 * (Ca + (1 - alpha) * k4) / (Ca + k4) + alpha_Glu / (1 + np.exp(-(G - 0.25) / 0.01))

@jit(nopython=True)
def func_z(Ca, IP3, z, v4):
  return a2 * (d2 * (IP3 + d1) / (IP3 + d3) * (1 - z) - Ca * z) 

@jit(nopython=True)
def func_G(G, V):
  return -alpha_G * G + beta_G * (1 / (1 + np.exp(-V / 0.5)))

@jit(nopython=True)
def J_channel(Ca, IP3, z):
  return c1 * v1 * IP3**3 * Ca**3 * z**3 * (c0 / c1 - (1 + 1/c1) * Ca) / ((IP3 + d1) * (Ca + d5))**3

@jit(nopython=True)
def J_pump(Ca):
  return v3 * Ca**2 / (k3**2 + Ca**2)

@jit(nopython=True)
def J_leak(Ca):
  return c1 * v2 * (c0 / c1 - (1 + 1/c1) * Ca)

@jit(nopython=True)
def J_in(IP3):
  return v5 + v6 * IP3**2 / (k2**2 + IP3**2)

@jit(nopython=True)
def J_out(Ca):
  return k1 * Ca

@jit(nopython=True)
def J_PLC(Ca, v4):
  return v4 * (Ca + (1 - alpha) * k4) / (Ca + k4)

@jit(nopython=True)
def J_Glu(G):
  return alpha_Glu / (1 + np.exp((0.25 - G) * 100))

@jit(nopython=True)
def Isyn(gsyn, Esyn, V1, V2):
   return gsyn * (Esyn - V2) / (1 + np.exp(-1 * V1 / 0.2))

@jit(nopython=True)
def alpha_m(V):
  return 0.182 * (V + 35) / (1 - np.exp(-1 * (V + 35) / 9))

@jit(nopython=True)
def beta_m(V):
  return -0.124 * (V + 35) / (1 - np.exp((V + 35) / 9))

@jit(nopython=True)
def alpha_n(V):
  return 0.02 * (V - 25) / (1 - np.exp(-1 * (V - 25) / 9))

@jit(nopython=True)
def beta_n(V):
  return -0.002 * (V - 25) / (1 - np.exp((V - 25) / 9))

@jit(nopython=True)
def alpha_h(V):
  return 0.25 * np.exp(-1 * (V + 90) / 12)

@jit(nopython=True)
def beta_h(V):
  return 0.25 * np.exp((V + 62) / 6) / np.exp((V + 90) / 12)

@jit(nopython=True)
def gsyn_(gsyn, Ca, gastro):
  if Ca >= 0.3:
    return gsyn * (1 + gastro * Ca)
  else:
    return gsyn 

def Maximum(Box):
  Cage = [[], []]
  Cell = [[], []]
  for i in range(0, len(Box[1]) - 1):
    if ((Box[1][i] > Box[1][i + 1]) and (Box[1][i] > Box[1][i - 1]) and Box[1][i] > -10):
      Cage[0].append(Box[0][i])
      Cage[1].append(Box[1][i])
  for i in range(len(Cage[1]) - 1):
    if (Cage[0][i + 1] - Cage[0][i]) > 0.002: 
      Cell[0].append(Cage[0][i])
      Cell[1].append(Cage[1][i])
  return Cell

def Period(Cage):
  Container = [[], []]
  for i in range(0, len(Cage[0]) - 1):
    Container[0].append(Cage[0][i])
    Container[1].append(Cage[0][i + 1] - Cage[0][i])
  return Container

def Frequency(Cage):
  Container = [[], []]
  for i in range(0, len(Cage[0]) - 1):
    Container[0].append(Cage[0][i])
    Container[1].append(1 / (Cage[0][i + 1] - Cage[0][i]))
  return Container

def Middle_Frequency(Container):
  sum = 0
  for i in Container[1]:
    sum += i
  if len(Container[1]) != 0:
    ssum = sum / len(Container[1])
    return 1 / ssum
  else:
    return 0.0
  

def upload_date():
  file = open('initial_conditions.txt', 'r')
  data = []

  while True:
    line = file.readline()  
    if not line:
      break
    data.append(float(line))

  file.close()

  vv = data[0]
  mm = data[1]
  nn = data[2]
  hh = data[3]
  
  return vv, mm, nn, hh

def upload_date1():
  file = open('initial_conditions_Ca.txt', 'r')
  data = []

  while True:
    line = file.readline()  
    if not line:
      break
    data.append(float(line))

  file.close()

  Ca = data[0]
  IP3 = data[1]
  zz = data[2]
  
  return Ca, IP3,zz


def load_date(Box):
  file4 = open('initial_conditions.txt', 'w')
  
  p = str(Box[1][-1])
  j = str(Box[2][-1])
  l = str(Box[3][-1])
  w = str(Box[4][-1])
  
  file4.write(p + '\n')
  file4.write(j + '\n')
  file4.write(l + '\n')
  file4.write(w)
  
  file4.close()

def load_date1(vv, mm, nn, hh):
  file4 = open('initial_conditions.txt', 'w')
  
  p = str(vv)
  j = str(mm)
  l = str(nn)
  w = str(hh)
  
  file4.write(p + '\n')
  file4.write(j + '\n')
  file4.write(l + '\n')
  file4.write(w)
  
  file4.close()
  
def load_date2(Box):
  file4 = open('initial_conditions_Ca.txt', 'w')

  o = str(Box[0][-1])
  l = str(Box[1][-1])
  s = str(Box[2][-1])
  
  file4.write(o + '\n')
  file4.write(l + '\n')
  file4.write(s)
  
  file4.close()

def integrate_date(vv = -60, mm = 0.09, nn = 0.0009, hh = 0.3, z = 10000000, q = 0.000001, iapp = 0.0):
  Box = RK4(z, q, vv, mm, nn, hh, iapp)
  load_date(Box)
  return Box

def integrate_date1(z = 100000, q = 0.0001, iapp_1 = 0.0, iapp_2 = 0.0, gsyn = 0.1, Esyn = 0, u4 = 0.5, gastro = 2.0):
  vv, mm, nn, hh = upload_date()
  # Ca, IP3, z_0 = upload_date1()
  Box1, Box2, Box3, Box4 = RK41(z, q, vv, mm, nn, hh, iapp_1, iapp_2, gsyn, Esyn, u4, gastro)
  load_date(Box1)
  #load_date2(Box3)
  return Box1, Box2, Box3, Box4
  
def data_cropping(Box):
  Cage1 = [0] * int(0.75*len(Box[0]))
  Cage2 = [0] * int(0.75*len(Box[1]))
  Cage3 = [0] * int(0.75*len(Box[2]))
  Cage4 = [0] * int(0.75*len(Box[3]))
  Cage5 = [0] * int(0.75*len(Box[4]))
  Cage1 = Box[0, int(0.25*len(Box[0])):len(Box[0])]
  Cage2 = Box[1, int(0.25*len(Box[1])):len(Box[1])]
  Cage3 = Box[2, int(0.25*len(Box[2])):len(Box[2])]
  Cage4 = Box[3, int(0.25*len(Box[3])):len(Box[3])]
  Cage5 = Box[4, int(0.25*len(Box[4])):len(Box[4])]
  Cage = [Cage1, Cage2, Cage3, Cage4, Cage5]
  return Cage

def check_data():
  load_date1(float_V.get(), float_m.get(), float_n.get(), float_h.get())
  return upload_date()

def integrate():
  iapp = float(get_Iapp())
  
  v2, m2, n2, h2 = check_data()

  Boxx = integrate_date(v2, m2, n2, h2, z= 30000000, q= 0.000001, iapp = iapp)
  
  label_param5.config(text= f"V[0] = {Boxx[1, -1]:.6f}")

  label_param6.config(text= f"m[0] = {Boxx[2, -1]:.6f}")

  label_param7.config(text= f"n[0] = {Boxx[3, -1]:.6f}") 

  label_param8.config(text= f"h[0] = {Boxx[4, -1]:.6f}")
  
  float_V.set(np.round(Boxx[1, -1], 6))
  float_m.set(np.round(Boxx[2, -1], 6))
  float_n.set(np.round(Boxx[3, -1], 6))
  float_h.set(np.round(Boxx[4, -1], 6))

  fig1 = draw1(Boxx)
  fig2 = draw2(Boxx)
  fig3 = draw3(Boxx)
  fig4 = draw4(Boxx)

  canvas1 = FigureCanvasTkAgg(fig1, master=frame1) 
  canvas1.get_tk_widget().place(relx=0.2, rely=0.0)
  canvas1.draw()
  
  canvas2 = FigureCanvasTkAgg(fig2, master=frame1) 
  canvas2.get_tk_widget().place(relx=0.6, rely=0.0)
  canvas2.draw()
  
  canvas3 = FigureCanvasTkAgg(fig3, master=frame1) 
  canvas3.get_tk_widget().place(relx=0.2, rely=0.45)
  canvas3.draw()
  
  canvas4 = FigureCanvasTkAgg(fig4, master=frame1) 
  canvas4.get_tk_widget().place(relx=0.6, rely=0.45)
  canvas4.draw()
 
  plt.close(fig1)
  plt.close(fig2)
  plt.close(fig3)
  plt.close(fig4)
  
  btn1 = ttk.Button(frame1, text="Сохранить 1 график", command=lambda: save_pic1(fig1))
  btn1.place(x=20, y=450)
  
  btn2 = ttk.Button(frame1, text="Сохранить 2 график", command=lambda: save_pic2(fig2))
  btn2.place(x=20, y=500)
  
  btn3 = ttk.Button(frame1, text="Сохранить 3 график", command=lambda: save_pic3(fig3))
  btn3.place(x=20, y=550)
  
  btn4 = ttk.Button(frame1, text="Сохранить 4 график", command=lambda: save_pic4(fig4))
  btn4.place(x=20, y=600)
  
def save_pic1(fig):
  fig.savefig('_V(t)_.png')
  
def save_pic2(fig):
  fig.savefig('_V(h)_.png')
  
def save_pic3(fig):
  fig.savefig('_mnh(t)_.png')
  
def save_pic4(fig):
  fig.savefig('_Freq_.png')
  
def save_pic5(fig):
  fig.savefig('_Iapp_.png')
  
def save_pic6(fig):
  fig.savefig('_V(t)1_.png')
  
def save_pic7(fig):
  fig.savefig('_V(t)2_.png')
  
def save_pic8(fig):
  fig.savefig('qwerty.png')
  
def save_pic10(fig):
  fig.savefig('_Ca(t)_.png')

def get_Iapp() -> float:
    fiapp = float(float_iapp.get())
    return fiapp

def draw1(Box):
  fig1 = plt.figure(figsize=(7.5, 4.5))
  plt.plot(Box[0], Box[1])
  plt.ylim(-80, 20)
  plt.xlim(0, 1)
  plt.xlabel('t')
  plt.ylabel('V, mV')
  plt.grid(True)
  return fig1

def draw2(Box):
  fig2 = plt.figure(figsize=(7.5, 4.5))
  plt.plot(Box[4], Box[1])
  plt.ylim(-80, 20)
  plt.xlim(0, 1)
  plt.xlabel('h')
  plt.ylabel('V, mV')
  plt.grid(True)
  return fig2

def draw3(Box):
  fig3 = plt.figure(figsize=(7.5, 4.5))
  plt.plot(Box[0], Box[2], 'r-', label='m')  # Красный цвет, сплошная линия
  plt.plot(Box[0], Box[3], 'g-', label='n')  # Зеленый цвет, сплошная линия
  plt.plot(Box[0], Box[4], 'b-', label='h')  # Синий цвет, сплошная линия
  plt.ylim(0, 1)
  plt.xlim(0, 1)
  plt.xlabel('t')
  plt.ylabel('m, n, h')
  plt.legend(loc= 'upper right')  # Добавление легенды
  plt.grid(True)
  return fig3

def draw4(Box):
  Box = data_cropping(Box)
  Cage = Maximum(Box)
  Cell = Frequency(Cage)
  fig4 = plt.figure(figsize=(7.5, 4.5))
  plt.plot(Cell[0], Cell[1])
  plt.ylim(0, 100)
  plt.xlabel('t')
  plt.ylabel('Frequency, s^-1')
  plt.grid(True)
  return fig4


def integrate1():
  iapp_min = float_iapp_min.get()
  iapp_delta = float_iapp_delta.get()
  iapp_max = float_iapp_max.get()
  if (iapp_min < iapp_max):
    T = np.arange(iapp_min, iapp_max, iapp_delta)
    Box = direct_passage(T)
    Cage = return_passage(T)
  
    fig = plt.figure(figsize=(10, 6))
    plt.plot(T, Box, 'r-')  # Красный цвет, сплошная линия
    plt.plot(T, Cage[::-1], 'b-')
    plt.xticks(np.arange(0.0, 2.1, 0.2))
    plt.ylim(0, 35)
    plt.xlim(0, 2)
    plt.xlabel('iapp')
    plt.ylabel('Средняя частота, s^-1')
  
    canvas5 = FigureCanvasTkAgg(fig, master=frame2) 
    canvas5.get_tk_widget().place(relx=0.3, rely=0.0)
    canvas5.draw()
  
    button_integrate1 = ttk.Button(frame2, text="Сохранить", command=lambda: save_pic5(fig))
    button_integrate1.place(x=20, y=550)
  
    plt.close(fig)
  else:
    showerror(title= "Ошибка", message= "Введите корректные данные")

def direct_passage(T):
  Cell = []
  for i in T:
    vv, mm, nn, hh = upload_date()
    Box = integrate_date(vv, mm, nn, hh, z= 30000000, q= 0.000001, iapp= i)
    Box = data_cropping(Box)
    Cage = Maximum(Box)
    Container = Period(Cage)
    middle = Middle_Frequency(Container)
    Cell.append(middle)
  return Cell

def return_passage(T):
  Cell = []
  for i in T[::-1]:
    vv, mm, nn, hh = upload_date()
    Box = integrate_date(vv, mm, nn, hh, z= 30000000, q= 0.000001, iapp= i)
    Box = data_cropping(Box)
    Cage = Maximum(Box)
    Container = Period(Cage)
    middle = Middle_Frequency(Container)
    Cell.append(middle)
  return Cell

def integrate2():
  iapp_1 = float_iapp_1.get()
  iapp_2 = float_iapp_2.get()
  gsyn = float_gsyn.get()
  Esyn = int_Esyn.get()
  
  Box1, Box2 = integrate_date1(z= 10000000, q= 0.000001, iapp_1 = iapp_1, iapp_2 = iapp_2, gsyn= gsyn, Esyn= Esyn)
  
  Boxx1 = data_cropping(Box1)
  Boxx2 = data_cropping(Box2)
  
  Cage1 = Maximum(Boxx1)
  Cage2 = Maximum(Boxx2)
  
  Cell1 = Frequency(Cage1)
  Cell2 = Frequency(Cage2)
  
  fig1 = plt.figure(figsize=(8, 6))
  plt.plot(Box1[0], Box1[1], 'r-', label= "1")
  plt.plot(Box2[0], Box2[1], 'b-', label= "2")
  plt.ylim(-80, 20)
  plt.xlim(0, 1)
  plt.ylabel("V1, V2, mV")
  plt.xlabel("t")
  plt.legend(loc= "upper right")
  plt.grid(True)
  
  canvas6 = FigureCanvasTkAgg(fig1, master=frame3) 
  canvas6.get_tk_widget().place(relx=0.15, rely=0.0)
  canvas6.draw()
  
  button_integrate2 = ttk.Button(frame3, text="Сохранить 1 график", command=lambda: save_pic6(fig1))
  button_integrate2.place(x=20, y=350)
  
  fig2 = plt.figure(figsize=(8, 6))
  plt.plot(Cell1[0], Cell1[1], 'r-', label= "1")
  plt.plot(Cell2[0], Cell2[1], 'b-', label= "2")
  plt.ylim(0, 100)
  plt.xlim(10, 13)
  plt.ylabel("Frequency1, Frequency2, s^-1")
  plt.xlabel("t")
  plt.legend(loc= "upper right")
  plt.grid(True)
  
  canvas7 = FigureCanvasTkAgg(fig2, master=frame3) 
  canvas7.get_tk_widget().place(relx=0.55, rely=0.0)
  canvas7.draw()
  
  button_integrate3 = ttk.Button(frame3, text="Сохранить 2 график", command=lambda: save_pic7(fig2))
  button_integrate3.place(x=20, y=400)
  
  plt.close(fig1)
  plt.close(fig2)

def validate_float_V():
    try:
        value = float(float_V.get())
        if value < -80 or value > 20:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от -80 до 20 для V")
        
def validate_float_m():
    try:
        value = float(float_m.get())
        if value < 0 or value > 1:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от 0 до 1 для m")

def validate_float_n():
    try:
        value = float(float_n.get())
        if value < 0 or value > 1:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от 0 до 1 для n")

def validate_float_h():
    try:
        value = float(float_h.get())
        if value < 0 or value > 1:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от 0 до 1 для h")
        
def validate_float_gsyn():
    try:
        value = float(float_gsyn.get())
        if value < 0 or value > 0.2:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от 0 до 0.2 для g_syn")
        
def validate_float_gastro():
    try:
        value = float(float_gastro.get())
        if value < -4 or value > 4:
            raise ValueError
    except ValueError:
        showerror("Ошибка", "Пожалуйста, введите число от -4 до 4 для g_astro")
        

def plot_image_from_data(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_coords = np.unique([item[0] for item in data])
    y_coords = np.unique([item[1] for item in data])
    values = np.array([item[2] for item in data]).reshape(len(y_coords), len(x_coords))


    im = ax.imshow(values, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower', cmap='binary_r')
    ax.set_xlabel('gsyn')
    ax.set_aspect('auto')
    ax.set_ylabel('delta Iapp')
    yticks = ax.get_yticks()
    new_ytick_labels = [round(y - 1.5, 1) for y in yticks]
    ax.set_yticklabels(new_ytick_labels)
    fig.colorbar(im, ax=ax, label='Color')

    return fig

def integrate3():
  iapp_1 = 1.5
  Esyn = int_Esyn.get()

  WWW = []
  for iapp_2 in np.arange(1.2, 1.814, 0.014):
    for gsyn in np.arange(0, 0.204, 0.004): 
      Box1, Box2 = integrate_date1(z= 4000000, q= 0.000001, iapp_1 = iapp_1, iapp_2 = iapp_2, gsyn= gsyn, Esyn= Esyn)
      Boxx1 = data_cropping(Box1)
      Boxx2 = data_cropping(Box2)
  
      Cage1 = Maximum(Boxx1)
      Cage2 = Maximum(Boxx2)
      
      Cell1 = Period(Cage1)
      Cell2 = Period(Cage2)
      
      q1 = Middle_Frequency(Cell1)
      q2 = Middle_Frequency(Cell2)
      if abs(q2 - q1) < 0.3:
          WWW.append([gsyn, iapp_2, 1])
      else:
          WWW.append([gsyn, iapp_2, 0]) 
          
  fig, ax = plt.subplots(figsize=(8, 6))
    
  x_coords = np.unique([item[0] for item in WWW])
  y_coords = np.unique([item[1] for item in WWW])
  values = np.array([item[2] for item in WWW]).reshape(len(y_coords), len(x_coords))


  im = ax.imshow(values, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower', cmap='binary_r')
  ax.set_xlabel('gsyn')
  ax.set_aspect('auto')
  ax.set_ylabel('delta Iapp')
  yticks = ax.get_yticks()
  new_ytick_labels = [round(y - 1.5, 1) for y in yticks]
  ax.set_yticklabels(new_ytick_labels)
  fig.colorbar(im, ax=ax, label='Color')
  
  button_integrate9 = ttk.Button(frame4, text="Сохранить график", command=lambda: save_pic8(fig))
  button_integrate9.place(x=20, y=400)
          
  canvas7 = FigureCanvasTkAgg(fig, master=frame4) 
  canvas7.get_tk_widget().place(relx=0.15, rely=0.0)
  canvas7.draw()
  
  plt.close(fig)
  
def integrate4():
  iapp_1 = 1.5
  gastro = float_gastro.get()
  Esyn = int_Esyn.get()
  v4  = float_u4.get()
  WWW = []
  
  for iapp_2 in np.arange(1.2, 1.805, 0.005):
    for gsyn in np.arange(0, 0.20125, 0.00125): 
      Box1, Box2, Box3, Box4 = integrate_date1(z= 300000, q= 0.0001, iapp_1 = iapp_1, iapp_2 = iapp_2, gsyn= gsyn, Esyn= Esyn, u4= v4, gastro= gastro)
      
      Boxx1 = data_cropping(Box1)
      Boxx2 = data_cropping(Box2)
  
      Cage1 = Maximum(Boxx1)
      Cage2 = Maximum(Boxx2)
      
      Cell1 = Frequency(Cage1)
      Cell2 = Frequency(Cage2)
      
      time_1 = np.array(Cell1[0])
      omega_1 = np.array(Cell1[1])
      time_2 = np.array(Cell2[0])
      omega_2 = np.array(Cell2[1])
      time_V = np.array(Boxx1[0])

      time_11 = np.linspace(time_1[0], time_1[-1], len(time_2))

      interp_func = scipy.interpolate.interp1d(time_1, omega_1, kind='linear', fill_value="extrapolate")
      omega_11 = interp_func(time_11)

      Delta_omega = omega_2 - omega_11

      ind1 = np.abs(Delta_omega) <= 0.3

      t1 = np.diff(time_11[ind1])

      it1 = np.where(t1 < 1)[0]

      T_12 = np.sum(t1[it1])

      abs_t_sync_12 = T_12 / (time_V[-1] - time_V[0])
      
      WWW.append(([gsyn, iapp_2, abs_t_sync_12]))
      
     
  fig, ax = plt.subplots(figsize=(13, 10))
  x_coords = np.unique([item[0] for item in WWW])
  y_coords = np.unique([item[1] for item in WWW])
  values = np.array([item[2] for item in WWW]).reshape(len(y_coords), len(x_coords))
  

  im = ax.imshow(values, extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()], origin='lower', cmap = 'hot')
  ax.set_xlabel('gsyn')
  ax.set_aspect('auto')
  ax.set_ylabel('Δ Iapp')
  yticks = ax.get_yticks()
  new_ytick_labels = [round(y - 1.5, 1) for y in yticks]
  ax.set_yticklabels(new_ytick_labels)
  cbar = fig.colorbar(im, ax=ax, label='Color')
  cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
  button_integrate9 = ttk.Button(frame5, text="Сохранить график", command=lambda: save_pic8(fig))
  button_integrate9.place(x=20, y=430)
          
  canvas7 = FigureCanvasTkAgg(fig, master=frame5) 
  canvas7.get_tk_widget().place(relx=0.23, rely=0.0)
  canvas7.draw()
  
  plt.close(fig)

def integrate5():
  iapp_1 = float_iapp_1.get()
  iapp_2 = float_iapp_2.get()
  gsyn = float_gsyn.get()
  gastro = float_gastro.get()
  Esyn = int_Esyn.get()
  v4  = float_u4.get()
  z= 6000000
  q= 0.00001
  Box1, Box2, Box3, Box4 = integrate_date1(z, q, iapp_1 = iapp_1, iapp_2 = iapp_2, gsyn= gsyn, Esyn= Esyn, u4= v4, gastro= gastro)

  Boxx1 = data_cropping(Box1)
  Boxx2 = data_cropping(Box2)
  
  Cage1 = Maximum(Boxx1)
  Cage2 = Maximum(Boxx2)
  
  Cell1 = Frequency(Cage1)
  Cell2 = Frequency(Cage2)
  
  fig1 = plt.figure(figsize=(16, 4))
  plt.plot(Box3[0], Box3[1], 'r-', label= 'Ca')
  # plt.plot(Box3[0], Box3[2], 'b-', label= 'IP3')
  plt.ylabel("Ca")
  plt.xlabel("t")
  plt.xlim(22, 40)
  plt.axhline(y=0.3, color='gray', linestyle='--')
  plt.legend()
  plt.axvspan(28.1, 32.5, color='yellow', alpha=0.3)
  plt.axvspan(32.5, 36.8, color='pink', alpha=0.3)
  # plt.grid(True)
  
  # fig2 = plt.figure(figsize=(7.5, 4.5))
  # plt.plot(Box4[0], Box4[1], 'b-', label= 'G')
  # plt.ylabel("G")
  # plt.xlabel("t")
  # plt.ylim(0, 0.7)
  # plt.xlim(0, 2)
  # plt.legend()
  # plt.grid(True)
  
  # fig3 = plt.figure(figsize=(7.5, 4.5))
  # plt.plot(Box1[0], Box1[1], 'b-')
  # plt.plot(Box2[0], Box2[1], 'r-')
  # plt.ylabel("V1, V2")
  # plt.xlabel("t")
  # plt.ylim(-80, 20)
  # plt.xlim(0, 1)
  # plt.grid(True)
  
  fig4 = plt.figure(figsize=(16, 4))
  plt.plot(Cell1[0], Cell1[1], 'b-', label= 'v1')
  plt.plot(Cell2[0], Cell2[1], 'r-', label= 'v2')
  plt.ylim(25, 35)
  plt.xlim(22, 40)
  plt.ylabel("v1, v2")
  plt.xlabel("t")
  plt.legend()
  plt.axvspan(28.1, 32.5, color='yellow', alpha=0.3)
  plt.axvspan(32.5, 36.8, color='pink', alpha=0.3)
  # plt.grid(True)
  
  canvas6 = FigureCanvasTkAgg(fig1, master=frame6) 
  canvas6.get_tk_widget().place(relx=0.22, rely=0.0)
  canvas6.draw()
  
  # canvas7 = FigureCanvasTkAgg(fig2, master=frame6) 
  # canvas7.get_tk_widget().place(relx=0.55, rely=0.0)
  # canvas7.draw()
  
  # canvas8 = FigureCanvasTkAgg(fig3, master=frame6) 
  # canvas8.get_tk_widget().place(relx=0.15, rely=0.55)
  # canvas8.draw()
  
  canvas9 = FigureCanvasTkAgg(fig4, master=frame6) 
  canvas9.get_tk_widget().place(relx=0.22, rely=0.41)
  canvas9.draw()
  
  button_integrate11 = ttk.Button(frame6, text="Сохранить график", command=lambda: save_pic10(fig1))
  button_integrate11.place(x=20, y=600)
  
  plt.close(fig1)
  # plt.close(fig2)
  # plt.close(fig3)
  plt.close(fig4)

    
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

t = np.linspace(0, 100, 1000000)

v0, m0, n0, h0 = upload_date()
Ca0, IP30, z_0 = 0.07, 0.16, 0.67

#создание окна
root = Tk()
root.title("Динамика нейрона")
root.geometry("1920x1080")
# root.minsize(1920,1080)

root.state('zoomed')

style = ttk.Style()
style.configure(".", background= "white")

#создание вкладок
notebook = ttk.Notebook()
notebook.pack(expand=True, fill=BOTH)

frame1 = ttk.Frame(notebook)
frame2 = ttk.Frame(notebook)
frame3 = ttk.Frame(notebook)
frame4 = ttk.Frame(notebook)
frame5 = ttk.Frame(notebook)
frame6 = ttk.Frame(notebook)
 
frame1.pack(fill=BOTH, expand=True)
frame2.pack(fill=BOTH, expand=True)
frame3.pack(fill=BOTH, expand=True)
frame4.pack(fill=BOTH, expand=True)
frame5.pack(fill=BOTH, expand=True)
frame6.pack(fill=BOTH, expand=True)

notebook.add(frame1, text="1")
notebook.add(frame2, text="2")
notebook.add(frame3, text="3")
notebook.add(frame4, text="4")
notebook.add(frame5, text="5")
notebook.add(frame6, text="6")

#создание переменных для полей ввода
float_iapp = DoubleVar(value = 0.7)

float_iapp_min = DoubleVar(value = 0.75)
float_iapp_delta = DoubleVar(value = 0.05)
float_iapp_max = DoubleVar(value = 1.1)

float_iapp_1 = DoubleVar(value = 1.5)
float_iapp_2 = DoubleVar(value = 1.2)

float_V = DoubleVar(value = np.round(v0, 6))
float_m = DoubleVar(value = np.round(m0, 6))
float_n = DoubleVar(value = np.round(n0, 6))
float_h = DoubleVar(value = np.round(h0, 6))

float_gsyn = DoubleVar(value= 0.1)

int_Esyn = IntVar(value= 0)

int_test = IntVar(value=0)

float_iapp_delta2 = DoubleVar(value = 0.3)

float_u4 = DoubleVar(value= 0.5)

float_gastro = DoubleVar(value= 2.0)

#текст

label_param = ttk.Label(frame1, text="Параметр Iapp:", background="white")
label_param.place(x= 20, y=120)

label1 = ttk.Label(frame1, text="V:", background="white")
label1.place(x= 20, y=300)

label2 = ttk.Label(frame1, text="m:", background="white")
label2.place(x= 20, y=325)

label3 = ttk.Label(frame1, text="n:", background="white")
label3.place(x= 20, y=350)

label4 = ttk.Label(frame1, text="h:", background="white")
label4.place(x= 20, y=375)

label_param4 = ttk.Label(frame1, text="Начальные условия:", background="white")
label_param4.place(x= 20, y=650)

label_param5 = ttk.Label(frame1, text=f"V[0] = {v0:.6f}", background="white")
label_param5.place(x= 20, y=675)

label_param6 = ttk.Label(frame1, text=f"m[0] = {m0:.6f}", background="white")
label_param6.place(x= 20, y=700)

label_param7 = ttk.Label(frame1, text=f"n[0] = {n0:.6f}", background="white")
label_param7.place(x= 20, y=725)

label_param8 = ttk.Label(frame1, text=f"h[0] = {h0:.6f}", background="white")
label_param8.place(x= 20, y=750)

#поля для ввода значений
entry_param = ttk.Entry(frame1, textvariable=float_iapp)
entry_param.place(x= 115, y= 120)

entry1 = ttk.Entry(frame1, textvariable=float_V, validate="focusout", validatecommand=validate_float_V)
entry1.place(x= 40, y= 300)

entry2 = ttk.Entry(frame1, textvariable=float_m, validate="focusout", validatecommand=validate_float_m)
entry2.place(x= 40, y= 325)

entry3 = ttk.Entry(frame1, textvariable=float_n, validate="focusout", validatecommand=validate_float_n)
entry3.place(x= 40, y= 350)

entry4 = ttk.Entry(frame1, textvariable=float_h, validate="focusout", validatecommand=validate_float_h)
entry4.place(x= 40, y= 375)

#создание кнопок
#в command вводишь для выполнения нужной функции(например, функция отрисовки графика)
button_integrate = ttk.Button(frame1, text="Запустить интегрирование", command=integrate)
button_integrate.place(x=20, y=150)

label_param1 = ttk.Label(frame2, text="Минимальный Iapp:", background="white")
label_param1.place(x= 20, y=60)

label_param2 = ttk.Label(frame2, text="Дельта Iapp:", background="white")
label_param2.place(x= 20, y=100)

label_param3 = ttk.Label(frame2, text="Максимальный Iapp:", background="white")
label_param3.place(x= 20, y=140)

entry_param1 = ttk.Entry(frame2, textvariable=float_iapp_min)
entry_param1.place(x= 145, y= 60)

entry_param2 = ttk.Entry(frame2, textvariable=float_iapp_delta)
entry_param2.place(x= 145, y= 100)

entry_param3 = ttk.Entry(frame2, textvariable=float_iapp_max)
entry_param3.place(x= 145, y= 140)

button_integrate1 = ttk.Button(frame2, text="Запустить интегрирование", command=integrate1)
button_integrate1.place(x=20, y=500)

option9 = ttk.Radiobutton(frame2, text="Метод Эйлера 1-го порядка с фиксированным шагом", variable=int_test, value=1)
option9.place(x= 20, y=200)

option10 = ttk.Radiobutton(frame2, text="Метод Рунге-Кутта 4-го порядка с фиксированным шагом", variable=int_test, value=2)
option10.place(x= 20, y=250)

option11 = ttk.Radiobutton(frame2, text="Метод Рунге-Кутта-Фельберга 4-5-го порядка с адаптивным шагом", variable=int_test, value=3)
option11.place(x= 20, y=300)

option12 = ttk.Radiobutton(frame2, text="Метод Гира для жёстких систем с адаптивным шагом", variable=int_test, value=4)
option12.place(x= 20, y=350)

label_param9 = ttk.Label(frame3, text="Iapp_1:", background="white")
label_param9.place(x= 20, y=60)

label_param10 = ttk.Label(frame3, text="Iapp_2:", background="white")
label_param10.place(x= 20, y=100)

label_param11 = ttk.Label(frame3, text="g_syn:", background="white")
label_param11.place(x= 20, y=140)

entry_param4 = ttk.Entry(frame3, textvariable=float_iapp_1)
entry_param4.place(x= 65, y= 60)

entry_param5 = ttk.Entry(frame3, textvariable=float_iapp_2)
entry_param5.place(x= 65, y= 100)

entry_param6 = ttk.Entry(frame3, textvariable=float_gsyn, validate="focusout", validatecommand=validate_float_gsyn)
entry_param6.place(x= 65, y= 140)

option1 = ttk.Radiobutton(frame3, text="Возбуждающая связь (E_syn = 0)", variable=int_Esyn, value=0)
option1.place(x= 20, y=180)

option2 = ttk.Radiobutton(frame3, text="Тормозная связь (E_syn = -90)", variable=int_Esyn, value=-90)
option2.place(x= 20, y=200)

button_integrate2 = ttk.Button(frame3, text="Запустить интегрирование", command=integrate2)
button_integrate2.place(x=20, y=250)

button_integrate3 = ttk.Button(frame4, text="Запустить интегрирование", command=integrate3)
button_integrate3.place(x=20, y=250)

option3 = ttk.Radiobutton(frame4, text="Возбуждающая связь (E_syn = 0)", variable=int_Esyn, value=0)
option3.place(x= 20, y=180)

option4 = ttk.Radiobutton(frame4, text="Тормозная связь (E_syn = -90)", variable=int_Esyn, value=-90)
option4.place(x= 20, y=200)

button_integrate4 = ttk.Button(frame5, text="Запустить интегрирование", command=integrate4)
button_integrate4.place(x=20, y=380)

label_param15 = ttk.Label(frame5, text="g_astro:", background="white")
label_param15.place(x= 20, y=50)

entry_param10 = ttk.Entry(frame5, textvariable=float_gastro, validate="focusout", validatecommand=validate_float_gastro)
entry_param10.place(x= 65, y=50)

option7 = ttk.Radiobutton(frame5, text="Возбуждающая связь (E_syn = 0)", variable=int_Esyn, value=0)
option7.place(x= 20, y=80)

option8 = ttk.Radiobutton(frame5, text="Тормозная связь (E_syn = -90)", variable=int_Esyn, value=-90)
option8.place(x= 20, y=100)

label_param16 = ttk.Label(frame5, text="v4:", background="white")
label_param16.place(x= 20, y=130)

entry_param11 = ttk.Entry(frame5, textvariable=float_u4)
entry_param11.place(x= 65, y= 130)

option13 = ttk.Radiobutton(frame5, text="Метод Эйлера 1-го порядка с фиксированным шагом", variable=int_test, value=1)
option13.place(x= 20, y=180)

option14 = ttk.Radiobutton(frame5, text="Метод Рунге-Кутта 4-го порядка с фиксированным шагом", variable=int_test, value=2)
option14.place(x= 20, y=205)

option15 = ttk.Radiobutton(frame5, text="Метод Рунге-Кутта-Фельберга 4-5-го порядка с адаптивным шагом", variable=int_test, value=3)
option15.place(x= 20, y=230)

option16 = ttk.Radiobutton(frame5, text="Метод Гира для жёстких систем с адаптивным шагом", variable=int_test, value=4)
option16.place(x= 20, y=255)

label_param12 = ttk.Label(frame6, text="Iapp_1:", background="white")
label_param12.place(x= 20, y=60)

label_param13 = ttk.Label(frame6, text="Iapp_2:", background="white")
label_param13.place(x= 20, y=100)

label_param14 = ttk.Label(frame6, text="g_syn:", background="white")
label_param14.place(x= 20, y=140)

entry_param7 = ttk.Entry(frame6, textvariable=float_iapp_1)
entry_param7.place(x= 65, y= 60)

entry_param8 = ttk.Entry(frame6, textvariable=float_iapp_2)
entry_param8.place(x= 65, y= 100)

entry_param9 = ttk.Entry(frame6, textvariable=float_gsyn, validate="focusout", validatecommand=validate_float_gsyn)
entry_param9.place(x= 65, y= 140)

option5 = ttk.Radiobutton(frame6, text="Возбуждающая связь (E_syn = 0)", variable=int_Esyn, value=0)
option5.place(x= 20, y=180)

option6 = ttk.Radiobutton(frame6, text="Тормозная связь (E_syn = -90)", variable=int_Esyn, value=-90)
option6.place(x= 20, y=200)

label_param14 = ttk.Label(frame6, text="v4:", background="white")
label_param14.place(x= 20, y=230)

entry_param7 = ttk.Entry(frame6, textvariable=float_u4)
entry_param7.place(x= 65, y= 230)

label_param14 = ttk.Label(frame6, text="gastro:", background="white")
label_param14.place(x= 20, y=250)

entry_param7 = ttk.Entry(frame6, textvariable=float_gastro)
entry_param7.place(x= 65, y= 250)

button_integrate5 = ttk.Button(frame6, text="Запустить интегрирование", command=integrate5)
button_integrate5.place(x=20, y=550)

option13 = ttk.Radiobutton(frame6, text="Метод Эйлера 1-го порядка с фиксированным шагом", variable=int_test, value=1)
option13.place(x= 20, y=300)

option14 = ttk.Radiobutton(frame6, text="Метод Рунге-Кутта 4-го порядка с фиксированным шагом", variable=int_test, value=2)
option14.place(x= 20, y=350)

option15 = ttk.Radiobutton(frame6, text="Метод Рунге-Кутта-Фельберга 4-5-го порядка с адаптивным шагом", variable=int_test, value=3)
option15.place(x= 20, y=400)

option16 = ttk.Radiobutton(frame6, text="Метод Гира для жёстких систем с адаптивным шагом", variable=int_test, value=4)
option16.place(x= 20, y=450)

root.mainloop()