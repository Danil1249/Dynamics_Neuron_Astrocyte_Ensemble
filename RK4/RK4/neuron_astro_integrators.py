import numpy as np
from numba import jit
import scipy

from neuron_astro_model import (
    func_Ca,
    func_G,
    func_IP3,
    func_V,
    func_V2,
    func_h,
    func_m,
    func_n,
    func_z,
    gsyn_,
)


@jit(nopython=True, cache=True, nogil=True)
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


@jit(nopython=True, cache=True, nogil=True)
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


@jit(nopython=True, cache=True, nogil=True)
def EULER(z, q, vv, mm, nn, hh, iapp):
  Box = np.zeros((5, z + 1))
  V = vv
  m = mm
  n = nn
  h = hh

  Box[0, 0] = 0.0
  Box[1, 0] = V
  Box[2, 0] = m
  Box[3, 0] = n
  Box[4, 0] = h

  for i in range(1, z + 1):
    d_m = func_m(m, V)
    d_n = func_n(n, V)
    d_h = func_h(h, V)
    d_V = func_V(m, n, h, V, iapp)

    m = m + q * d_m
    n = n + q * d_n
    h = h + q * d_h
    V = V + q * d_V

    Box[0, i] = i * q
    Box[1, i] = V
    Box[2, i] = m
    Box[3, i] = n
    Box[4, i] = h

  return Box


@jit(nopython=True, cache=True, nogil=True)
def EULER1(z, q, vv, mm, nn, hh, iapp_1, iapp_2, gsyn, Esyn, u4, gastro):
    Box1 = np.zeros((5, z + 1))
    Box2 = np.zeros((5, z + 1))
    Box3 = np.zeros((4, z + 1))
    Box4 = np.zeros((2, z + 1))

    V1 = vv
    m1 = mm
    n1 = nn
    h1 = hh
    V2 = vv
    m2 = mm
    n2 = nn
    h2 = hh
    Ca = 0.07
    IP3 = 0.16
    zz = 0.67
    G = 0.0

    Box1[:, 0] = np.array([0.0, V1, m1, n1, h1])
    Box2[:, 0] = np.array([0.0, V2, m2, n2, h2])
    Box3[:, 0] = np.array([0.0, Ca, IP3, zz])
    Box4[:, 0] = np.array([0.0, G])

    for i in range(1, z + 1):
        gsyn1 = gsyn_(gsyn, Ca, gastro)

        d_m1 = func_m(m1, V1)
        d_n1 = func_n(n1, V1)
        d_h1 = func_h(h1, V1)
        d_V1 = func_V(m1, n1, h1, V1, iapp_1)

        d_m2 = func_m(m2, V2)
        d_n2 = func_n(n2, V2)
        d_h2 = func_h(h2, V2)
        d_V2 = func_V2(m2, n2, h2, V1, V2, iapp_2, gsyn1, Esyn)

        d_Ca = func_Ca(Ca, IP3, zz, u4)
        d_IP3 = func_IP3(Ca, IP3, zz, u4, G)
        d_z = func_z(Ca, IP3, zz, u4)
        d_G = func_G(G, V2)

        m1 = m1 + q * d_m1
        n1 = n1 + q * d_n1
        h1 = h1 + q * d_h1
        V1 = V1 + q * d_V1

        m2 = m2 + q * d_m2
        n2 = n2 + q * d_n2
        h2 = h2 + q * d_h2
        V2 = V2 + q * d_V2

        Ca = Ca + q * d_Ca
        IP3 = IP3 + q * d_IP3
        zz = zz + q * d_z
        G = G + q * d_G

        t = i * q
        Box1[:, i] = np.array([t, V1, m1, n1, h1])
        Box2[:, i] = np.array([t, V2, m2, n2, h2])
        Box3[:, i] = np.array([t, Ca, IP3, zz])
        Box4[:, i] = np.array([t, G])

    return Box1, Box2, Box3, Box4


SOLVE_IVP_OUTPUT_DT = 1e-5
SOLVE_IVP_MAX_POINTS = 3000001
SOLVE_IVP_MAX_STEP = 1e-4


def make_dense_t_eval(z, q):
  """Формирует достаточно плотную сетку вывода для RK45/BDF.

  В прошлой версии для адаптивных методов возвращалось максимум 20 000 точек
  на весь интервал. Для 30 секунд расчёта это давало шаг вывода около 0.0015 с,
  поэтому узкие спайки пропускались, а графики BDF/RK45 выглядели "рваными".
  Здесь шаг вывода делается существенно плотнее, но ограничивается сверху,
  чтобы не съесть память на очень длинных расчётах.
  """
  t_end = float(z) * float(q)
  if t_end <= 0:
    return np.array([0.0])

  # Не делаем сетку грубее, чем SOLVE_IVP_OUTPUT_DT, и не делаем её грубее
  # исходного фиксированного шага, если пользователь задал q больше.
  output_dt = max(float(q), SOLVE_IVP_OUTPUT_DT)
  points = int(np.floor(t_end / output_dt)) + 1
  points = max(2, min(points, int(z) + 1, SOLVE_IVP_MAX_POINTS))
  return np.linspace(0.0, t_end, points)


def solve_ivp_single(z, q, vv, mm, nn, hh, iapp, method):
  t_end = z * q
  t_eval = make_dense_t_eval(z, q)
  scipy_method = 'RK45' if method == 3 else 'BDF'

  def rhs(_, y):
    V, m, n, h = y
    return [
        func_V(m, n, h, V, iapp),
        func_m(m, V),
        func_n(n, V),
        func_h(h, V),
    ]

  sol = scipy.integrate.solve_ivp(
      rhs,
      (0, t_end),
      [vv, mm, nn, hh],
      method=scipy_method,
      t_eval=t_eval,
      rtol=1e-8,
      atol=1e-10,
      max_step=SOLVE_IVP_MAX_STEP,
  )
  if not sol.success:
    raise RuntimeError(sol.message)

  return np.vstack((sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]))


def solve_ivp_ensemble(z, q, vv, mm, nn, hh, iapp_1, iapp_2, gsyn, Esyn, u4, gastro, method):
  t_end = z * q
  t_eval = make_dense_t_eval(z, q)
  scipy_method = 'RK45' if method == 3 else 'BDF'

  def rhs(_, y):
    V1, m1, n1, h1, V2, m2, n2, h2, Ca, IP3, zz, G = y
    gsyn1 = gsyn_(gsyn, Ca, gastro)
    return [
        func_V(m1, n1, h1, V1, iapp_1),
        func_m(m1, V1),
        func_n(n1, V1),
        func_h(h1, V1),
        func_V2(m2, n2, h2, V1, V2, iapp_2, gsyn1, Esyn),
        func_m(m2, V2),
        func_n(n2, V2),
        func_h(h2, V2),
        func_Ca(Ca, IP3, zz, u4),
        func_IP3(Ca, IP3, zz, u4, G),
        func_z(Ca, IP3, zz, u4),
        func_G(G, V2),
    ]

  sol = scipy.integrate.solve_ivp(
      rhs,
      (0, t_end),
      [vv, mm, nn, hh, vv, mm, nn, hh, 0.07, 0.16, 0.67, 0.0],
      method=scipy_method,
      t_eval=t_eval,
      rtol=1e-8,
      atol=1e-10,
      max_step=SOLVE_IVP_MAX_STEP,
  )
  if not sol.success:
    raise RuntimeError(sol.message)

  Box1 = np.vstack((sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3]))
  Box2 = np.vstack((sol.t, sol.y[4], sol.y[5], sol.y[6], sol.y[7]))
  Box3 = np.vstack((sol.t, sol.y[8], sol.y[9], sol.y[10]))
  Box4 = np.vstack((sol.t, sol.y[11]))
  return Box1, Box2, Box3, Box4



__all__ = [
    'RK41', 'RK4', 'EULER', 'EULER1',
    'SOLVE_IVP_OUTPUT_DT', 'SOLVE_IVP_MAX_POINTS', 'SOLVE_IVP_MAX_STEP',
    'make_dense_t_eval', 'solve_ivp_single', 'solve_ivp_ensemble',
]
