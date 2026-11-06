import numpy as np


def Maximum(Box):
  """Быстрый поиск максимумов по траектории V(t).
  Возвращает тот же формат, что и старая функция: [times, values].
  """
  arr = np.asarray(Box)
  t = arr[0]
  V = arr[1]

  if len(V) < 3:
    return [[], []]

  mask = (V[1:-1] > V[:-2]) & (V[1:-1] > V[2:]) & (V[1:-1] > -10)
  times = t[1:-1][mask]
  values = V[1:-1][mask]

  if len(times) < 2:
    return [times.tolist(), values.tolist()]

  keep = np.diff(times) > 0.002
  return [times[:-1][keep].tolist(), values[:-1][keep].tolist()]


def Period(Cage):
  times = np.asarray(Cage[0])
  if len(times) < 2:
    return [[], []]
  return [times[:-1].tolist(), np.diff(times).tolist()]


def Частота(Cage):
  times = np.asarray(Cage[0])
  if len(times) < 2:
    return [[], []]
  periods = np.diff(times)
  freq = np.zeros_like(periods)
  valid = periods != 0
  freq[valid] = 1 / periods[valid]
  return [times[:-1].tolist(), freq.tolist()]


def Middle_Частота(Container):
  values = np.asarray(Container[1], dtype=float)
  if len(values) == 0:
    return 0.0
  mean_value = np.mean(values)
  if mean_value == 0:
    return 0.0
  return 1 / mean_value
  



__all__ = ['Maximum', 'Period', 'Частота', 'Middle_Частота']
