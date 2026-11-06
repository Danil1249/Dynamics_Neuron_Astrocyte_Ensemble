import numpy as np
from matplotlib import pyplot as plt


def arr(dataset, name):
    try:
        if dataset is None or name not in dataset:
            return np.asarray([], dtype=float)
        return np.asarray(dataset.get(name), dtype=float)
    except Exception:
        return np.asarray([], dtype=float)


def scalar(dataset, name, default=None):
    try:
        if dataset is None or name not in dataset:
            return default
        value = np.asarray(dataset.get(name))
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    except Exception:
        return default


def empty_figure(title='Нет сохранённых численных данных'):
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.text(0.5, 0.5, title, ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def create_single_figures(dataset):
    t = arr(dataset, 't')
    v = arr(dataset, 'v')
    m = arr(dataset, 'm')
    n = arr(dataset, 'n')
    h = arr(dataset, 'h')
    phase_v = arr(dataset, 'phase_v')
    phase_h = arr(dataset, 'phase_h')
    freq_t = arr(dataset, 'freq_t')
    freq = arr(dataset, 'freq')

    figures = {}
    if len(t) and len(v):
        fig = plt.figure(figsize=(6.2, 3.2))
        plt.plot(t, v)
        plt.xlabel('Время, с')
        plt.ylabel('Мембранный потенциал V, мВ')
        plt.ylim(-80, 20)
        if len(t) > 1:
            plt.xlim(float(t[0]), float(t[-1]))
        plt.grid(True)
        figures['vt'] = fig
    else:
        figures['vt'] = empty_figure()

    if len(phase_h) and len(phase_v):
        min_len = min(len(phase_h), len(phase_v))
        fig = plt.figure(figsize=(6.2, 3.2))
        plt.plot(phase_h[:min_len], phase_v[:min_len])
        plt.xlabel('Воротная переменная h')
        plt.ylabel('Мембранный потенциал V, мВ')
        plt.ylim(-80, 20)
        plt.xlim(0, 1)
        plt.grid(True)
        figures['phase'] = fig
    elif len(h) and len(v) and len(h) == len(v):
        fig = plt.figure(figsize=(6.2, 3.2))
        plt.plot(h, v)
        plt.xlabel('Воротная переменная h')
        plt.ylabel('Мембранный потенциал V, мВ')
        plt.ylim(-80, 20)
        plt.xlim(0, 1)
        plt.grid(True)
        figures['phase'] = fig

    if len(t) and len(m) and len(n) and len(h):
        min_len = min(len(t), len(m), len(n), len(h))
        fig = plt.figure(figsize=(6.2, 3.2))
        plt.plot(t[:min_len], m[:min_len], 'r-', label='m')
        plt.plot(t[:min_len], n[:min_len], 'g-', label='n')
        plt.plot(t[:min_len], h[:min_len], 'b-', label='h')
        plt.xlabel('Время, с')
        plt.ylabel('Воротные переменные m, n, h')
        plt.ylim(0, 1)
        if len(t) > 1:
            plt.xlim(float(t[0]), float(t[min_len - 1]))
        plt.grid(True)
        plt.legend(loc='upper right')
        figures['gates'] = fig

    if len(freq_t) and len(freq):
        fig = plt.figure(figsize=(6.2, 3.2))
        plt.plot(freq_t, freq)
        plt.xlabel('Время, с')
        plt.ylabel('Частота, 1/с')
        plt.ylim(0, 100)
        plt.grid(True)
        figures['freq'] = fig

    return figures


def first_active_iapp(values_x, values_y, eps=0.5):
    try:
        for x, y in zip(values_x, values_y):
            if np.isfinite(y) and float(y) > eps:
                return float(x)
    except Exception:
        pass
    return None


def frequency_sweep_metrics(x_values, forward_values, backward_values):
    metrics = {
        'forward_threshold': None,
        'backward_threshold': None,
        'max_diff': 0.0,
        'hysteresis_from': None,
        'hysteresis_to': None,
        'hysteresis_width': 0.0,
    }
    try:
        x_values = np.asarray(x_values, dtype=float)
        forward_values = np.asarray(forward_values, dtype=float)
        backward_values = np.asarray(backward_values, dtype=float)
        metrics['forward_threshold'] = first_active_iapp(x_values, forward_values)
        metrics['backward_threshold'] = first_active_iapp(x_values, backward_values)
        if len(forward_values) == len(backward_values) and len(forward_values):
            diff = np.abs(forward_values - backward_values)
            diff = np.where(np.isfinite(diff), diff, 0.0)
            metrics['max_diff'] = float(np.max(diff))
            max_value = float(np.nanmax([np.nanmax(forward_values), np.nanmax(backward_values)]))
            mask = diff > max(0.5, 0.05 * max(1.0, max_value))
            if np.any(mask):
                metrics['hysteresis_from'] = float(x_values[mask][0])
                metrics['hysteresis_to'] = float(x_values[mask][-1])
                metrics['hysteresis_width'] = max(0.0, metrics['hysteresis_to'] - metrics['hysteresis_from'])
    except Exception:
        pass
    return metrics


def get_xlim_for_frequency_series(*series, fallback_time=None):
    times = []
    for values in series:
        try:
            arr_values = np.asarray(values, dtype=float)
            arr_values = arr_values[np.isfinite(arr_values)]
            if arr_values.size:
                times.extend(arr_values.tolist())
        except Exception:
            pass

    if times:
        x0 = float(min(times))
        x1 = float(max(times))
        if x1 <= x0:
            x1 = x0 + 1.0
        pad = max(0.05, 0.03 * (x1 - x0))
        return x0 - pad, x1 + pad

    try:
        fallback = np.asarray(fallback_time, dtype=float)
        fallback = fallback[np.isfinite(fallback)]
        if fallback.size:
            x0 = float(fallback[0])
            x1 = float(fallback[-1])
            if x1 <= x0:
                x1 = x0 + 1.0
            return x0, x1
    except Exception:
        pass

    return 0.0, 1.0


def create_frequency_figure(dataset):
    x_values = arr(dataset, 'iapp')
    if not len(x_values):
        x_values = arr(dataset, 'T')
    forward = arr(dataset, 'forward')
    backward = arr(dataset, 'backward')
    if not len(backward):
        backward = arr(dataset, 'backward_plot')
    if not len(backward):
        backward_raw = arr(dataset, 'backward_raw')
        if len(backward_raw):
            backward = backward_raw[::-1]

    metrics = frequency_sweep_metrics(x_values, forward, backward)
    show_forward = scalar(dataset, 'show_forward', 1) == 1
    show_backward = scalar(dataset, 'show_backward', 1) == 1
    show_points = scalar(dataset, 'show_points', 1) == 1
    show_lines = scalar(dataset, 'show_lines', 1) == 1
    show_hysteresis = scalar(dataset, 'show_hysteresis', 1) == 1
    linestyle = '-' if show_lines else 'None'
    marker_forward = 'o' if show_points else None
    marker_backward = 'x' if show_points else None

    fig = plt.figure(figsize=(10.8, 6.2))
    ax = fig.add_subplot(111)

    if show_hysteresis and len(x_values) and len(forward) and len(backward):
        n = min(len(x_values), len(forward), len(backward))
        fill_mask = (
            np.isfinite(x_values[:n])
            & np.isfinite(forward[:n])
            & np.isfinite(backward[:n])
            & (x_values[:n] >= (metrics['hysteresis_from'] if metrics['hysteresis_from'] is not None else -np.inf))
            & (x_values[:n] <= (metrics['hysteresis_to'] if metrics['hysteresis_to'] is not None else np.inf))
            & (np.abs(forward[:n] - backward[:n]) > 0.25)
        )
        if np.any(fill_mask):
            ax.fill_between(x_values[:n], forward[:n], backward[:n], where=fill_mask, interpolate=True,
                            color='#fde68a', alpha=0.72, label='зона расхождения')
        elif metrics['hysteresis_from'] is not None and metrics['hysteresis_to'] is not None:
            ax.axvspan(metrics['hysteresis_from'], metrics['hysteresis_to'], ymin=0, ymax=1,
                       color='#fde68a', alpha=0.55, label='зона расхождения')

    if show_forward and len(x_values) and len(forward):
        n = min(len(x_values), len(forward))
        ax.plot(x_values[:n], forward[:n], color='red', linestyle=linestyle, marker=marker_forward,
                markersize=4, linewidth=1.5, label='прямой проход')
    if show_backward and len(x_values) and len(backward):
        n = min(len(x_values), len(backward))
        ax.plot(x_values[:n], backward[:n], color='blue', linestyle=linestyle, marker=marker_backward,
                markersize=4, linewidth=1.5, label='обратный проход')

    try:
        finite = np.concatenate([forward[np.isfinite(forward)], backward[np.isfinite(backward)]])
        max_y = max(35.0, float(np.max(finite)) * 1.15) if finite.size else 35.0
    except Exception:
        max_y = 35.0
    ax.set_xlabel('Ток Iapp, отн. ед.')
    ax.set_ylabel('Средняя частота, 1/с')
    ax.set_ylim(0, max_y)
    ax.set_xlim(0, 2)
    ax.set_xticks(np.arange(0.0, 2.1, 0.2))
    ax.legend(loc='upper right')
    return fig, metrics, {'iapp': x_values, 'forward': forward, 'backward': backward}


def create_pair_figures(dataset):
    t = arr(dataset, 't_v')
    if not len(t):
        t = arr(dataset, 't')
    v1 = arr(dataset, 'v1')
    v2 = arr(dataset, 'v2')
    ft1 = arr(dataset, 'freq_t1')
    f1 = arr(dataset, 'freq1')
    ft2 = arr(dataset, 'freq_t2')
    f2 = arr(dataset, 'freq2')

    fig1 = plt.figure(figsize=(6.2, 3.2))
    if len(t) and len(v1):
        n = min(len(t), len(v1))
        plt.plot(t[:n], v1[:n], 'r-', label='1')
    if len(t) and len(v2):
        n = min(len(t), len(v2))
        plt.plot(t[:n], v2[:n], 'b-', label='2')
    if not (len(t) and (len(v1) or len(v2))):
        plt.text(0.5, 0.5, 'Нет сохранённых потенциалов', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Время, с')
    plt.ylabel('Мембранные потенциалы V1, V2, мВ')
    plt.ylim(-80, 20)
    x_min = scalar(dataset, 'v_xlim_min', None)
    x_max = scalar(dataset, 'v_xlim_max', None)
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    elif len(t) > 1:
        plt.xlim(float(t[0]), float(t[-1]))
    plt.grid(True)
    plt.legend(loc='upper right')

    fig2 = plt.figure(figsize=(6.2, 3.2))
    if len(ft1) and len(f1):
        n = min(len(ft1), len(f1))
        plt.plot(ft1[:n], f1[:n], 'r-', label='1')
    if len(ft2) and len(f2):
        n = min(len(ft2), len(f2))
        plt.plot(ft2[:n], f2[:n], 'b-', label='2')
    if not ((len(ft1) and len(f1)) or (len(ft2) and len(f2))):
        plt.text(0.5, 0.5, 'Недостаточно спайков для расчёта частоты',
                 ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Время, с')
    plt.ylabel('Частоты нейронов, 1/с')
    plt.ylim(0, 100)
    fx0 = scalar(dataset, 'freq_xlim_min', None)
    fx1 = scalar(dataset, 'freq_xlim_max', None)
    if fx0 is not None and fx1 is not None:
        plt.xlim(fx0, fx1)
    else:
        plt.xlim(*get_xlim_for_frequency_series(ft1, ft2, fallback_time=t))
    plt.grid(True)
    plt.legend(loc='upper right')
    return fig1, fig2


def create_sync_map_figure(dataset, kind='sync_map'):
    x = arr(dataset, 'x')
    y = arr(dataset, 'y')
    values = arr(dataset, 'values')
    figsize = (8.8, 5.8) if kind == 'sync_time' else (7.8, 5.2)
    cmap = 'hot' if kind == 'sync_time' else 'binary_r'
    fig, ax = plt.subplots(figsize=figsize)
    if len(x) and len(y) and values.size:
        try:
            values_2d = np.asarray(values, dtype=float).reshape(len(y), len(x))
            im = ax.imshow(values_2d, extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
                           origin='lower', cmap=cmap)
            ax.set_aspect('auto')
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            yticks = ax.get_yticks()
            ax.set_yticklabels([round(item - 1.5, 1) for item in yticks])
            cbar = fig.colorbar(im, ax=ax, label='Значение')
            if kind == 'sync_time':
                cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        except Exception:
            ax.text(0.5, 0.5, 'Не удалось восстановить карту', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Нет сохранённой карты', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Синаптическая проводимость g_syn, отн. ед.')
    ax.set_ylabel('ΔIapp, отн. ед.')
    ax.grid(False)
    return fig


def draw_astro_highlight_bands(ax, display_settings):
    ax.axvspan(display_settings['yellow_start'], display_settings['yellow_end'], color='yellow', alpha=0.3, label='жёлтая зона')
    ax.axvspan(display_settings['pink_start'], display_settings['pink_end'], color='pink', alpha=0.3, label='розовая зона')


def create_astro_figures(dataset, display_settings):
    t_ca = arr(dataset, 't_ca')
    ca = arr(dataset, 'ca')
    ft1 = arr(dataset, 'freq_t1')
    f1 = arr(dataset, 'freq1')
    ft2 = arr(dataset, 'freq_t2')
    f2 = arr(dataset, 'freq2')

    fig1 = plt.figure(figsize=(11.8, 2.75))
    ax1 = fig1.add_subplot(111)
    if len(t_ca) and len(ca):
        n = min(len(t_ca), len(ca))
        ax1.plot(t_ca[:n], ca[:n], 'r-', label='Ca')
    else:
        ax1.text(0.5, 0.5, 'Нет сохранённой Ca-динамики', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_ylabel('Концентрация Ca, отн. ед.')
    ax1.set_xlabel('Время, с')
    ax1.axhline(y=0.3, color='gray', linestyle='--')
    try:
        ax1.set_xlim(display_settings['t_min'], display_settings['t_max'])
        draw_astro_highlight_bands(ax1, display_settings)
    except Exception:
        pass
    ax1.legend(loc='upper right')

    fig2 = plt.figure(figsize=(11.8, 2.75))
    ax2 = fig2.add_subplot(111)
    if len(ft1) and len(f1):
        n = min(len(ft1), len(f1))
        ax2.plot(ft1[:n], f1[:n], 'b-', label='нейрон 1')
    if len(ft2) and len(f2):
        n = min(len(ft2), len(f2))
        ax2.plot(ft2[:n], f2[:n], 'r-', label='нейрон 2')
    if not ((len(ft1) and len(f1)) or (len(ft2) and len(f2))):
        ax2.text(0.5, 0.5, 'Недостаточно спайков для расчёта частоты',
                 ha='center', va='center', transform=ax2.transAxes)
    ax2.set_ylabel('Частоты нейронов, 1/с')
    ax2.set_xlabel('Время, с')
    ax2.set_ylim(25, 35)
    try:
        ax2.set_xlim(display_settings['t_min'], display_settings['t_max'])
        draw_astro_highlight_bands(ax2, display_settings)
    except Exception:
        pass
    ax2.legend(loc='upper right')
    return fig1, fig2
