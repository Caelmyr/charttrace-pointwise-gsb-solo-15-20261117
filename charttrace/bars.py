"""Stacked bar collection on a linear value axis with a calibrated zero baseline."""
import csv
import io
import uuid
import cv2
import numpy as np
from image_lab.validation import Invalid
from .coordinates import Coordinates, axis_forward, axis_inverse, finite

MAX_BARS = 60
MAX_COLORS = 12
ROWS = 1200
COLS = 21
MATCH = .5


def identifier():
    return uuid.uuid4().hex[:24]


def default_bars():
    return {'zero_pixel': None, 'width_px': 40, 'tolerance': 26.,
            'min_height_px': 4, 'colors': [], 'items': []}


def bars_config(value):
    if not isinstance(value, dict):
        raise Invalid('堆叠柱设置必须是对象')
    config = default_bars()
    zero = value.get('zero_pixel')
    if zero is not None:
        point = finite(zero, '零基线点')
        if point.shape != (2,):
            raise Invalid('零基线点需要两个像素坐标')
        config['zero_pixel'] = point.tolist()
    for key, lower, upper in [('width_px', 2, 600), ('min_height_px', 1, 120)]:
        raw = value.get(key, config[key])
        if isinstance(raw, bool) or not isinstance(raw, int) or not lower <= raw <= upper:
            raise Invalid(key + ' 超出范围')
        config[key] = raw
    tolerance = float(finite(value.get('tolerance', config['tolerance']), 'tolerance'))
    if not 1 <= tolerance <= 100:
        raise Invalid('tolerance 超出范围')
    config['tolerance'] = tolerance
    colors = value.get('colors') or []
    if not isinstance(colors, list) or len(colors) > MAX_COLORS:
        raise Invalid('分段颜色最多 ' + str(MAX_COLORS) + ' 个')
    for index, item in enumerate(colors):
        if not isinstance(item, dict):
            raise Invalid('分段颜色必须是对象')
        rgb = finite(item.get('color'), '分段颜色')
        if rgb.shape != (3,) or rgb.min() < 0 or rgb.max() > 255:
            raise Invalid('分段颜色需要三个0–255通道')
        config['colors'].append({'name': str(item.get('name') or '段%d' % (index + 1))[:40],
                                 'color': rgb.tolist()})
    items = value.get('items') or []
    if not isinstance(items, list) or len(items) > MAX_BARS:
        raise Invalid('柱最多 ' + str(MAX_BARS) + ' 根')
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise Invalid('柱必须是对象')
        point = finite(item.get('pixel'), '柱位置')
        if point.shape != (2,):
            raise Invalid('柱位置需要两个像素坐标')
        config['items'].append({'id': str(item.get('id') or identifier())[:24],
                                'pixel': point.tolist(),
                                'label': str(item.get('label') or 'G%d' % (index + 1))[:40]})
    return config


def _runs(labels, min_rows):
    """Constant-label runs of at least min_rows. Same-label runs separated only
    by short unknown gaps (gridlines, thin borders) are merged back together."""
    runs = []
    start = 0
    for row in range(1, len(labels) + 1):
        if row == len(labels) or labels[row] != labels[start]:
            runs.append([start, row - 1, labels[start]])
            start = row
    merged = []
    for run in runs:
        gap = labels[merged[-1][1] + 1:run[0]] if merged else []
        if (merged and run[2] >= 0 and merged[-1][2] == run[2]
                and run[0] - merged[-1][1] - 1 <= min_rows and (np.asarray(gap) == -1).all()):
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [run for run in merged if run[2] >= 0 and run[1] - run[0] + 1 >= min_rows]


def _unknown_gaps(labels, nonbg, min_rows):
    """Unlabeled non-background stretches: likely segments whose color was never
    picked, reported so nothing is silently absorbed into neighbors."""
    gaps = []
    row = 0
    while row < len(labels):
        if labels[row] == -1 and nonbg[row]:
            start = row
            while row + 1 < len(labels) and labels[row + 1] == -1 and nonbg[row + 1]:
                row += 1
            if row - start + 1 >= min_rows:
                gaps.append((start, row))
        row += 1
    return gaps


def extract_bars(image, calibration, config, overlay):
    coords = Coordinates(calibration)
    axis = calibration['y']
    if axis['scale'] != 'linear':
        raise Invalid('堆叠柱采集暂仅支持线性数值轴，请将 Y 轴改为线性')
    if not config['colors']:
        raise Invalid('请先添加分段颜色')
    if not config['items']:
        raise Invalid('请先点击添加柱位置')
    basis_x = np.asarray(calibration['x_end'], float) - np.asarray(calibration['origin'], float)
    basis_y = np.asarray(calibration['y_end'], float) - np.asarray(calibration['origin'], float)
    len_x = float(np.hypot(*basis_x))
    len_y = float(np.hypot(*basis_y))
    if config['zero_pixel'] is not None:
        y0 = float(coords.pixels_to_chart([config['zero_pixel']])[0, 1])
    else:
        y0 = float(axis_forward(0., axis))
    zero_value = float(axis_inverse(np.array([y0]), axis)[0])
    half_w = config['width_px'] / len_x / 2
    y_lo = min(0., y0) - .03
    y_hi = max(1., y0) + .03
    min_h = config['min_height_px'] / len_y
    min_rows = max(2, int(round(min_h / ((y_hi - y_lo) / ROWS))))
    rgb = image[:, :, :3].astype(np.float32) / 255
    alpha = image[:, :, 3].astype(np.float32) / 255
    premult = rgb * alpha[:, :, None]
    targets = [cv2.cvtColor(np.asarray(c['color'], np.float32).reshape(1, 1, 3) / 255,
                            cv2.COLOR_RGB2LAB)[0, 0] for c in config['colors']]
    result = {'zero_value': zero_value, 'zero_chart_y': y0, 'colors': config['colors'],
              'width_px': config['width_px'], 'items': []}
    for index, item in enumerate(config['items']):
        cx = float(coords.pixels_to_chart([item['pixel']])[0, 0])
        xs = np.linspace(cx - half_w, cx + half_w, COLS)
        ys = np.linspace(y_lo, y_hi, ROWS)
        grid = coords.chart_to_pixels(np.column_stack((np.tile(xs, ROWS),
                                                       np.repeat(ys, COLS)))).astype(np.float32)
        map_x = grid[:, 0].reshape(ROWS, COLS)
        map_y = grid[:, 1].reshape(ROWS, COLS)
        sampled = cv2.remap(premult, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        coverage = cv2.remap(alpha, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Inner columns only: thin borders and neighbor bars stay outside the vote.
        inner = slice(COLS // 4, COLS - COLS // 4)
        strip = sampled + (1 - coverage[:, :, None])
        strip = strip[:, inner, :]
        inner_cov = coverage[:, inner]
        lab = cv2.cvtColor(strip, cv2.COLOR_RGB2LAB)
        prob = np.stack([np.exp(-.5 * (np.sqrt(((lab - t) ** 2).sum(axis=2))
                                / config['tolerance']) ** 2) * inner_cov for t in targets], axis=2)
        row_prob = np.median(prob, axis=1)
        labels = np.argmax(row_prob, axis=1)
        labels[np.max(row_prob, axis=1) < MATCH] = -1
        runs = _runs(labels, min_rows)
        median = np.median(strip, axis=1)
        nonbg = ((median.max(axis=1) - median.min(axis=1) > .08) | (median.mean(axis=1) < .78))
        nonbg &= np.median(inner_cov, axis=1) > .5
        gaps = _unknown_gaps(labels, nonbg, min_rows)

        def edge(row_edge):
            return y_lo + row_edge / ROWS * (y_hi - y_lo)

        segments = []
        for i, run in enumerate(runs):
            left = run[0]
            if i:
                prev = runs[i - 1]
                between = [g for g in gaps if prev[1] < g[0] and g[1] < run[0]]
                left = (prev[1] + 1 + run[0]) / 2 if not between else (between[-1][1] + 1 + run[0]) / 2
            right = run[1] + 1
            if i + 1 < len(runs):
                following = runs[i + 1]
                between = [g for g in gaps if run[1] < g[0] and g[1] < following[0]]
                right = (run[1] + 1 + following[0]) / 2 if not between else (run[1] + 1 + between[0][0]) / 2
            ya, yb = edge(left), edge(right)
            pieces = [(ya, y0), (y0, yb)] if ya < y0 < yb else [(ya, yb)]
            for lo, hi in pieces:
                if hi - lo < min_h:
                    continue
                side = 'positive' if (lo + hi) / 2 >= y0 else 'negative'
                v_lo = float(axis_inverse(np.array([lo]), axis)[0])
                v_hi = float(axis_inverse(np.array([hi]), axis)[0])
                start, end = (v_lo, v_hi) if side == 'positive' else (v_hi, v_lo)
                p_lo = coords.chart_to_pixels([(cx, lo)])[0]
                p_hi = coords.chart_to_pixels([(cx, hi)])[0]
                pixel_start, pixel_end = (p_lo, p_hi) if side == 'positive' else (p_hi, p_lo)
                segments.append({'color_index': int(run[2]), 'name': config['colors'][run[2]]['name'],
                                 'color': config['colors'][run[2]]['color'], 'side': side,
                                 'start': start, 'end': end, 'delta': end - start,
                                 'y_start': lo, 'y_end': hi,
                                 'pixel_start': pixel_start.tolist(), 'pixel_end': pixel_end.tolist(),
                                 'confidence': float(np.mean(row_prob[run[0]:run[1] + 1, run[2]]))})
        positives = sorted((s for s in segments if s['side'] == 'positive'), key=lambda s: s['y_start'])
        negatives = sorted((s for s in segments if s['side'] == 'negative'), key=lambda s: -s['y_end'])
        for stack, seg in enumerate(positives, 1):
            seg['stack'] = stack
        for stack, seg in enumerate(negatives, 1):
            seg['stack'] = stack
        detected = {s['color_index'] for s in segments}
        unlabeled = [{'y_start': edge(g[0]), 'y_end': edge(g[1] + 1),
                      'start': float(axis_inverse(np.array([edge(g[0])]), axis)[0]),
                      'end': float(axis_inverse(np.array([edge(g[1] + 1)]), axis)[0])} for g in gaps]
        result['items'].append({'id': item['id'], 'label': item['label'], 'pixel': item['pixel'],
                                'index': index + 1, 'chart_x': cx,
                                'x': float(axis_inverse(np.array([cx]), calibration['x'])[0]),
                                'segments': positives + negatives,
                                'missing': [i for i in range(len(config['colors'])) if i not in detected],
                                'unlabeled': unlabeled})
    layer = overlay.copy()
    for item in result['items']:
        for seg in item['segments']:
            quad = coords.chart_to_pixels([(item['chart_x'] - half_w, seg['y_start']),
                                           (item['chart_x'] + half_w, seg['y_start']),
                                           (item['chart_x'] + half_w, seg['y_end']),
                                           (item['chart_x'] - half_w, seg['y_end'])]).astype(np.int32)
            cv2.fillPoly(layer, [quad], tuple(int(c) for c in seg['color']))
    cv2.addWeighted(layer, .28, overlay, .72, 0, overlay)
    for item in result['items']:
        strip_quad = coords.chart_to_pixels([(item['chart_x'] - half_w, y_lo),
                                             (item['chart_x'] + half_w, y_lo),
                                             (item['chart_x'] + half_w, y_hi),
                                             (item['chart_x'] - half_w, y_hi)]).astype(np.int32)
        cv2.polylines(overlay, [strip_quad], True, (120, 128, 140), 1, cv2.LINE_AA)
        for seg in item['segments']:
            quad = coords.chart_to_pixels([(item['chart_x'] - half_w, seg['y_start']),
                                           (item['chart_x'] + half_w, seg['y_start']),
                                           (item['chart_x'] + half_w, seg['y_end']),
                                           (item['chart_x'] - half_w, seg['y_end'])]).astype(np.int32)
            cv2.polylines(overlay, [quad], True, (25, 32, 45), 1, cv2.LINE_AA)
        for u in item['unlabeled']:
            quad = coords.chart_to_pixels([(item['chart_x'] - half_w, u['y_start']),
                                           (item['chart_x'] + half_w, u['y_start']),
                                           (item['chart_x'] + half_w, u['y_end']),
                                           (item['chart_x'] - half_w, u['y_end'])]).astype(np.int32)
            cv2.polylines(overlay, [quad], True, (220, 40, 55), 2, cv2.LINE_AA)
        at = coords.chart_to_pixels([(item['chart_x'], y0)])[0]
        cv2.putText(overlay, str(item['index']), (int(at[0]) - 5, int(at[1]) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 90, 200), 2)
    zero = coords.chart_to_pixels([(-.03, y0), (1.03, y0)]).astype(np.int32)
    cv2.line(overlay, tuple(zero[0]), tuple(zero[1]), (25, 160, 90), 2, cv2.LINE_AA)
    return result, overlay


def bars_csv_bytes(bars):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['bar', 'bar_index', 'x', 'stack', 'side', 'segment', 'color_r', 'color_g',
                     'color_b', 'start', 'end', 'delta', 'pixel_x', 'pixel_y_start',
                     'pixel_y_end', 'confidence', 'status'])
    for item in bars['items']:
        for seg in item['segments']:
            writer.writerow([item['label'], item['index'], item['x'], seg['stack'], seg['side'],
                             seg['name'], *[round(c, 1) for c in seg['color']],
                             seg['start'], seg['end'], seg['delta'],
                             round(seg['pixel_start'][0], 2), round(seg['pixel_start'][1], 2),
                             round(seg['pixel_end'][1], 2), round(seg['confidence'], 4), 'ok'])
        for missing in item['missing']:
            color = bars['colors'][missing]
            writer.writerow([item['label'], item['index'], item['x'], '', '', color['name'],
                             *[round(c, 1) for c in color['color']],
                             '', '', '', '', '', '', '', 'missing'])
        for u in item['unlabeled']:
            writer.writerow([item['label'], item['index'], item['x'], '', '', '', '', '', '',
                             u['start'], u['end'], '', '', '', '', '', 'unlabeled'])
    return output.getvalue().encode('utf-8-sig')
