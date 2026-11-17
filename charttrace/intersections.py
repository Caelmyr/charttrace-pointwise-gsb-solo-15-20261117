"""Intersections and overlap intervals between two extracted series.

All geometry is evaluated in the calibrated chart basis: axis scale transforms
(linear / log) and reversed axis limits are already baked into chart
coordinates, so crossings are interpreted exactly as the curves are drawn on
the image. Only observed runs are compared; gaps split the polylines before any
interpolation, so blank columns can never fabricate a crossing.
"""
import csv
import io
import numpy as np
from image_lab.validation import Invalid
from .coordinates import Coordinates

EPS = 1e-12
CONTACT_PX = 1e-6


def _series(record, sid, label):
    series = next((s for s in record['series'] if s['id'] == sid), None)
    if series is None:
        raise Invalid(label + '不在该提取结果中')
    return series


def observed_polylines(series):
    """Split traced paths into polylines of continuously observed columns."""
    out = []
    for path in series.get('paths', []):
        chart = np.asarray(path.get('chart', []), dtype=float)
        observed = np.asarray(path.get('observed', []), dtype=bool)
        if chart.ndim != 2 or chart.shape[0] == 0 or observed.shape[0] != chart.shape[0]:
            continue
        positions = np.flatnonzero(observed)
        if not len(positions):
            continue
        for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
            out.append(chart[run].copy())
    return out


def segment_gap(a0, a1, b0, b1):
    """Minimum Euclidean distance between two pixel-space segments."""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    A = float(u @ u)
    B = float(u @ v)
    C = float(v @ v)
    D = float(u @ w)
    E = float(v @ w)
    denom = A * C - B * B
    s = np.clip((B * E - C * D) / denom, 0, 1) if denom > EPS else 0.0
    if C > EPS:
        t = (B * s + E) / C
        if t < 0:
            t = 0.0
            s = np.clip(-D / A, 0, 1) if A > EPS else 0.0
        elif t > 1:
            t = 1.0
            s = np.clip((B - D) / A, 0, 1) if A > EPS else 0.0
    else:
        t = 0.0
    pa = a0 + s * u
    pb = b0 + t * v
    return float(np.linalg.norm(pa - pb)), float(s), float(t), pa, pb


def point_segment_gap(p, a0, a1):
    u = a1 - a0
    length = float(u @ u)
    if length <= EPS:
        q = a0.copy()
        return float(np.linalg.norm(p - q)), 0.0, q, q
    t = float(np.clip((p - a0) @ u / length, 0, 1))
    q = a0 + t * u
    return float(np.linalg.norm(p - q)), t, q, q


def _point_payload(coords, x, ca, cb, pxa, pxb, crossing, extra=None):
    ya = float(np.interp(x, ca[:, 0], ca[:, 1]))
    yb = float(np.interp(x, cb[:, 0], cb[:, 1]))
    pa = np.array([np.interp(x, ca[:, 0], pxa[:, 0]),
                   np.interp(x, ca[:, 0], pxa[:, 1])])
    pb = np.array([np.interp(x, cb[:, 0], pxb[:, 0]),
                   np.interp(x, cb[:, 0], pxb[:, 1])])
    data_a = coords.chart_to_data([[x, ya]])[0]
    data_b = coords.chart_to_data([[x, yb]])[0]
    event = {'kind': 'point', 'crossing': bool(crossing),
             'chart_x': float(x), 'pixel': ((pa + pb) / 2).tolist(),
             'pixel_a': pa.tolist(), 'pixel_b': pb.tolist(),
             'gap_px': float(np.linalg.norm(pa - pb)),
             'x': float(data_a[0]),
             'y_a': float(data_a[1]), 'y_b': float(data_b[1])}
    if extra:
        event.update(extra)
    return event


def _interval_payload(coords, x0, x1, ka, kb, pxa, pxb, gaps, crossing):
    ya0 = float(np.interp(x0, ka[:, 0], ka[:, 1]))
    ya1 = float(np.interp(x1, ka[:, 0], ka[:, 1]))
    yb0 = float(np.interp(x0, kb[:, 0], kb[:, 1]))
    yb1 = float(np.interp(x1, kb[:, 0], kb[:, 1]))
    pa0 = np.array([np.interp(x0, ka[:, 0], pxa[:, 0]), np.interp(x0, ka[:, 0], pxa[:, 1])])
    pa1 = np.array([np.interp(x1, ka[:, 0], pxa[:, 0]), np.interp(x1, ka[:, 0], pxa[:, 1])])
    pb0 = np.array([np.interp(x0, kb[:, 0], pxb[:, 0]), np.interp(x0, kb[:, 0], pxb[:, 1])])
    pb1 = np.array([np.interp(x1, kb[:, 0], pxb[:, 0]), np.interp(x1, kb[:, 0], pxb[:, 1])])
    da0 = coords.chart_to_data([[x0, ya0]])[0]
    da1 = coords.chart_to_data([[x1, ya1]])[0]
    db0 = coords.chart_to_data([[x0, yb0]])[0]
    db1 = coords.chart_to_data([[x1, yb1]])[0]
    return {'kind': 'interval', 'crossing': bool(crossing),
            'chart_x': [float(x0), float(x1)],
            'polygon': [pa0.tolist(), pa1.tolist(), pb1.tolist(), pb0.tolist()],
            'pixel_a_start': pa0.tolist(), 'pixel_a_end': pa1.tolist(),
            'pixel_b_start': pb0.tolist(), 'pixel_b_end': pb1.tolist(),
            'x_start': float(da0[0]), 'x_end': float(da1[0]),
            'y_a_start': float(da0[1]), 'y_a_end': float(da1[1]),
            'y_b_start': float(db0[1]), 'y_b_end': float(db1[1]),
            'min_gap_px': float(np.min(gaps)), 'mean_gap_px': float(np.mean(gaps))}


def _sign_change(d, k0, k1):
    """Nonzero signs flanking a close run decide whether an interval crosses."""
    left = 0.0
    for j in range(k0 - 1, -1, -1):
        if abs(d[j]) > EPS:
            left = np.sign(d[j])
            break
    right = 0.0
    for j in range(k1 + 1, len(d)):
        if abs(d[j]) > EPS:
            right = np.sign(d[j])
            break
    return bool(left and right and left != right)


def _compare(ca, pxa, cb, pxb, coords, tol):
    """Events between one observed polyline of each series."""
    events = []
    ax0, ax1 = float(ca[0, 0]), float(ca[-1, 0])
    bx0, bx1 = float(cb[0, 0]), float(cb[-1, 0])
    lo, hi = max(ax0, bx0), min(ax1, bx1)
    if hi < lo - EPS:
        return events, False
    # Degenerate polylines: an observed run of a single column.
    if len(ca) == 1 or len(cb) == 1:
        if hi - lo <= EPS:
            pa = pxa[0] if len(ca) == 1 else np.array([np.interp(lo, ca[:, 0], pxa[:, 0]),
                                                        np.interp(lo, ca[:, 0], pxa[:, 1])])
            pb = pxb[0] if len(cb) == 1 else np.array([np.interp(lo, cb[:, 0], pxb[:, 0]),
                                                        np.interp(lo, cb[:, 0], pxb[:, 1])])
            gap = float(np.linalg.norm(pa - pb))
            if gap <= tol + CONTACT_PX:
                events.append(_point_payload(coords, lo, ca, cb, pxa, pxb, gap <= CONTACT_PX))
            return events, True
        if len(ca) == 1:
            point, poly, ppix, apix = ca[0], cb, pxa[0], pxb
        else:
            point, poly, ppix, apix = cb[0], ca, pxb[0], pxa
        if poly[0, 0] <= point[0] <= poly[-1, 0]:
            best_gap = np.inf
            for i in range(len(poly) - 1):
                gap, _, _, _ = point_segment_gap(ppix, apix[i], apix[i + 1])
                if gap < best_gap:
                    best_gap = gap
            if best_gap <= tol + CONTACT_PX:
                x = float(point[0])
                events.append(_point_payload(coords, x, ca, cb, pxa, pxb, best_gap <= CONTACT_PX))
        return events, True

    if hi - lo <= EPS:
        pa = np.array([np.interp(lo, ca[:, 0], pxa[:, 0]), np.interp(lo, ca[:, 0], pxa[:, 1])])
        pb = np.array([np.interp(lo, cb[:, 0], pxb[:, 0]), np.interp(lo, cb[:, 0], pxb[:, 1])])
        if np.linalg.norm(pa - pb) <= tol + CONTACT_PX:
            events.append(_point_payload(coords, lo, ca, cb, pxa, pxb, False))
        return events, True

    knots = np.unique(np.concatenate([
        ca[(ca[:, 0] >= lo - EPS) & (ca[:, 0] <= hi + EPS), 0],
        cb[(cb[:, 0] >= lo - EPS) & (cb[:, 0] <= hi + EPS), 0],
        [lo, hi]]))
    ya = np.interp(knots, ca[:, 0], ca[:, 1])
    yb = np.interp(knots, cb[:, 0], cb[:, 1])
    ax = np.interp(knots, ca[:, 0], pxa[:, 0])
    ay = np.interp(knots, ca[:, 0], pxa[:, 1])
    bx = np.interp(knots, cb[:, 0], pxb[:, 0])
    by = np.interp(knots, cb[:, 0], pxb[:, 1])
    gaps = np.hypot(ax - bx, ay - by)
    d = ya - yb
    close = gaps <= tol + CONTACT_PX

    roots = {}
    for i in range(len(knots) - 1):
        if d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            roots[i] = float(knots[i] + t * (knots[i + 1] - knots[i]))

    used_intervals = set()
    touches = []
    i = 0
    while i < len(knots):
        if not close[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(knots) and close[j + 1]:
            j += 1
        if i == j:
            # Isolated knot inside the tolerance band. A zero here is an exact
            # meeting point; flanking signs tell a crossing from a tangent.
            # Otherwise it is a single-sample near contact that must not be missed.
            if gaps[i] <= CONTACT_PX:
                touches.append((float(knots[i]), 0.0, _sign_change(d, i, i)))
            else:
                touches.append((float(knots[i]), float(gaps[i]), False))
        else:
            crossing = _sign_change(d, i, j)
            run_gaps = [gaps[i:j + 1]]
            for k in range(i, j):
                gap, _, _, _, _ = segment_gap(
                    np.array([ax[k], ay[k]]), np.array([ax[k + 1], ay[k + 1]]),
                    np.array([bx[k], by[k]]), np.array([bx[k + 1], by[k + 1]]))
                run_gaps.append(np.array([gap]))
            events.append(_interval_payload(
                coords, float(knots[i]), float(knots[j]), ca, cb, pxa, pxb,
                np.concatenate(run_gaps), crossing))
            # Cells covered by the band cannot raise separate root/touch events.
            used_intervals.update(range(i, j))
        i = j + 1

    for k in range(len(knots) - 1):
        if k in used_intervals:
            continue
        if close[k] or close[k + 1]:
            continue
        if k in roots:
            events.append(_point_payload(coords, roots[k], ca, cb, pxa, pxb, True))
            continue
        gap, s, t, pa, pb = segment_gap(
            np.array([ax[k], ay[k]]), np.array([ax[k + 1], ay[k + 1]]),
            np.array([bx[k], by[k]]), np.array([bx[k + 1], by[k + 1]]))
        if gap <= tol + CONTACT_PX:
            x = float(knots[k] + s * (knots[k + 1] - knots[k]))
            touches.append((x, float(gap), False))

    # A single geometric glance may touch several adjacent segments; merge
    # candidates whose closest points fall within the same tolerance band.
    touches.sort()
    merged = []
    x_span = max(float(np.linalg.norm(coords.basis[:, 0])), EPS)
    for x, gap, crossing in touches:
        if merged and x - merged[-1][0] <= (tol + CONTACT_PX) * 1.0001 / x_span:
            if gap < merged[-1][1]:
                merged[-1] = (x, gap, crossing)
        else:
            merged.append((x, gap, crossing))
    for x, gap, crossing in merged:
        events.append(_point_payload(coords, x, ca, cb, pxa, pxb, crossing,
                                     None if gap <= CONTACT_PX else {'glance_gap_px': gap}))
    return events, True


def measure(record, series_a, series_b, tolerance):
    if series_a == series_b:
        raise Invalid('请选择两条不同的曲线')
    try:
        tol = float(tolerance)
    except (TypeError, ValueError):
        raise Invalid('容差必须是数字')
    if not np.isfinite(tol) or tol < 0 or tol > 500:
        raise Invalid('容差需在0–500像素之间')
    coords = Coordinates(record['document']['calibration'])
    a = _series(record, series_a, '曲线A')
    b = _series(record, series_b, '曲线B')
    lines_a = [(c, coords.chart_to_pixels(c)) for c in observed_polylines(a)]
    lines_b = [(c, coords.chart_to_pixels(c)) for c in observed_polylines(b)]
    events = []
    pairs = 0
    overlapping = 0
    for ca, pxa in lines_a:
        for cb, pxb in lines_b:
            pairs += 1
            found, has_overlap = _compare(ca, pxa, cb, pxb, coords, tol)
            overlapping += int(has_overlap)
            events.extend(found)

    def order_key(e):
        return e['chart_x'] if e['kind'] == 'point' else e['chart_x'][0]

    events.sort(key=order_key)
    for index, event in enumerate(events, 1):
        event['id'] = index
    counts = {'total': len(events),
              'crossings': sum(1 for e in events if e['kind'] == 'point' and e['crossing']),
              'touches': sum(1 for e in events if e['kind'] == 'point' and not e['crossing']),
              'intervals': sum(1 for e in events if e['kind'] == 'interval')}
    return {'tolerance_px': tol,
            'series_a': {'id': a['id'], 'name': a['name']},
            'series_b': {'id': b['id'], 'name': b['name']},
            'polyline_pairs': pairs, 'overlapping_pairs': overlapping,
            'counts': counts, 'events': events}


def measurement_csv(payload):
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(['event_id', 'type', 'crossing', 'tolerance_px',
                     'x_start', 'x_end', 'y_a_start', 'y_a_end',
                     'y_b_start', 'y_b_end', 'gap_min_px', 'gap_mean_px',
                     'pixel_x_start', 'pixel_y_start', 'pixel_x_end', 'pixel_y_end'])
    for e in payload['events']:
        if e['kind'] == 'point':
            px, py = e['pixel']
            writer.writerow([e['id'], 'point', int(e['crossing']), payload['tolerance_px'],
                             e['x'], e['x'], e['y_a'], e['y_a'], e['y_b'], e['y_b'],
                             e['gap_px'], e['gap_px'], px, py, px, py])
        else:
            x0, _ = e['polygon'][0]
            _, y0 = e['polygon'][0]
            x1, y1 = e['polygon'][1]
            writer.writerow([e['id'], 'interval', int(e['crossing']), payload['tolerance_px'],
                             e['x_start'], e['x_end'], e['y_a_start'], e['y_a_end'],
                             e['y_b_start'], e['y_b_end'], e['min_gap_px'], e['mean_gap_px'],
                             x0, y0, x1, y1])
    return output.getvalue().encode('utf-8-sig')
