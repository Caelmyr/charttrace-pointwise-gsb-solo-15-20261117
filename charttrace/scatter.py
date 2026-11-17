"""Scatter marker detection in source pixel coordinates.

Filled dots and hollow rings are both reduced to one center per marker.
Hollow rings are closed and filled before connected components so they never
become a ring of tiny points; touching markers are split with distance
transform seeds and a nearest-seed assignment inside the shared blob.
"""
import math
import uuid
import cv2
import numpy as np
from scipy import ndimage
from .coordinates import Coordinates


def new_id():
    return uuid.uuid4().hex[:24]


def scatter_defaults():
    return {'kind': 'scatter', 'color': [210, 60, 70], 'tolerance': 26.,
            'min_confidence': .28, 'point_min': 8, 'point_max': 28}


def evidence_map(image, color, tolerance):
    """LAB color probability; transparent pixels can never be evidence."""
    rgb = image[:, :, :3].astype(np.float32) / 255
    alpha = image[:, :, 3].astype(np.float32) / 255
    visible = rgb * alpha[:, :, None] + (1 - alpha[:, :, None])
    lab = cv2.cvtColor(visible, cv2.COLOR_RGB2LAB)
    target = cv2.cvtColor(np.asarray(color, np.float32).reshape(1, 1, 3) / 255,
                          cv2.COLOR_RGB2LAB)[0, 0]
    distance = np.sqrt(np.sum((lab - target) ** 2, axis=2))
    return np.exp(-.5 * (distance / tolerance) ** 2) * alpha


def fill_holes(mask):
    """Fill enclosed background (the inside of a hollow ring) without a flood."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    flooded = padded.copy()
    cv2.floodFill(flooded, None, (0, 0), 2)
    holes = (flooded == 0).astype(np.uint8)
    return np.maximum(mask, holes[1:-1, 1:-1])


def _peak_seeds(distance, dmin):
    peaks = (distance == cv2.dilate(distance, np.ones((3, 3), np.uint8))) & \
            (distance > max(1.5, dmin * .28))
    ys, xs = np.nonzero(peaks)
    order = np.argsort(distance[ys, xs])[::-1]
    chosen = []
    min_gap = dmin * .62
    for i in order:
        cx, cy = int(xs[i]), int(ys[i])
        if all((cx - x) ** 2 + (cy - y) ** 2 > min_gap ** 2 for x, y in chosen):
            chosen.append((cx, cy))
    if not chosen:
        cy, cx = np.unravel_index(int(np.argmax(distance)), distance.shape)
        chosen.append((int(cx), int(cy)))
    return chosen


def _farthest_seeds(region, seeds, area, dmin, dmax):
    """A single broad peak can still hide several touching markers."""
    disk = math.pi * (dmax / 2) ** 2
    wanted = min(8, max(1, math.ceil(area / disk)))
    if len(seeds) >= wanted or area < disk * 1.5:
        return seeds
    coords = np.column_stack(np.nonzero(region))  # (y, x)
    while len(seeds) < wanted:
        distances = [
            (coords[:, 0] - sy) ** 2 + (coords[:, 1] - sx) ** 2
            for sx, sy in seeds
        ]
        nearest = np.minimum.reduce(distances)
        j = int(np.argmax(nearest))
        if nearest[j] < (dmin * .5) ** 2:
            break
        seeds.append((int(coords[j, 1]), int(coords[j, 0])))
    return seeds


def _assign_labels(region, seeds):
    """Nearest-seed Voronoi labels restricted to the blob."""
    seed_map = np.zeros(region.shape, np.int32)
    for i, (x, y) in enumerate(seeds, 1):
        seed_map[y, x] = i
    _, (yy, xx) = ndimage.distance_transform_edt(seed_map == 0, return_indices=True)
    nearest = seed_map[yy, xx]
    return np.where(region, nearest, 0).astype(np.int32)


def _is_ring(part, colored):
    """A hollow marker: colored pixels sit around an empty (or enclosed) center."""
    region = part.astype(np.uint8)
    interior = fill_holes(region) - region
    ys2, xs2 = np.nonzero(colored & part)
    if not len(xs2):
        return False
    cy, cx = ys2.mean(), xs2.mean()
    radius = np.sqrt((ys2 - cy) ** 2 + (xs2 - cx) ** 2)
    outer_r = radius.mean()
    if outer_r < 3:
        return False
    if int(interior.sum()) >= 4:
        ys, xs = np.nonzero(interior)
        inner_r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).mean()
        if outer_r > inner_r * 1.2:
            return True
    # Open (gapped) ring: the centroid of the colored pixels is itself empty,
    # and the colored mass concentrates at a consistent radius around it.
    h, w = region.shape
    ix, iy = int(round(cx)), int(round(cy))
    if not (0 <= iy < h and 0 <= ix < w) or colored[iy, ix]:
        return False
    return float(radius.std() / outer_r) < .32


def _component_points(region, colored, evidence, dmin, dmax):
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    seeds = _farthest_seeds(region, _peak_seeds(distance, dmin),
                            int(region.sum()), dmin, dmax)
    labels = _assign_labels(region, seeds)
    points = []
    for label in range(1, len(seeds) + 1):
        part = labels == label
        area = int(part.sum())
        if area < 2:
            continue
        support = part & colored.astype(bool)
        weights = evidence[support]
        ys, xs = np.nonzero(support)
        if weights.sum() > 0:
            cx = float(np.dot(xs, weights) / weights.sum())
            cy = float(np.dot(ys, weights) / weights.sum())
            confidence = float(weights.mean())
        else:
            ys, xs = np.nonzero(part)
            cx, cy = float(xs.mean()), float(ys.mean())
            confidence = float(evidence[part].mean())
        radius = min(dmax, max(dmin * .35, math.sqrt(area / math.pi)))
        if radius * 2 < dmin * .75:
            continue
        points.append({'id': new_id(), 'pixel': [cx, cy], 'radius': float(radius),
                       'confidence': confidence, 'source': 'auto'})
    return points


def detect_points(image, settings, exclusions, roi=None):
    evidence = evidence_map(image, settings['color'], settings['tolerance'])
    mask = (evidence >= settings['min_confidence']).astype(np.uint8)
    height, width = mask.shape
    for exclusion in exclusions or []:
        x, y, rw, rh = [int(round(v)) for v in exclusion['rect']]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + rw), min(height, y + rh)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    if roi is not None:
        x, y, rw, rh = [int(round(v)) for v in roi]
        keep = np.zeros_like(mask)
        keep[max(0, y):min(height, y + rh), max(0, x):min(width, x + rw)] = 1
        mask &= keep
    # Close small anti-alias gaps before filling; the kernel stays well below
    # marker size so distinct (even touching) markers are not welded together.
    kernel = max(1, min(3, round(settings['point_min'] / 6)))
    if kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((kernel, kernel), np.uint8))
    colored = mask.copy()
    filled = fill_holes(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
    dmin, dmax = settings['point_min'], settings['point_max']
    area_min = math.pi * (dmin / 2) ** 2 * .32
    area_max = math.pi * (dmax / 2) ** 2 * 6.
    points = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < area_min or area > area_max or max(bw, bh) > dmax * 2.6:
            continue
        # Markers are compact and roughly circular; thin curve fragments and
        # mixed anti-alias fringes stay small, elongated or sparsely filled.
        # Hollow markers (including gapped rings) keep their interior hole.
        if max(bw, bh) < dmin * .75 or min(bw, bh) < dmin * .55:
            continue
        part = labels == label
        if area / float(bw * bh) < .62 and not _is_ring(part, colored):
            continue
        points.extend(_component_points(part, colored, evidence, dmin, dmax))
    if roi is not None:
        rx, ry, rw, rh = roi
        points = [q for q in points
                  if rx - 1 <= q['pixel'][0] <= rx + rw + 1
                  and ry - 1 <= q['pixel'][1] <= ry + rh + 1]
    # NMS across neighboring blobs that survived as separate components.
    points.sort(key=lambda q: -q['confidence'])
    kept = []
    for q in points:
        if all(math.hypot(q['pixel'][0] - k['pixel'][0],
                          q['pixel'][1] - k['pixel'][1]) > dmin * .55 for k in kept):
            kept.append(q)
    kept.sort(key=lambda q: (q['pixel'][1], q['pixel'][0]))
    return kept


def scatter_rows(points, calibration):
    if not points:
        return []
    coords = Coordinates(calibration)
    pixels = np.asarray([q['pixel'] for q in points], dtype=float)
    data = coords.pixels_to_data(pixels)
    rows = []
    for i, point in enumerate(points):
        rows.append({'point_id': point['id'], 'segment': '',
                     'x': float(data[i, 0]), 'y': float(data[i, 1]),
                     'pixel_x': float(pixels[i, 0]), 'pixel_y': float(pixels[i, 1]),
                     'confidence': float(point.get('confidence', 0.) or 0.),
                     'source': point.get('source', 'auto')})
    return rows


def point_overlay(image, calibration, series):
    """Numbered markers: row order matches the CSV order for this series."""
    result = image[:, :, :3].copy()
    basis = np.rint([calibration['x_end'], calibration['origin'],
                     calibration['y_end']]).astype(np.int32)
    cv2.polylines(result, [basis], False, (255, 150, 25), 2, cv2.LINE_AA)
    for exclusion in series.get('exclusions', []):
        x, y, w, h = [int(round(v)) for v in exclusion['rect']]
        cv2.rectangle(result, (x, y), (x + w, y + h), (225, 60, 60), 2)
        cv2.line(result, (x, y), (x + w, y + h), (225, 60, 60), 1)
        cv2.line(result, (x + w, y), (x, y + h), (225, 60, 60), 1)
    for index, point in enumerate(series.get('points', []), 1):
        x, y = [int(round(v)) for v in point['pixel']]
        radius = int(round(float(point.get('radius', 6.) or 6.))) + 2
        color = (247, 154, 38) if point.get('source') == 'manual' else (0, 178, 150)
        cv2.circle(result, (x, y), radius, color, 2, cv2.LINE_AA)
        cv2.drawMarker(result, (x, y), color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.putText(result, str(index), (x + radius + 2, y - radius),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)
    return result
