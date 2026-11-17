"""Named panels partition one source image into independently calibrated plots.

Panel regions are rectangles in source pixels.  A panel's region is also its
sampling boundary: tracing never looks outside it.  Regions may overlap (for
example over a shared legend), but the default ``exclude`` overlap policy keeps
the shared pixels out of *both* panels' evidence so one panel can never pick up
a neighbour's curve.  Setting a panel to ``include`` explicitly claims the
shared area for that panel.
"""
import copy
import hashlib
import json
import numpy as np
from image_lab.validation import Invalid
from .coordinates import default_calibration, finite

MIN_SIZE = 24
MAX_PANELS = 8
OVERLAP_POLICIES = ('exclude', 'include')


def default_region(width, height):
    x, y = int(round(width * .06)), int(round(height * .06))
    return {'x': x, 'y': y, 'w': max(MIN_SIZE, width - 2 * x),
            'h': max(MIN_SIZE, height - 2 * y)}


def calibration_for(region, x_range=(0., 10.), y_range=(0., 100.)):
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    return {'origin': [x, y + h], 'x_end': [x + w, y + h], 'y_end': [x, y],
            'x': {'min': x_range[0], 'max': x_range[1], 'scale': 'linear'},
            'y': {'min': y_range[0], 'max': y_range[1], 'scale': 'linear'}}


def clamp_region(value, width, height):
    """Fit a rectangle to the image, preserving its size where possible."""
    if not isinstance(value, dict):
        raise Invalid('面板区域必须是对象')
    if width < MIN_SIZE or height < MIN_SIZE:
        raise Invalid('源图过小，无法划分面板')
    values = {}
    for key in ('x', 'y', 'w', 'h'):
        point = finite(value.get(key), '面板区域 ' + key)
        if point.shape != ():
            raise Invalid('面板区域需要四个数字')
        values[key] = int(round(float(point)))
    w = min(max(MIN_SIZE, values['w']), width)
    h = min(max(MIN_SIZE, values['h']), height)
    x = min(max(0, values['x']), width - w)
    y = min(max(0, values['y']), height - h)
    return {'x': x, 'y': y, 'w': w, 'h': h}


def validate_region(value, width, height):
    region = clamp_region(value, width, height)
    raw = finite([value.get(k) for k in ('x', 'y', 'w', 'h')], '面板区域').ravel()
    if raw[2] < MIN_SIZE or raw[3] < MIN_SIZE:
        raise Invalid('面板区域至少需要 24×24 像素')
    if raw[2] > width or raw[3] > height:
        raise Invalid('面板区域不能大于源图')
    return region


def rect_mask(region, width, height):
    x0 = max(0, min(width, int(round(region['x']))))
    y0 = max(0, min(height, int(round(region['y']))))
    x1 = max(x0, min(width, int(round(region['x'] + region['w']))))
    y1 = max(y0, min(height, int(round(region['y'] + region['h']))))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def panel_masks(panels, width, height):
    """Per-panel sampling masks; shared pixels are excluded unless claimed."""
    rectangles = [rect_mask(p['region'], width, height) for p in panels]
    result = []
    for index, panel in enumerate(panels):
        owned = rectangles[index].copy()
        if panel.get('overlap', 'exclude') != 'include':
            for other, rectangle in enumerate(rectangles):
                if other != index:
                    owned &= ~rectangle
        result.append(owned)
    return result


def panel_index(panels, pid):
    for index, panel in enumerate(panels):
        if panel['id'] == pid:
            return index
    raise Invalid('面板不存在')


def apply_anchor_states(panels, index, anchors):
    """Copy anchors with the state each one has against the current region."""
    result = []
    for anchor in anchors:
        anchor = dict(anchor)
        anchor['state'] = point_state(panels, index, anchor['pixel'])
        result.append(anchor)
    return result


def point_state(panels, index, pixel):
    """Whether a source-pixel proof point is sampled by the current panel."""
    panel = panels[index]
    x, y = float(pixel[0]), float(pixel[1])
    region = panel['region']
    inside = (region['x'] <= x < region['x'] + region['w']
              and region['y'] <= y < region['y'] + region['h'])
    if not inside:
        return 'outside'
    for other, neighbour in enumerate(panels):
        if other == index:
            continue
        area = neighbour['region']
        if (area['x'] <= x < area['x'] + area['w']
                and area['y'] <= y < area['y'] + area['h']):
            return 'inside' if panel.get('overlap', 'exclude') == 'include' else 'overlap'
    return 'inside'


def applicable_anchors(panels, index, anchors):
    return [a for a in anchors if point_state(panels, index, a['pixel']) == 'inside']


def _canonical_numbers(value):
    """Normalise 548 and 548.0 so a moved-and-restored panel compares equal."""
    if isinstance(value, dict):
        return {k: _canonical_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_numbers(v) for v in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 6)
    return value


def panel_signature(panel):
    """Stable comparison key for "does this stored result still apply?"."""
    body = {'region': panel.get('region'), 'calibration': panel.get('calibration'),
            'overlap': panel.get('overlap', 'exclude'),
            'series': [{'id': s['id'], 'settings': s.get('settings'),
                        'anchors': [list(a['pixel']) for a in s.get('anchors', [])]}
                       for s in panel.get('series', [])]}
    return hashlib.sha256(json.dumps(_canonical_numbers(body), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:24]


def decorate(document, width, height):
    """Attach sampling and applicability facts used by the interface."""
    document = copy.deepcopy(document)
    panels = document.get('panels', [])
    for index, panel in enumerate(panels):
        panel.setdefault('overlap', 'exclude')
        counts = {'inside': 0, 'overlap': 0, 'outside': 0}
        for series in panel.get('series', []):
            series['anchors'] = apply_anchor_states(panels, index, series.get('anchors', []))
            for anchor in series['anchors']:
                counts[anchor['state']] += 1
        panel['anchor_status'] = counts
        panel['signature'] = panel_signature(panel)
    signatures = {p['id']: p['signature'] for p in panels}
    legacy = {legacy_id(document.get('id')): panels[0]['signature']} if len(panels) == 1 else {}
    for entry in document.get('results', []):
        recorded = entry.get('panels')
        if not recorded:
            # Legacy single-panel results belonged to the whole-image panel.
            entry['panel_states'] = [{'id': next(iter(legacy), ''),
                                      'name': panels[0]['name'] if panels else '',
                                      'state': 'fresh' if legacy else 'orphan'}]
            entry['state'] = 'legacy'
            continue
        panel_states = []
        for item in recorded:
            if item.get('id') not in signatures:
                state = 'orphan'
            elif item.get('signature') != signatures[item['id']]:
                state = 'stale'
            else:
                state = 'fresh'
            panel_states.append({'id': item.get('id'), 'name': item.get('name', ''),
                                 'state': state})
        entry['panel_states'] = panel_states
        entry['state'] = ('fresh' if panel_states and all(x['state'] == 'fresh' for x in panel_states)
                          else 'mixed' if panel_states else 'legacy')
    return document


def legacy_id(project_id, index=0):
    return hashlib.sha256((str(project_id) + ':legacy-panel:' + str(index)).encode()).hexdigest()[:24]


def normalise(document, width, height):
    """Return a canonical panel document, migrating legacy single-panel files.

    A legacy project stored ``calibration`` and ``series`` at the top level.  It
    becomes one panel whose region covers the whole source image, so its tracing
    behaviour is unchanged.
    """
    document = copy.deepcopy(document)
    if isinstance(document.get('panels'), list) and document['panels']:
        document.setdefault('active_panel', document['panels'][0]['id'])
        for order, panel in enumerate(document['panels']):
            panel.setdefault('id', legacy_id(document.get('id'), order))
            panel.setdefault('name', '面板 %d' % (order + 1))
            panel.setdefault('order', order)
            panel.setdefault('overlap', 'exclude')
            panel.setdefault('series', [])
            panel.setdefault('region', default_region(width, height))
            panel.setdefault('calibration', calibration_for(panel['region']))
            panel['order'] = order
        document['width'], document['height'] = int(width), int(height)
        if document.get('active_panel') not in {p['id'] for p in document['panels']}:
            document['active_panel'] = document['panels'][0]['id']
        return document
    if 'calibration' not in document and 'series' not in document:
        return document
    panel_id = legacy_id(document.get('id'))
    region = {'x': 0, 'y': 0, 'w': int(width), 'h': int(height)}
    document['panels'] = [{
        'id': panel_id, 'name': '面板 1', 'order': 0, 'region': region,
        'calibration': copy.deepcopy(document.pop('calibration',
                                                  default_calibration(width, height))),
        'series': copy.deepcopy(document.pop('series', [])), 'overlap': 'exclude'}]
    document['active_panel'] = panel_id
    document['width'], document['height'] = int(width), int(height)
    return document
