"""Validated tracing settings shared by storage, tracing and delivery."""
from image_lab.validation import Invalid
from .coordinates import finite

# The band is a separate optional colour layer.  Its defaults are chosen so that
# enabling a band never changes a centre-only series until the user edits them.
BAND_KEYS = ('tolerance', 'min_confidence', 'gap_limit', 'vertical_gap', 'occlusion_gap')


def band_defaults():
    return {'enabled': False, 'color': [245, 190, 130], 'tolerance': 18.,
            'min_confidence': .30, 'gap_limit': 6, 'vertical_gap': 6,
            'occlusion_gap': 12}


def defaults():
    return {'color': [35, 100, 210], 'tolerance': 24., 'width': 480, 'height': 320,
            'max_step': 20, 'smoothness': 2., 'min_confidence': .25, 'gap_limit': 8,
            'samples': 100, 'band': band_defaults()}


def color(value, label):
    value = finite(value, label)
    if value.shape != (3,) or value.min() < 0 or value.max() > 255:
        raise Invalid(label + '需要三个0–255通道')
    return value.tolist()


def band(value):
    b = {**band_defaults(), **(value if isinstance(value, dict) else {})}
    if not isinstance(b['enabled'], bool):
        raise Invalid('阴影带开关必须是布尔值')
    for key, lower, upper in [('gap_limit', 0, 80), ('vertical_gap', 0, 80),
                              ('occlusion_gap', 0, 80)]:
        if not isinstance(b[key], int) or isinstance(b[key], bool) or not lower <= b[key] <= upper:
            raise Invalid('阴影带 ' + key + ' 超出范围')
    for key, lower, upper in [('tolerance', 1, 100), ('min_confidence', .01, .99)]:
        b[key] = float(finite(b[key], '阴影带 ' + key))
        if not lower <= b[key] <= upper:
            raise Invalid('阴影带 ' + key + ' 超出范围')
    b['color'] = color(b['color'], '阴影颜色')
    return b


def settings(value):
    value = value if isinstance(value, dict) else {}
    s = {**defaults(), **value}
    for key, lower, upper in [('width', 64, 1200), ('height', 64, 800), ('max_step', 1, 80),
                              ('gap_limit', 0, 80), ('samples', 10, 2000)]:
        if not isinstance(s[key], int) or isinstance(s[key], bool) or not lower <= s[key] <= upper:
            raise Invalid(key + ' 超出范围')
    for key, lower, upper in [('tolerance', 1, 100), ('smoothness', 0, 100),
                              ('min_confidence', .01, .99)]:
        s[key] = float(finite(s[key], key))
        if not lower <= s[key] <= upper:
            raise Invalid(key + ' 超出范围')
    s['color'] = color(s['color'], '颜色')
    s['band'] = band(s.get('band'))
    return s
