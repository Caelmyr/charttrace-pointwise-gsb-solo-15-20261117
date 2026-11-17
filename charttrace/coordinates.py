"""Chart coordinates use an oblique pixel basis and independently calibrated axes."""
import copy
import numpy as np
from image_lab.validation import Invalid


def finite(value, name):
    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        raise Invalid(name + ' 必须是有限数字')
    if not np.isfinite(a).all():
        raise Invalid(name + ' 必须是有限数字')
    return a


def default_calibration(width, height):
    return {'origin': [width*.12, height*.85], 'x_end': [width*.9,height*.85],
            'y_end': [width*.12,height*.12], 'x': {'min':0.,'max':10.,'scale':'linear'},
            'y': {'min':0.,'max':100.,'scale':'linear'}}


def validate(value):
    c = copy.deepcopy(value)
    for key in ('origin','x_end','y_end'):
        point = finite(c.get(key), key)
        if point.shape != (2,):
            raise Invalid('标定点需要两个像素坐标')
        c[key] = point.tolist()
    matrix = np.column_stack((np.array(c['x_end'])-c['origin'],np.array(c['y_end'])-c['origin']))
    if abs(np.linalg.det(matrix)) < 4 or np.linalg.cond(matrix) > 1000:
        raise Invalid('坐标轴近乎共线或长度过短')
    for key in ('x','y'):
        axis = c[key]
        axis['min'],axis['max'] = [float(finite(axis[n], n)) for n in ('min','max')]
        if axis['scale'] not in ('linear','log') or axis['min']==axis['max']:
            raise Invalid('坐标范围或比例无效')
        if axis['scale']=='log' and min(axis['min'],axis['max'])<=0:
            raise Invalid('对数轴端点必须为正')
    return c


def axis_forward(values, axis):
    values = finite(values, '数据')
    limits = np.array([axis['min'],axis['max']])
    if axis['scale']=='log':
        if (values<=0).any():
            raise Invalid('对数轴上的值必须为正')
        values,limits=np.log10(values),np.log10(limits)
    return (values-limits[0])/(limits[1]-limits[0])


def axis_inverse(values, axis):
    limits=np.array([axis['min'],axis['max']])
    if axis['scale']=='log':
        limits=np.log10(limits)
    result=limits[0]+np.asarray(values)*(limits[1]-limits[0])
    return np.power(10.,result) if axis['scale']=='log' else result


class Coordinates:
    def __init__(self, calibration):
        self.c=validate(calibration)
        self.origin=np.array(self.c['origin'])
        self.basis=np.column_stack((np.array(self.c['x_end'])-self.origin,
                                    np.array(self.c['y_end'])-self.origin))
        self.inverse=np.linalg.inv(self.basis)

    def pixels_to_chart(self, points):
        p=finite(points,'像素点').reshape((-1,2))
        return (p-self.origin)@self.inverse.T

    def chart_to_pixels(self, points):
        return np.asarray(points).reshape((-1,2))@self.basis.T+self.origin

    def chart_to_data(self, points):
        p=np.asarray(points).reshape((-1,2))
        return np.column_stack((axis_inverse(p[:,0],self.c['x']),axis_inverse(p[:,1],self.c['y'])))

    def data_to_chart(self, points):
        p=np.asarray(points).reshape((-1,2))
        return np.column_stack((axis_forward(p[:,0],self.c['x']),axis_forward(p[:,1],self.c['y'])))

    def pixels_to_data(self, points):
        return self.chart_to_data(self.pixels_to_chart(points))

    def data_to_pixels(self, points):
        return self.chart_to_pixels(self.data_to_chart(points))

    def grid(self, width, height):
        x,y=np.meshgrid(np.linspace(0,1,width),np.linspace(1,0,height))
        return self.chart_to_pixels(np.column_stack((x.ravel(),y.ravel()))).reshape(height,width,2)
