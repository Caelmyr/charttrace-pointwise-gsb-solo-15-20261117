"""Editable chart documents and immutable captures are kept separately."""
import copy
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from image_lab.validation import Invalid, Conflict
from .coordinates import default_calibration, validate, finite
from .scatter import scatter_defaults, detect_points, new_id


def identifier():
    return uuid.uuid4().hex[:24]


def decode(raw):
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim!=3 or image.shape[2] not in (3,4):
        raise Invalid('需要RGB/RGBA PNG或JPEG图表')
    if max(image.shape[:2])>4096:
        raise Invalid('图像最长边不能超过4096')
    if image.shape[2]==3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
    return cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA)


def encode(image):
    code=cv2.COLOR_RGBA2BGRA if image.shape[2]==4 else cv2.COLOR_RGB2BGR
    ok,raw=cv2.imencode('.png',cv2.cvtColor(image,code))
    if not ok:
        raise Invalid('图像编码失败')
    return raw.tobytes()


def defaults():
    return {'kind':'curve','color':[35,100,210],'tolerance':24.,'width':480,'height':320,
            'max_step':20,'smoothness':2.,'min_confidence':.25,'gap_limit':8,'samples':100}


def settings(value):
    s={**defaults(),**value}
    for key,lower,upper in [('width',64,1200),('height',64,800),('max_step',1,80),
                            ('gap_limit',0,80),('samples',10,2000)]:
        if not isinstance(s[key],int) or not lower<=s[key]<=upper:
            raise Invalid(key+' 超出范围')
    for key,lower,upper in [('tolerance',1,100),('smoothness',0,100),('min_confidence',.01,.99)]:
        s[key]=float(finite(s[key],key))
        if not lower<=s[key]<=upper:
            raise Invalid(key+' 超出范围')
    color=finite(s['color'],'颜色')
    if color.shape!=(3,) or color.min()<0 or color.max()>255:
        raise Invalid('颜色需要三个0–255通道')
    s['color']=color.tolist()
    return s


def scatter_settings(value):
    s={**scatter_defaults(),**(value or {})}
    if s['kind']!='scatter':
        raise Invalid('散点系列类型无效')
    for key,lower,upper in [('point_min',3,200),('point_max',3,400)]:
        if not isinstance(s[key],(int,float)) or not lower<=s[key]<=upper:
            raise Invalid(key+' 超出范围')
        s[key]=float(s[key])
    if s['point_min']>s['point_max']:
        raise Invalid('点尺寸范围无效：最小不能大于最大')
    for key,lower,upper in [('tolerance',1,100),('min_confidence',.01,.99)]:
        s[key]=float(finite(s[key],key))
        if not lower<=s[key]<=upper:
            raise Invalid(key+' 超出范围')
    color=finite(s['color'],'颜色')
    if color.shape!=(3,) or color.min()<0 or color.max()>255:
        raise Invalid('颜色需要三个0–255通道')
    s['color']=color.tolist()
    return s


def check_points(value, image_shape):
    if not isinstance(value,list) or len(value)>500:
        raise Invalid('散点最多500个')
    points=[]
    height,width=image_shape[:2]
    for item in value:
        if not isinstance(item,dict):
            raise Invalid('散点格式无效')
        point=finite(item.get('pixel'),'散点')
        if point.shape!=(2,):
            raise Invalid('散点需要两个像素坐标')
        if not(-50<=point[0]<=width+50 and -50<=point[1]<=height+50):
            raise Invalid('散点超出源图范围')
        radius=item.get('radius',6.)
        if not isinstance(radius,(int,float)) or not 0.5<=float(radius)<=200:
            raise Invalid('点半径无效')
        confidence=item.get('confidence',0.)
        if not isinstance(confidence,(int,float)) or not 0<=float(confidence)<=1:
            raise Invalid('点置信度无效')
        source=item.get('source','manual')
        if source not in ('auto','manual'):
            raise Invalid('点来源无效')
        points.append({'id':str(item.get('id') or new_id())[:24],
                       'pixel':point.tolist(),'radius':float(radius),
                       'confidence':float(confidence),'source':source})
    return points


def check_exclusions(value, image_shape):
    if not isinstance(value,list) or len(value)>50:
        raise Invalid('排除区最多50个')
    height,width=image_shape[:2]
    result=[]
    for item in value:
        if not isinstance(item,dict) or not isinstance(item.get('rect'),list) or len(item['rect'])!=4:
            raise Invalid('排除区需要 x,y,宽,高 四个数')
        rect=finite(item['rect'],'排除区')
        x,y,w,h=rect.tolist()
        if w<=0 or h<=0 or x+w<0 or y+h<0 or x>width or y>height:
            raise Invalid('排除区超出源图范围')
        result.append({'id':str(item.get('id') or new_id())[:24],
                       'rect':[x,y,float(w),float(h)]})
    return result


class Store:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()

    def directory(self,key):
        if not isinstance(key,str) or len(key)!=24 or any(c not in '0123456789abcdef' for c in key):
            raise Invalid('工程编号无效')
        return self.root/key

    def read(self,key):
        path=self.directory(key)/'project.json'
        if not path.exists():
            raise Invalid('工程不存在')
        return json.loads(path.read_text())

    def write(self,p):
        directory=self.directory(p['id'])
        directory.mkdir(exist_ok=True)
        temp=directory/('write-'+uuid.uuid4().hex+'.json')
        temp.write_text(json.dumps(p,ensure_ascii=False,indent=2))
        os.replace(temp,directory/'project.json')

    def listing(self):
        return [{'id':p['id'],'name':p['name'],'revision':p['revision'],
                 'series':len(p['series']),'current_result':p['current_result']}
                for p in [json.loads(f.read_text()) for f in self.root.glob('*/project.json')]]

    def create(self,name,image):
        with self.lock:
            p={'id':identifier(),'name':str(name)[:100],'revision':1,'created':time.time(),
               'source':None,'calibration':default_calibration(image.shape[1],image.shape[0]),
               'series':[],'versions':[],'results':[],'current_result':None}
            self.write(p)
            p=self.replace_source(p['id'],1,image)
            return self.add_series(p['id'],p['revision'],'蓝色曲线')

    def revision(self,p,expected):
        if not isinstance(expected,int) or expected!=p['revision']:
            raise Conflict('工程已变化，请刷新')

    def mutate(self,key,expected,operation):
        with self.lock:
            p=self.read(key)
            self.revision(p,expected)
            operation(p)
            p['revision']+=1
            p['current_result']=None
            self.write(p)
            return p

    def replace_source(self,key,expected,image):
        raw=encode(image)
        filename=hashlib.sha256(raw).hexdigest()+'.png'
        def operation(p):
            (self.directory(key)/filename).write_bytes(raw)
            p['source']=filename
            p['calibration']=default_calibration(image.shape[1],image.shape[0])
            for series in p['series']:
                series['anchors']=[]
                series['points']=[]
                series['exclusions']=[]
        return self.mutate(key,expected,operation)

    def update(self,key,expected,data):
        def operation(p):
            if 'name' in data:
                p['name']=str(data['name'])[:100]
            if 'calibration' in data:
                p['calibration']=validate(data['calibration'])
        return self.mutate(key,expected,operation)

    def add_series(self,key,expected,name,kind='curve'):
        def operation(p):
            if len(p['series'])>=8:
                raise Invalid('最多8条曲线')
            if kind=='scatter':
                p['series'].append({'id':identifier(),'name':str(name)[:80],
                                    'settings':scatter_defaults(),'anchors':[],
                                    'points':[],'exclusions':[]})
            elif kind=='curve':
                p['series'].append({'id':identifier(),'name':str(name)[:80],
                                    'settings':defaults(),'anchors':[]})
            else:
                raise Invalid('采集类型无效')
        return self.mutate(key,expected,operation)

    def edit_series(self,key,expected,sid,data,image_shape=None):
        def operation(p):
            s=next((x for x in p['series'] if x['id']==sid),None)
            if s is None:
                raise Invalid('曲线不存在')
            kind=s['settings'].get('kind','curve')
            if kind=='scatter':
                if 'settings' in data:
                    s['settings']=scatter_settings({**s['settings'],**data['settings']})
            elif 'settings' in data:
                s['settings']=settings({**s['settings'],**data['settings']})
            if 'name' in data:
                s['name']=str(data['name'])[:80]
            if 'anchors' in data:
                if len(data['anchors'])>200:
                    raise Invalid('最多200个手动点')
                anchors=[]
                for item in data['anchors']:
                    point=finite(item['pixel'],'手动点')
                    if point.shape!=(2,):
                        raise Invalid('手动点需要两个像素坐标')
                    anchors.append({'id':item.get('id',identifier()),'pixel':point.tolist()})
                s['anchors']=anchors
            if kind=='scatter':
                shape=image_shape or self.source(p).shape
                if 'points' in data:
                    s['points']=check_points(data['points'],shape)
                elif 'points' not in s:
                    s['points']=[]
                if 'exclusions' in data:
                    s['exclusions']=check_exclusions(data['exclusions'],shape)
                elif 'exclusions' not in s:
                    s['exclusions']=[]
        return self.mutate(key,expected,operation)

    def delete_series(self,key,expected,sid):
        return self.mutate(key,expected,lambda p:p.update(series=[s for s in p['series'] if s['id']!=sid]))

    def source(self,p):
        return decode((self.directory(p['id'])/p['source']).read_bytes())

    def capture(self,key):
        with self.lock:
            p=self.read(key)
            return copy.deepcopy(p),self.source(p)

    def detect(self,key,expected,sid,roi):
        """Run marker detection; proofread points are stored in source pixels."""
        with self.lock:
            p=self.read(key)
            self.revision(p,expected)
            series=next((x for x in p['series'] if x['id']==sid),None)
            if series is None:
                raise Invalid('曲线不存在')
            if series['settings'].get('kind')!='scatter':
                raise Invalid('当前不是散点系列')
            image=self.source(p)
            found=detect_points(image,series['settings'],series.get('exclusions',[]),roi)
            current=series.get('points',[])
            if roi is None:
                merged=found
            else:
                # ROI detection only re-interprets the box: every old point
                # inside the box is removed before NMS so a merged blob can
                # legitimately split into several new markers.
                rx,ry,rw,rh=roi
                def inside(q):
                    return rx-1<=q['pixel'][0]<=rx+rw+1 and ry-1<=q['pixel'][1]<=ry+rh+1
                outside=[q for q in current if not inside(q)]
                merged=outside+found
            series['points']=check_points(merged,image.shape)
            p['revision']+=1
            p['current_result']=None
            self.write(p)
            return copy.deepcopy(p),copy.deepcopy(series['points'])

    def snapshot(self,key,expected,label):
        with self.lock:
            p=self.read(key)
            self.revision(p,expected)
            snapshot={k:copy.deepcopy(p[k]) for k in ('source','calibration','series')}
            p['versions'].append({'id':identifier(),'label':str(label)[:80],'document':snapshot})
            p['versions']=p['versions'][-20:]
            p['revision']+=1
            self.write(p)
            return p

    def restore(self,key,expected,vid):
        def operation(p):
            v=next((v for v in p['versions'] if v['id']==vid),None)
            if v is None:
                raise Invalid('版本不存在')
            p.update(copy.deepcopy(v['document']))
        return self.mutate(key,expected,operation)

    def publish(self,captured,series,overlay):
        with self.lock:
            p=self.read(captured['id'])
            if p['revision']!=captured['revision']:
                raise Conflict('提取期间工程已变化，旧结果未发布')
            rid=identifier()
            folder=self.directory(p['id'])/rid
            folder.mkdir()
            record={'id':rid,'revision':captured['revision'],'created':time.time(),
                    'document':{k:copy.deepcopy(captured[k]) for k in ('source','calibration','series')},
                    'series':series}
            (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False))
            (folder/'overlay.png').write_bytes(encode(overlay))
            p['results'].append({'id':rid,'revision':record['revision'],'created':record['created']})
            p['current_result']=rid
            self.write(p)
            return record

    def result(self,key,rid):
        p=self.read(key)
        rid=rid or p['current_result']
        if not any(r['id']==rid for r in p['results']):
            raise Invalid('请先提取数据')
        folder=self.directory(key)/rid
        return json.loads((folder/'result.json').read_text()),folder
