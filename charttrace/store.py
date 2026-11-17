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
from .settings import defaults as default_settings, settings as normalize_settings


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


defaults=default_settings
settings=normalize_settings


def normalize(p):
    """Fill settings added after older projects and captures were saved."""
    for series in p.get('series',[]):
        series['settings']=normalize_settings(series.get('settings'))
        series.setdefault('band_seeds',[])
    return p


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
        return normalize(json.loads(path.read_text()))

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
                series['band_seeds']=[]
        return self.mutate(key,expected,operation)

    def update(self,key,expected,data):
        def operation(p):
            if 'name' in data:
                p['name']=str(data['name'])[:100]
            if 'calibration' in data:
                p['calibration']=validate(data['calibration'])
        return self.mutate(key,expected,operation)

    def add_series(self,key,expected,name):
        def operation(p):
            if len(p['series'])>=8:
                raise Invalid('最多8条曲线')
            p['series'].append({'id':identifier(),'name':str(name)[:80],
                                'settings':defaults(),'anchors':[],'band_seeds':[]})
        return self.mutate(key,expected,operation)

    def edit_series(self,key,expected,sid,data):
        def operation(p):
            s=next((x for x in p['series'] if x['id']==sid),None)
            if s is None:
                raise Invalid('曲线不存在')
            if 'settings' in data:
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
            if 'band_seeds' in data:
                if not isinstance(data['band_seeds'],list) or len(data['band_seeds'])>40:
                    raise Invalid('阴影带校对点最多40个')
                seeds=[]
                for item in data['band_seeds']:
                    point=finite(item['pixel'],'阴影带校对点')
                    if point.shape!=(2,):
                        raise Invalid('阴影带校对点需要两个像素坐标')
                    seeds.append({'id':item.get('id',identifier()),'pixel':point.tolist()})
                s['band_seeds']=seeds
        return self.mutate(key,expected,operation)

    def delete_series(self,key,expected,sid):
        return self.mutate(key,expected,lambda p:p.update(series=[s for s in p['series'] if s['id']!=sid]))

    def source(self,p):
        return decode((self.directory(p['id'])/p['source']).read_bytes())

    def capture(self,key):
        with self.lock:
            p=self.read(key)
            return copy.deepcopy(p),self.source(p)

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
            document=normalize(copy.deepcopy(v['document']))
            p.update(document)
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
