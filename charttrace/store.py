"""Editable panel documents and immutable captures are kept separately."""
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
from .coordinates import Coordinates, default_calibration, validate, finite
from . import panels as panel_model


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
    return {'color':[35,100,210],'tolerance':24.,'width':480,'height':320,
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


def move_calibration(calibration,old,new):
    """Keep a panel's axes attached to its content when the region moves/resizes."""
    scale_x=new['w']/old['w']
    scale_y=new['h']/old['h']
    moved=copy.deepcopy(calibration)
    for key in ('origin','x_end','y_end'):
        x,y=moved[key]
        moved[key]=[new['x']+(x-old['x'])*scale_x,new['y']+(y-old['y'])*scale_y]
    return validate(moved)


class Store:
    def __init__(self,root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock()

    def directory(self,key):
        if not isinstance(key,str) or len(key)!=24 or any(c not in '0123456789abcdef' for c in key):
            raise Invalid('工程编号无效')
        return self.root/key

    def _raw(self,key):
        path=self.directory(key)/'project.json'
        if not path.exists():
            raise Invalid('工程不存在')
        return json.loads(path.read_text())

    def _dimensions(self,p):
        if p.get('width') and p.get('height'):
            return int(p['width']),int(p['height'])
        if p.get('source'):
            image=decode((self.directory(p['id'])/p['source']).read_bytes())
            return image.shape[1],image.shape[0]
        return 100,100

    def _canonical(self,p):
        width,height=self._dimensions(p)
        return panel_model.normalise(p,width,height)

    def read(self,key):
        p=self._raw(key)
        canonical=self._canonical(p)
        # Legacy single-panel files are migrated once, preserving the revision.
        if 'panels' not in p:
            self._write_raw(canonical)
        return canonical

    def _write_raw(self,p):
        directory=self.directory(p['id'])
        directory.mkdir(exist_ok=True)
        temp=directory/('write-'+uuid.uuid4().hex+'.json')
        temp.write_text(json.dumps(p,ensure_ascii=False,indent=2))
        os.replace(temp,directory/'project.json')

    def write(self,p):
        self._write_raw(self._canonical(p))

    def view(self,key):
        p=self.read(key)
        return panel_model.decorate(p,p['width'],p['height'])

    def listing(self):
        return [{'id':p['id'],'name':p['name'],'revision':p['revision'],
                 'series':sum(len(x['series']) for x in p['panels']),
                 'panels':len(p['panels']),
                 'current_result':p['current_result']}
                for p in [self.read(f.parent.name) for f in self.root.glob('*/project.json')]]

    def create(self,name,image,multi=False):
        from .delivery import demo,demo_panels
        with self.lock:
            source=demo_panels() if multi else demo()
            p={'id':identifier(),'name':str(name)[:100],'revision':1,'created':time.time(),
               'source':None,'panels':[],'active_panel':None,
               'versions':[],'results':[],'current_result':None,
               'width':source.shape[1],'height':source.shape[0]}
            self.write(p)
            p=self.replace_source(p['id'],1,source)
            if multi:
                return self._seed_multiple(p['id'])
            return self.add_series(p['id'],p['revision'],'蓝色曲线')

    def _seed_multiple(self,key):
        """Demonstration only: two differently scaled plots on one screenshot."""
        p=self.read(key)
        width,height=p['width'],p['height']
        layouts=[({'x':40,'y':60,'w':520,'h':470},(0.,10.),(0.,100.),'左图 蓝色曲线'),
                 ({'x':510,'y':60,'w':570,'h':470},(1.,100.),(0.,1.),'右图 红色曲线')]
        def operation(p):
            p['panels']=[]
            for index,(region,rx,ry,name) in enumerate(layouts):
                region=panel_model.validate_region(region,width,height)
                p['panels'].append({
                    'id':identifier(),'name':'面板 %d'%(index+1),'order':index,
                    'region':region,'overlap':'exclude',
                    'calibration':panel_model.calibration_for(region,rx,ry),
                    'series':[{'id':identifier(),'name':name,
                               'settings':{**defaults(),'color':[35,100,210] if index==0 else [200,55,65]},
                               'anchors':[]}]})
            p['active_panel']=p['panels'][0]['id']
            # The two demo panels deliberately differ in range and scale so their
            # independent calibration is visible immediately.
            p['panels'][1]['calibration']['x']={'min':1.,'max':100.,'scale':'log'}
        return self.mutate(key,p['revision'],operation)

    def revision(self,p,expected):
        if not isinstance(expected,int) or expected!=p['revision']:
            raise Conflict('工程已变化，请刷新')

    def mutate(self,key,expected,operation):
        with self.lock:
            p=self.read(key)
            self.revision(p,expected)
            operation(p)
            p=self._canonical(p)
            p['revision']+=1
            p['current_result']=None
            self.write(p)
            return p

    def replace_source(self,key,expected,image):
        raw=encode(image)
        filename=hashlib.sha256(raw).hexdigest()+'.png'
        width,height=image.shape[1],image.shape[0]
        def operation(p):
            (self.directory(key)/filename).write_bytes(raw)
            p['source']=filename
            p['width'],p['height']=width,height
            for panel in p.get('panels') or []:
                region=panel_model.clamp_region(panel.get('region') or
                                                panel_model.default_region(width,height),
                                                width,height)
                panel['region']=region
                # Keep each panel's own axis ranges; only refit the corner points
                # to the new image, then empty the now-unsampled proof points.
                old=panel.get('calibration') or {}
                calibration=panel_model.calibration_for(region)
                for axis in ('x','y'):
                    previous=old.get(axis) or {}
                    calibration[axis]={**calibration[axis],**previous}
                panel['calibration']=validate(calibration)
                for series in panel['series']:
                    series['anchors']=[]
            if not p.get('panels'):
                p['calibration']=default_calibration(width,height)
                p['series']=[]
        return self.mutate(key,expected,operation)

    def update(self,key,expected,data):
        def operation(p):
            if 'name' in data:
                p['name']=str(data['name'])[:100]
            if 'active_panel' in data:
                panel_model.panel_index(p['panels'],data['active_panel'])
                p['active_panel']=data['active_panel']
        return self.mutate(key,expected,operation)

    def _panel(self,p,pid):
        return p['panels'][panel_model.panel_index(p['panels'],pid)]

    def add_panel(self,key,expected,data):
        def operation(p):
            if len(p['panels'])>=panel_model.MAX_PANELS:
                raise Invalid('最多%d个面板'%panel_model.MAX_PANELS)
            width,height=p['width'],p['height']
            region=panel_model.validate_region(
                data.get('region') or panel_model.default_region(width,height),width,height)
            ranges=data.get('ranges') or {}
            x_range=ranges.get('x') or (0.,10.)
            y_range=ranges.get('y') or (0.,100.)
            calibration=panel_model.calibration_for(region,x_range,y_range)
            if data.get('calibration'):
                calibration=validate(data['calibration'])
            panel={'id':identifier(),
                   'name':str(data.get('name') or '面板 %d'%(len(p['panels'])+1))[:80],
                   'order':len(p['panels']),'region':region,
                   'overlap':data.get('overlap','exclude'),
                   'calibration':calibration,'series':[]}
            if panel['overlap'] not in panel_model.OVERLAP_POLICIES:
                raise Invalid('面板重叠策略无效')
            p['panels'].append(panel)
            p['active_panel']=panel['id']
        return self.mutate(key,expected,operation)

    def edit_panel(self,key,expected,pid,data):
        def operation(p):
            panel=self._panel(p,pid)
            if 'name' in data:
                panel['name']=str(data['name'])[:80]
            if 'overlap' in data:
                if data['overlap'] not in panel_model.OVERLAP_POLICIES:
                    raise Invalid('面板重叠策略无效')
                panel['overlap']=data['overlap']
            if 'region' in data:
                region=panel_model.validate_region(data['region'],p['width'],p['height'])
                if region!=panel['region']:
                    if 'calibration' not in data:
                        panel['calibration']=move_calibration(panel['calibration'],
                                                              panel['region'],region)
                    panel['region']=region
            if 'calibration' in data:
                panel['calibration']=validate(data['calibration'])
            if data.get('active'):
                p['active_panel']=panel['id']
        return self.mutate(key,expected,operation)

    def delete_panel(self,key,expected,pid):
        def operation(p):
            if len(p['panels'])<=1:
                raise Invalid('至少保留一个面板')
            index=panel_model.panel_index(p['panels'],pid)
            p['panels'].pop(index)
            for order,panel in enumerate(p['panels']):
                panel['order']=order
            if p.get('active_panel')==pid:
                p['active_panel']=p['panels'][max(0,index-1)]['id']
        return self.mutate(key,expected,operation)

    def add_series(self,key,expected,name,pid=None):
        def operation(p):
            panel=self._panel(p,pid or p['active_panel'])
            if len(panel['series'])>=8:
                raise Invalid('最多8条曲线')
            panel['series'].append({'id':identifier(),'name':str(name)[:80],
                                    'settings':defaults(),'anchors':[]})
        return self.mutate(key,expected,operation)

    def edit_series(self,key,expected,sid,data,pid=None):
        def operation(p):
            panel=self._panel(p,pid or p['active_panel'])
            s=next((x for x in panel['series'] if x['id']==sid),None)
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
        return self.mutate(key,expected,operation)

    def delete_series(self,key,expected,sid,pid=None):
        def operation(p):
            panel=self._panel(p,pid or p['active_panel'])
            panel['series']=[s for s in panel['series'] if s['id']!=sid]
        return self.mutate(key,expected,operation)

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
            snapshot={k:copy.deepcopy(p[k]) for k in ('source','panels','active_panel')}
            summary=[{'id':panel['id'],'name':panel['name'],
                      'region':copy.deepcopy(panel['region']),
                      'calibration':copy.deepcopy(panel['calibration']),
                      'series':len(panel['series'])} for panel in p['panels']]
            p['versions'].append({'id':identifier(),'label':str(label)[:80],
                                  'created':time.time(),'panels':summary,
                                  'document':snapshot})
            p['versions']=p['versions'][-20:]
            p['revision']+=1
            self.write(p)
            return p

    def restore(self,key,expected,vid):
        def operation(p):
            v=next((v for v in p['versions'] if v['id']==vid),None)
            if v is None:
                raise Invalid('版本不存在')
            for key in ('panels','active_panel','calibration','series'):
                p.pop(key,None)
            document=copy.deepcopy(v['document'])
            p.update(document)
        return self.mutate(key,expected,operation)

    def publish(self,captured,panels_series,overlay):
        with self.lock:
            p=self.read(captured['id'])
            if p['revision']!=captured['revision']:
                raise Conflict('提取期间工程已变化，旧结果未发布')
            rid=identifier()
            folder=self.directory(p['id'])/rid
            folder.mkdir()
            recorded=[]
            for panel,data in zip(captured['panels'],panels_series):
                recorded.append({'id':panel['id'],'name':panel['name'],
                                 'signature':panel_model.panel_signature(panel),
                                 'region':copy.deepcopy(panel['region']),
                                 'calibration':copy.deepcopy(panel['calibration']),
                                 'series':data['series']})
            flat=[]
            for entry in recorded:
                for series in entry['series']:
                    flat.append({**series,'panel':entry['id'],'panel_name':entry['name']})
            record={'id':rid,'revision':captured['revision'],'created':time.time(),
                    'source':captured['source'],'width':captured['width'],
                    'height':captured['height'],'panels':recorded,'series':flat}
            (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False))
            (folder/'overlay.png').write_bytes(encode(overlay))
            p['results'].append({'id':rid,'revision':record['revision'],
                                 'created':record['created'],
                                 'panels':[{'id':x['id'],'name':x['name'],
                                            'signature':x['signature']} for x in recorded]})
            p['current_result']=rid
            self.write(p)
            return record

    def result(self,key,rid):
        p=self.read(key)
        rid=rid or p['current_result']
        if not any(r['id']==rid for r in p['results']):
            raise Invalid('请先提取数据')
        folder=self.directory(key)/rid
        record=json.loads((folder/'result.json').read_text())
        entry=next((r for r in p['results'] if r['id']==rid),{})
        if 'panels' not in record:
            # Result written before panels existed: one whole-image panel whose
            # calibration and series are the legacy top-level fields.
            document=record.get('document') or {}
            panel_id=panel_model.legacy_id(p['id'])
            record['panels']=[{'id':panel_id,'name':p['panels'][0]['name'],
                               'signature':None,
                               'region':{'x':0,'y':0,'w':p['width'],'h':p['height']},
                               'calibration':copy.deepcopy(document.get('calibration')),
                               'series':copy.deepcopy(record.get('series',[]))}]
            record['width'],record['height']=p['width'],p['height']
            for series in record['series']:
                series.setdefault('panel',panel_id)
                series.setdefault('panel_name',record['panels'][0]['name'])
        view=panel_model.decorate(p,p['width'],p['height'])
        state=next((r for r in view['results'] if r['id']==rid),{})
        record['state']=state.get('state','legacy')
        record['panel_states']=state.get('panel_states',[])
        record['revision_available']=entry.get('revision')
        return record,folder
