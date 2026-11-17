import csv
import io
import json
import zipfile
import numpy as np
import cv2
from image_lab.validation import Invalid
from .tracing import trace,sample_paths,overlays
from .bars import default_bars,extract_bars,bars_csv_bytes


def extract(document,image):
    series=[]
    preview=image[:,:,:3].copy()
    for s in document['series']:
        paths,rgb,evidence=trace(image,document['calibration'],s['settings'],s['anchors'])
        rows=sample_paths(paths,document['calibration'],s['settings']['samples'])
        series.append({'id':s['id'],'name':s['name'],'paths':paths,'rows':rows,
                       'observed_columns':int(sum(sum(p['observed']) for p in paths)),
                       'mean_confidence':float(np.mean([r['confidence'] for r in rows])) if rows else 0})
        preview=overlays(np.dstack((preview,image[:,:,3])),document['calibration'],paths,s['anchors'])
    config=document.get('bars') or default_bars()
    bars=None
    if config.get('items') and config.get('colors'):
        bars,preview=extract_bars(image,document['calibration'],config,preview)
    if not series and bars is None:
        raise Invalid('请先添加曲线或配置堆叠柱')
    return series,bars,preview


def csv_bytes(series):
    output=io.StringIO(newline='')
    writer=csv.writer(output)
    writer.writerow(['series','segment','x','y','pixel_x','pixel_y','confidence'])
    for s in series:
        for row in s['rows']:
            writer.writerow([s['name']]+[row[k] for k in ['segment','x','y','pixel_x','pixel_y','confidence']])
    return output.getvalue().encode('utf-8-sig')


def bundle(record,folder):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.csv',csv_bytes(record['series']))
        if record.get('bars'):
            archive.writestr('bars.csv',bars_csv_bytes(record['bars']))
        archive.writestr('extraction.json',json.dumps(record,ensure_ascii=False,indent=2))
        archive.write(folder/'overlay.png','overlay.png')
    return output.getvalue()


def demo():
    image=np.full((540,840,4),255,np.uint8)
    # Synthesized demonstration, not measured experimental data.
    for x in range(100,761,66):
        cv2.line(image,(x,65),(x,460),(230,233,238,255),1)
    for y in range(65,461,79):
        cv2.line(image,(100,y),(760,y),(230,233,238,255),1)
    cv2.line(image,(100,460),(760,460),(50,60,75,255),2)
    cv2.line(image,(100,460),(100,65),(50,60,75,255),2)
    x=np.arange(100,761)
    y=330-120*np.sin((x-100)/210)-.11*(x-100)
    cv2.polylines(image,[np.column_stack((x,y)).astype(np.int32)],False,(35,100,210,255),3,cv2.LINE_AA)
    cv2.putText(image,'SYNTHETIC CHART / ChartTrace',(105,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,60,75,255),1)
    return image


def demo_bars():
    # Synthesized tilted stacked-bar demonstration, not measured experimental data.
    image=np.full((540,840,4),255,np.uint8)
    origin=np.array([90.,430.]); x_end=np.array([740.,430.]); y_end=np.array([90.,60.])
    ymin,ymax=-50.,100.
    def to_px(frac,value):
        return origin+frac*(x_end-origin)+(value-ymin)/(ymax-ymin)*(y_end-origin)
    for value in range(-50,101,25):
        a,b=to_px(0,value),to_px(1,value)
        cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),(232,235,240,255),1)
    for frac in np.arange(.1,1.01,.1):
        a,b=to_px(frac,-50),to_px(frac,100)
        cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),(232,235,240,255),1)
    cv2.line(image,tuple(origin.astype(int)),tuple(x_end.astype(int)),(50,60,75,255),2)
    cv2.line(image,tuple(origin.astype(int)),tuple(y_end.astype(int)),(50,60,75,255),2)
    a,b=to_px(0,0),to_px(1,0)
    cv2.line(image,tuple(a.astype(int)),tuple(b.astype(int)),(150,158,170,255),1)
    colors={'系列A':(66,135,245),'系列B':(245,166,35),'系列C':(96,196,128)}
    groups=[('G1',.14,[('系列A',30),('系列B',20),('系列C',25)],[]),
            ('G2',.30,[('系列A',45),('系列C',30)],[]),
            ('G3',.46,[('系列A',25),('系列B',35)],[('系列C',-20)]),
            ('G4',.62,[('系列C',40)],[('系列A',-30),('系列B',-15)]),
            ('G5',.78,[('系列A',50),('系列B',25),('系列C',15)],[])]
    width_px=54
    half=width_px/650/2
    border=(70,76,88,255)
    for name,frac,positive,negative in groups:
        for stack in (positive,negative):
            base=0.
            for cname,value in stack:
                quad=np.array([to_px(frac-half,base),to_px(frac+half,base),
                               to_px(frac+half,base+value),to_px(frac-half,base+value)],np.int32)
                cv2.fillPoly(image,[quad],colors[cname]+(255,))
                cv2.polylines(image,[quad],True,border,1)
                base+=value
    for i,(cname,rgb) in enumerate(colors.items()):
        cv2.rectangle(image,(762,92+i*26),(780,106+i*26),rgb+(255,),-1)
        cv2.putText(image,'Series '+chr(65+i),(788,104+i*26),cv2.FONT_HERSHEY_SIMPLEX,.42,(60,68,80,255),1)
    cv2.putText(image,'SYNTHETIC STACKED BARS / ChartTrace',(95,40),cv2.FONT_HERSHEY_SIMPLEX,.6,(50,60,75,255),1)
    height,width=image.shape[:2]
    matrix=cv2.getRotationMatrix2D((width/2,height/2),2.5,1.)
    image=cv2.warpAffine(image,matrix,(width,height),borderValue=(255,255,255,255))
    def tx(point):
        return (matrix@np.array([point[0],point[1],1.])).tolist()
    calibration={'origin':tx(origin),'x_end':tx(x_end),'y_end':tx(y_end),
                 'x':{'min':0.,'max':10.,'scale':'linear'},'y':{'min':-50.,'max':100.,'scale':'linear'}}
    bars={'zero_pixel':tx(to_px(.5,0)),'width_px':width_px,'tolerance':26.,'min_height_px':4,
          'colors':[{'name':n,'color':list(c)} for n,c in colors.items()],
          'items':[{'pixel':tx(to_px(frac,0)),'label':name} for name,frac,_,_ in groups]}
    return image,calibration,bars
