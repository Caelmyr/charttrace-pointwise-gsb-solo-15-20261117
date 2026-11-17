import csv
import io
import json
import zipfile
import numpy as np
import cv2
from .tracing import trace,sample_paths,overlays
from .scatter import scatter_rows, point_overlay


def extract(document,image):
    series=[]
    preview=image[:,:,:3].copy()
    for s in document['series']:
        if s['settings'].get('kind')=='scatter':
            points=s.get('points',[])
            rows=scatter_rows(points,document['calibration'])
            series.append({'id':s['id'],'name':s['name'],'kind':'scatter',
                           'points':points,'rows':rows,
                           'observed_columns':len(points),
                           'mean_confidence':float(np.mean([q['confidence'] for q in points])) if points else 0})
            preview=point_overlay(np.dstack((preview,image[:,:,3])),
                                  document['calibration'],s)
            continue
        paths,rgb,evidence=trace(image,document['calibration'],s['settings'],s['anchors'])
        rows=sample_paths(paths,document['calibration'],s['settings']['samples'])
        series.append({'id':s['id'],'name':s['name'],'kind':'curve',
                       'paths':paths,'rows':rows,
                       'observed_columns':int(sum(sum(p['observed']) for p in paths)),
                       'mean_confidence':float(np.mean([r['confidence'] for r in rows])) if rows else 0})
        preview=overlays(np.dstack((preview,image[:,:,3])),document['calibration'],paths,s['anchors'])
    return series,preview


def csv_bytes(series):
    output=io.StringIO(newline='')
    writer=csv.writer(output)
    # point_id stays empty for curve rows; scatter rows map one-to-one to markers.
    writer.writerow(['series','kind','point_id','segment','x','y',
                     'pixel_x','pixel_y','confidence','source'])
    for s in series:
        kind=s.get('kind','curve')
        for row in s['rows']:
            if kind=='scatter':
                writer.writerow([s['name'],'scatter',row.get('point_id',''),row.get('segment',''),
                                 row['x'],row['y'],row['pixel_x'],row['pixel_y'],
                                 row['confidence'],row.get('source','auto')])
            else:
                writer.writerow([s['name'],'curve','',row['segment'],
                                 row['x'],row['y'],row['pixel_x'],row['pixel_y'],
                                 row['confidence'],'trace'])
    return output.getvalue().encode('utf-8-sig')


def bundle(record,folder):
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.csv',csv_bytes(record['series']))
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
    # Red scatter series: filled and hollow markers, including two overlapping
    # pairs and a same-color legend swatch that must be excluded from detection.
    xs=[140,215,290,365,440,515,590,665,720]
    ys=[405,360,330,300,275,255,235,220,210]
    red=(210,60,70,255)
    for i,(cx,cy) in enumerate(zip(xs,ys)):
        if i%2:
            cv2.circle(image,(cx,cy),9,red,2,cv2.LINE_AA)
        else:
            cv2.circle(image,(cx,cy),9,red,-1,cv2.LINE_AA)
    cv2.circle(image,(470,255),9,red,-1,cv2.LINE_AA)      # overlapping pair
    cv2.circle(image,(630,205),9,red,2,cv2.LINE_AA)       # overlapping hollow pair
    cv2.rectangle(image,(560,28),(720,58),(245,246,249,255),-1)
    cv2.rectangle(image,(560,28),(720,58),(205,212,222,255),1)
    cv2.circle(image,(580,43),8,red,-1,cv2.LINE_AA)       # legend marker, same color
    cv2.putText(image,'LEGEND MARKER',(596,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(80,90,105,255),1)
    cv2.putText(image,'SYNTHETIC CHART / ChartTrace',(105,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,60,75,255),1)
    return image
