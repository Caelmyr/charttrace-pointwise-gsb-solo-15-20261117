import csv
import io
import json
import re
import zipfile
import numpy as np
import cv2
from .tracing import trace,sample_paths,overlays
from . import panels as panel_model


def extract(document,image):
    """Trace every panel inside its own sampling mask."""
    masks=panel_model.panel_masks(document['panels'],image.shape[1],image.shape[0])
    panels=[]
    preview=image[:,:,:3].copy()
    for index,(panel,mask) in enumerate(zip(document['panels'],masks)):
        series=[]
        for s in panel['series']:
            anchors=panel_model.applicable_anchors(document['panels'],index,s['anchors'])
            paths,rgb,evidence=trace(image,panel['calibration'],s['settings'],anchors,mask)
            rows=sample_paths(paths,panel['calibration'],s['settings']['samples'])
            series.append({'id':s['id'],'name':s['name'],'paths':paths,'rows':rows,
                           'observed_columns':int(sum(sum(p['observed']) for p in paths)),
                           'mean_confidence':float(np.mean([r['confidence'] for r in rows])) if rows else 0})
            preview=overlays(np.dstack((preview,image[:,:,3])),panel['calibration'],
                             paths,anchors)
        preview=draw_panel_border(preview,panel)
        panels.append({'id':panel['id'],'name':panel['name'],'series':series,
                       'region':panel['region']})
    return panels,preview


def draw_panel_border(image,panel):
    x,y,w,h=panel['region']['x'],panel['region']['y'],panel['region']['w'],panel['region']['h']
    cv2.rectangle(image,(x,y),(x+w,y+h),(90,120,220),2)
    cv2.putText(image,panel['name'],(x+6,max(18,y+18)),cv2.FONT_HERSHEY_SIMPLEX,.55,
                (90,120,220),1,cv2.LINE_AA)
    return image


def flatten(panels):
    rows=[]
    for panel in panels:
        for series in panel['series']:
            rows.append({**series,'panel':panel['id'],'panel_name':panel['name']})
    return rows


def csv_bytes(series):
    output=io.StringIO(newline='')
    writer=csv.writer(output)
    writer.writerow(['panel','panel_name','series','segment','x','y','pixel_x','pixel_y','confidence'])
    for s in series:
        for row in s['rows']:
            writer.writerow([s.get('panel',''),s.get('panel_name',''),s['name']]+
                            [row[k] for k in ['segment','x','y','pixel_x','pixel_y','confidence']])
    return output.getvalue().encode('utf-8-sig')


def safe_name(text,fallback):
    cleaned=re.sub(r'[^\w.-]+','-',str(text),flags=re.UNICODE).strip('-')
    return cleaned[:48] or fallback


def bundle(record,folder):
    output=io.BytesIO()
    panels=record.get('panels') or []
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data.csv',csv_bytes(flatten(panels)))
        archive.writestr('extraction.json',json.dumps(record,ensure_ascii=False,indent=2))
        manifest={'source':record.get('source'),'width':record.get('width'),
                  'height':record.get('height'),'revision':record.get('revision'),
                  'panels':[{'id':panel['id'],'name':panel['name'],
                             'region':panel['region'],'calibration':panel['calibration'],
                             'series':[s['name'] for s in panel['series']]}
                            for panel in panels]}
        archive.writestr('panels.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for index,panel in enumerate(panels):
            body=csv_bytes([{**s,'panel':panel['id'],'panel_name':panel['name']}
                            for s in panel['series']])
            archive.writestr('panels/%02d-%s.csv'%(index+1,safe_name(panel['name'],'panel')),
                             body)
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


def demo_panels():
    """Two differently scaled plots whose regions meet over a shared legend band."""
    image=np.full((560,1120,4),255,np.uint8)
    left=(60,80,440,430)
    right=(620,80,440,430)
    for x0,y0,w,h in (left,right):
        for step in range(6):
            x=x0+int(w*step/5)
            cv2.line(image,(x,y0),(x,y0+h),(226,230,237,255),1)
        for step in range(5):
            y=y0+int(h*step/4)
            cv2.line(image,(x0,y),(x0+w,y),(226,230,237,255),1)
        cv2.line(image,(x0,y0+h),(x0+w,y0+h),(45,55,72,255),2)
        cv2.line(image,(x0,y0+h),(x0,y0),(45,55,72,255),2)
    x=np.arange(left[0],left[0]+left[2]+1)
    phase=(x-left[0])/left[2]
    cv2.polylines(image,[np.column_stack((x,left[1]+left[3]*.75-left[3]*.5*phase**1.6)).astype(np.int32)],
                  False,(35,100,210,255),3,cv2.LINE_AA)
    x=np.arange(right[0],right[0]+right[2]+1)
    phase=(x-right[0])/right[2]
    cv2.polylines(image,[np.column_stack((x,right[1]+right[3]*.82-right[3]*.62*np.sqrt(phase))).astype(np.int32)],
                  False,(200,55,65,255),3,cv2.LINE_AA)
    # A legend band shared by both demo panels.  Region rectangles overlap here;
    # the exclude policy keeps these swatches out of every panel's evidence.
    cv2.rectangle(image,(512,120),(558,250),(255,255,255,255),-1)
    cv2.rectangle(image,(512,120),(558,250),(120,130,145,255),1)
    cv2.line(image,(518,160),(552,160),(35,100,210,255),4)
    cv2.line(image,(518,210),(552,210),(200,55,65,255),4)
    cv2.putText(image,'SYNTHETIC TWO-PANEL SHEET / ChartTrace',(40,40),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(50,60,75,255),1)
    return image
