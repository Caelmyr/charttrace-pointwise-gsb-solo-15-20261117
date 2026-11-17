import csv
import io
import json
import zipfile
import numpy as np
import cv2
from .tracing import trace,sample_paths,overlays


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
    return series,preview


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
