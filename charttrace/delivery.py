import csv
import io
import json
import zipfile
import numpy as np
import cv2
from .tracing import trace, sample_paths, overlays

BAND_COLUMNS = ['band_segment', 'y_lower', 'y_center', 'y_upper',
                'pixel_y_lower', 'pixel_y_upper', 'band_confidence', 'band_observed']


def extract(document, image):
    series=[]
    preview=image[:,:,:3].copy()
    for s in document['series']:
        traced=trace(image,document['calibration'],s['settings'],s['anchors'],
                     s.get('band_seeds',[]))
        paths=traced['center']
        band=traced['band'] if s['settings']['band']['enabled'] else None
        rows=sample_paths(paths,document['calibration'],s['settings']['samples'],band)
        entry={'id':s['id'],'name':s['name'],'paths':paths,'rows':rows,
               'band':band or [],
               'band_enabled':bool(s['settings']['band']['enabled']),
               'observed_columns':int(sum(sum(p['observed']) for p in paths)),
               'mean_confidence':float(np.mean([r['confidence'] for r in rows])) if rows else 0}
        for row in rows:
            row.update({key:row.get(key) for key in BAND_COLUMNS})
        series.append(entry)
        preview=overlays(np.dstack((preview,image[:,:,3])),document['calibration'],paths,
                         s['anchors'],band,s.get('band_seeds',[]))
    return series,preview


def csv_bytes(series):
    output=io.StringIO(newline='')
    writer=csv.writer(output)
    # Centre-only series keep the original column layout, so old extraction
    # packages and existing consumers stay valid. Band columns appear only
    # when a series actually carries a shaded band.
    with_band=any(s.get('band_enabled') for s in series)
    header=['series','segment','x','y','pixel_x','pixel_y','confidence']
    if with_band:
        header+=BAND_COLUMNS
    writer.writerow(header)
    for s in series:
        for row in s['rows']:
            mapped=dict(row,y_center=row['y'])
            values=[s['name']]+[mapped.get(k,'') for k in
                ['segment','x','y','pixel_x','pixel_y','confidence']]
            if with_band:
                values+=['' if mapped.get(k) is None else mapped.get(k) for k in BAND_COLUMNS]
            writer.writerow(values)
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
    spread=17+7*np.sin((x-100)/95)
    upper=np.rint(y-spread).astype(np.int32)
    lower=np.rint(y+spread).astype(np.int32)
    # Shaded band with a fine one-column break at x=300 and a genuinely
    # unshaded interval 540–598; the centre line is drawn on top of the band.
    hidden=np.zeros(len(x),bool)
    hidden[(x>=540)&(x<598)]=True
    for index,column in enumerate(x):
        if hidden[index] or column==300:
            continue
        cv2.line(image,(column,upper[index]),(column,lower[index]),(245,190,130,255),1)
    cv2.polylines(image,[np.column_stack((x,y)).astype(np.int32)],False,(35,100,210,255),3,cv2.LINE_AA)
    cv2.putText(image,'SYNTHETIC CHART / ChartTrace',(105,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(50,60,75,255),1)
    return image
