"""Rectified color evidence and global path inference with explicit missing columns."""
import cv2
import numpy as np
from image_lab.validation import Invalid
from .coordinates import Coordinates


def rectify(image, calibration, size):
    width,height=size
    grid=Coordinates(calibration).grid(width,height).astype(np.float32)
    rgb=image[:,:,:3].astype(np.float32)/255
    alpha=image[:,:,3].astype(np.float32)/255
    premult=rgb*alpha[:,:,None]
    sampled=cv2.remap(premult,grid[:,:,0],grid[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    coverage=cv2.remap(alpha,grid[:,:,0],grid[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    visible=sampled+(1-coverage[:,:,None])
    return visible,coverage


def color_evidence(rgb, coverage, color, tolerance):
    target=np.asarray(color,dtype=np.float32).reshape(1,1,3)/255
    lab=cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2LAB)
    selected=cv2.cvtColor(target,cv2.COLOR_RGB2LAB)[0,0]
    distance=np.sqrt(np.sum((lab-selected)**2,axis=2))
    probability=np.exp(-.5*(distance/tolerance)**2)*coverage
    return probability.astype(float)


def segments(valid):
    positions=np.flatnonzero(valid)
    if not len(positions):
        return []
    return np.split(positions,np.flatnonzero(np.diff(positions)>1)+1)


def infer_path(evidence, settings, anchors):
    height,width=evidence.shape
    max_step=settings['max_step']
    smooth=settings['smoothness']
    threshold=settings['min_confidence']
    gap_limit=settings['gap_limit']
    # Every column has a best observation, but only supported columns enter the
    # path. Short unsupported bridges preserve continuity without becoming data.
    supported=np.max(evidence,axis=0)>=threshold
    allowed=supported.copy()
    for run in segments(~supported):
        if run[0]>0 and run[-1]<width-1 and len(run)<=gap_limit:
            allowed[run]=True
    forced={}
    for anchor in anchors:
        column=int(round(anchor[0]*(width-1)))
        row=(1-anchor[1])*(height-1)
        if 0<=column<width and 0<=row<height:
            forced[column]=row
            allowed[column]=True
            supported[column]=True
    paths=[]
    for run in segments(allowed):
        costs=np.full((len(run),height),np.inf)
        parent=np.full((len(run),height),-1,dtype=np.int32)
        for offset,column in enumerate(run):
            local=-np.log(np.clip(evidence[:,column],1e-8,1.))
            if not supported[column]:
                local[:]=0
            if column in forced:
                local+=1000*np.abs(np.arange(height)-forced[column])
            if offset==0:
                costs[offset]=local
                continue
            previous=costs[offset-1]
            for row in range(height):
                lower=max(0,row-max_step)
                upper=min(height,row+max_step+1)
                candidates=previous[lower:upper]+smooth*((np.arange(lower,upper)-row)/max_step)**2
                source=int(np.argmin(candidates))+lower
                parent[offset,row]=source
                costs[offset,row]=local[row]+candidates[source-lower]
        row=int(np.argmin(costs[-1]))
        rows=[]
        for offset in range(len(run)-1,-1,-1):
            rows.append(row)
            row=parent[offset,row]
        rows=np.array(rows[::-1],dtype=float)
        # Subpixel centroid uses only nearby evidence. A blank bridge remains a
        # path guide; supported flags prevent interpolating it into observed CSV.
        confidence=[]
        for index,column in enumerate(run):
            row=rows[index]
            if column not in forced and supported[column]:
                lo=max(0,int(row)-2)
                hi=min(height,int(row)+3)
                weights=evidence[lo:hi,column]
                if weights.sum()>0:
                    rows[index]=np.dot(np.arange(lo,hi),weights)/weights.sum()
            confidence.append(float(evidence[int(np.clip(round(rows[index]),0,height-1)),column]))
        chart=np.column_stack((run/(width-1),1-rows/(height-1)))
        paths.append({'chart':chart.tolist(),'observed':supported[run].tolist(),
                      'confidence':confidence})
    return paths


def trace(image, calibration, settings, anchors):
    size=(settings['width'],settings['height'])
    rgb,coverage=rectify(image,calibration,size)
    evidence=color_evidence(rgb,coverage,settings['color'],settings['tolerance'])
    coords=Coordinates(calibration)
    chart_anchors=coords.pixels_to_chart([a['pixel'] for a in anchors]) if anchors else []
    paths=infer_path(evidence,settings,chart_anchors)
    return paths,rgb,evidence


def sample_paths(paths, calibration, count):
    coords=Coordinates(calibration)
    rows=[]
    grid=np.linspace(0,1,count)
    # Missing evidence splits a run before interpolation. No extrapolation and
    # no interpolation across disconnected runs, even if their x ranges meet.
    for path_index,path in enumerate(paths):
        chart=np.asarray(path['chart'])
        confidence=np.asarray(path['confidence'])
        for run in segments(np.asarray(path['observed'],dtype=bool)):
            if len(run)<2:
                continue
            points=chart[run]
            selected=grid[(grid>=points[0,0]-1e-10)&(grid<=points[-1,0]+1e-10)]
            if not len(selected):
                continue
            y=np.interp(selected,points[:,0],points[:,1])
            q=np.interp(selected,points[:,0],confidence[run])
            chart_points=np.column_stack((selected,y))
            pixels=coords.chart_to_pixels(chart_points)
            data=coords.chart_to_data(chart_points)
            for i in range(len(selected)):
                rows.append({'segment':path_index,'x':float(data[i,0]),'y':float(data[i,1]),
                             'pixel_x':float(pixels[i,0]),'pixel_y':float(pixels[i,1]),
                             'confidence':float(q[i])})
    return rows


def overlays(image, calibration, paths, anchors):
    result=image[:,:,:3].copy()
    coords=Coordinates(calibration)
    for path in paths:
        pixels=np.rint(coords.chart_to_pixels(path['chart'])).astype(np.int32)
        for run in segments(np.asarray(path['observed'],bool)):
            cv2.polylines(result,[pixels[run]],False,(0,210,160),2,cv2.LINE_AA)
    basis=np.rint([calibration['x_end'],calibration['origin'],calibration['y_end']]).astype(np.int32)
    cv2.polylines(result,[basis],False,(255,150,25),2,cv2.LINE_AA)
    for anchor in anchors:
        cv2.circle(result,tuple(np.rint(anchor['pixel']).astype(int)),5,(180,60,240),-1)
    return result
