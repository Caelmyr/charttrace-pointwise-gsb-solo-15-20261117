"""Rectified colour evidence, centre paths and optional shaded-band bounds."""
import cv2
import numpy as np
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


def _runs(mask, maximum_gap):
    """Inclusive True runs, bridging holes no wider than maximum_gap rows."""
    runs=[]
    start=last=None
    for index,value in enumerate(mask):
        if value:
            if start is None:
                start=index
            last=index
        elif start is not None and index-last-1>maximum_gap:
            runs.append((start,last))
            start=last=None
    if start is not None:
        runs.append((start,last))
    return runs


def _select_runs(runs, line, occlusion):
    """Keep band runs that belong to the traced curve.

    The band normally sits across the centre line, so both sides are kept.
    When the stroke hides the middle, the two sides are rejoined across a gap
    no wider than ``occlusion``. Runs that are neither bridged sides nor
    adjacent to the line are another same-coloured region (legend or panel).
    A gap wider than the limit is a genuine unshaded interval and splits.
    """
    if not runs:
        return []
    bridged=[]
    for top,bottom in runs:
        if bridged:
            previous_top,previous_bottom=bridged[-1]
            if top-previous_bottom-1<=occlusion and previous_bottom<line<top:
                bridged[-1]=(previous_top,max(previous_bottom,bottom))
                continue
        bridged.append((top,bottom))
    return [run for run in bridged if max(run[0]-line,line-run[1],0)<=occlusion]


def _centre_rows(paths, width, height):
    rows=np.full(width,np.nan)
    for path in paths:
        chart=np.asarray(path['chart'],dtype=float)
        columns=np.rint(chart[:,0]*(width-1)).astype(int)
        np.clip(columns,0,width-1,out=columns)
        rows[columns]=(1-chart[:,1])*(height-1)
    return rows


def _matches(first, second, tolerance):
    return first[0]-tolerance<=second[1] and second[0]-tolerance<=first[1]


def _components(nodes, gap_limit, tolerance):
    """Join per-column vertical runs into 2-D band regions.

    Only columns separated by a small break are joined; a genuine unshaded
    interval stays a separate component, so its bounds are never invented.
    """
    parent={}
    def find(item):
        while parent[item]!=item:
            parent[item]=parent[parent[item]]
            item=parent[item]
        return item
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[b]=a
    for column,items in enumerate(nodes):
        for index in range(len(items)):
            parent[(column,index)]=(column,index)
    last=None
    for column,items in enumerate(nodes):
        if not items:
            continue
        if last is not None and column-last-1<=gap_limit:
            for i,first in enumerate(nodes[last]):
                for j,second in enumerate(items):
                    if _matches(first,second,tolerance):
                        union((last,i),(column,j))
        last=column
    groups={}
    for node in parent:
        groups.setdefault(find(node),[]).append(node)
    return [sorted(members) for members in groups.values()]


def infer_band(evidence, paths, settings, seeds):
    """Return shaded-band regions around the traced centre line.

    Each region contains the interpolated vertical span for every grid column
    in its range plus an ``observed`` flag per column. Small breaks are bridged
    up to the band gap limit; longer genuinely unshaded intervals split regions.
    Seeds from the checking clicks keep only the regions the user picked.
    """
    band=settings['band']
    height,width=evidence.shape
    centre=_centre_rows(paths,width,height)
    mask=evidence>=band['min_confidence']
    # Without checking clicks the band must hug the centre line; with clicks a
    # wider net keeps other same-coloured regions around so the clicks can
    # explicitly pick or reject them.
    reach=max(band['occlusion_gap'],height*.5) if seeds else band['occlusion_gap']
    nodes=[[] for _ in range(width)]
    for column in range(width):
        line=centre[column]
        if not np.isfinite(line):
            continue
        runs=_runs(mask[:,column].tolist(),band['vertical_gap'])
        for top,bottom in _select_runs(runs,line,reach):
            nodes[column].append((top,bottom,float(np.mean(evidence[top:bottom+1,column]))))
    if not any(nodes):
        return []
    regions=_components(nodes,band['gap_limit'],band['occlusion_gap'])
    if seeds:
        # Checking clicks pick the wanted band. A region is kept when a click
        # lands on or just beside its span, so a same-coloured legend or
        # another panel can be rejected explicitly.
        tolerance=max(6.,band['occlusion_gap'])
        kept=[]
        for members in regions:
            best=None
            for column,node_index in members:
                top,bottom,_=nodes[column][node_index]
                for seed_column,seed_row in seeds:
                    vertical=max(top-seed_row,seed_row-bottom,0)
                    distance=float(np.hypot(column-seed_column,vertical))
                    if best is None or distance<best:
                        best=distance
            if best is not None and best<=tolerance:
                kept.append(members)
        if not kept:
            return []
        regions=kept
    output=[]
    for members in regions:
        grouped={}
        for column,index in members:
            grouped.setdefault(column,[]).append(nodes[column][index])
        ordered=sorted(grouped)
        columns=np.array(ordered,dtype=float)
        # Several vertical runs in one column can belong to the same region;
        # the band spans all of them, never more.
        tops=np.array([min(node[0] for node in grouped[c]) for c in ordered],dtype=float)
        bottoms=np.array([max(node[1] for node in grouped[c]) for c in ordered],dtype=float)
        confidence=np.array([max(node[2] for node in grouped[c]) for c in ordered],dtype=float)
        span=np.arange(ordered[0],ordered[-1]+1)
        top=np.interp(span,columns,tops)
        bottom=np.interp(span,columns,bottoms)
        quality=np.interp(span,columns,confidence)
        observed=np.isin(span,ordered)
        chart=np.column_stack((span/(width-1),1-bottom/(height-1),1-top/(height-1)))
        output.append({'chart':chart.tolist(),'observed':observed.tolist(),
                       'confidence':quality.tolist()})
    return output


def trace(image, calibration, settings, anchors, band_seeds=None):
    size=(settings['width'],settings['height'])
    rgb,coverage=rectify(image,calibration,size)
    evidence=color_evidence(rgb,coverage,settings['color'],settings['tolerance'])
    coords=Coordinates(calibration)
    chart_anchors=coords.pixels_to_chart([a['pixel'] for a in anchors]) if anchors else []
    paths=infer_path(evidence,settings,chart_anchors)
    band=[]
    if settings['band']['enabled']:
        band_settings=settings['band']
        band_evidence=color_evidence(rgb,coverage,band_settings['color'],band_settings['tolerance'])
        seeds=[]
        if band_seeds:
            chart_seeds=coords.pixels_to_chart([a['pixel'] for a in band_seeds])
            for u,v in chart_seeds:
                seeds.append((int(round(u*(size[0]-1))),(1-v)*(size[1]-1)))
        band=infer_band(band_evidence,paths,settings,seeds)
    return {'center':paths,'rgb':rgb,'evidence':evidence,'band':band}


def _band_at(band, x):
    for index,region in enumerate(band):
        chart=np.asarray(region['chart'],dtype=float)
        if chart[0,0]-1e-9<=x<=chart[-1,0]+1e-9:
            observed=bool(np.interp(x,chart[:,0],np.asarray(region['observed'],dtype=float))>=.5)
            confidence=float(np.interp(x,chart[:,0],region['confidence']))
            return (index,float(np.interp(x,chart[:,0],chart[:,1])),
                    float(np.interp(x,chart[:,0],chart[:,2])),observed,confidence)
    return None


def sample_paths(paths, calibration, count, band=None):
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
                row={'segment':path_index,'x':float(data[i,0]),'y':float(data[i,1]),
                     'pixel_x':float(pixels[i,0]),'pixel_y':float(pixels[i,1]),
                     'confidence':float(q[i])}
                if band is not None:
                    row.update(band_fields(coords,band,float(selected[i]),
                                           float(pixels[i,1]),float(data[i,1])))
                rows.append(row)
    return rows


def band_fields(coords, band, x, centre_pixel, centre_data):
    blank={'band_segment':None,'y_center':None,'y_lower':None,'y_upper':None,
           'pixel_y_lower':None,'pixel_y_upper':None,
           'band_confidence':None,'band_observed':False}
    found=_band_at(band,x)
    if found is None:
        return blank
    index,bottom_chart,top_chart,observed,confidence=found
    bounds=[(float(coords.chart_to_data([[x,bottom_chart]])[0,1]),
             float(coords.chart_to_pixels([[x,bottom_chart]])[0,1])),
            (float(coords.chart_to_data([[x,top_chart]])[0,1]),
             float(coords.chart_to_pixels([[x,top_chart]])[0,1]))]
    # The centre stroke hides the band behind it, so a measured edge can sit a
    # fraction inside the real band. Keep the recorded interval around the
    # centre value rather than publishing a bound the centre crosses.
    bounds.append((centre_data,centre_pixel))
    lower=min(bounds,key=lambda item:item[0])
    upper=max(bounds,key=lambda item:item[0])
    return {'band_segment':index,'y_center':centre_data,
            'y_lower':lower[0],'y_upper':upper[0],
            'pixel_y_lower':lower[1],'pixel_y_upper':upper[1],
            'band_confidence':confidence,'band_observed':observed}


def overlays(image, calibration, paths, anchors, band=None, band_seeds=None):
    result=image[:,:,:3].copy()
    coords=Coordinates(calibration)
    if band:
        for region in band:
            chart=np.asarray(region['chart'],dtype=float)
            upper=np.rint(coords.chart_to_pixels(np.column_stack((chart[:,0],chart[:,2])))).astype(np.int32)
            lower=np.rint(coords.chart_to_pixels(np.column_stack((chart[:,0],chart[:,1])))).astype(np.int32)
            polygon=np.vstack((upper,lower[::-1]))
            fill=np.zeros(result.shape[:2],np.uint8)
            cv2.fillPoly(fill,[polygon],255,cv2.LINE_AA)
            result[fill>0]=(result[fill>0]*.84+np.array([255,178,92])*.16).astype(np.uint8)
            cv2.polylines(result,[upper],False,(225,125,30),2,cv2.LINE_AA)
            cv2.polylines(result,[lower],False,(225,125,30),2,cv2.LINE_AA)
    for path in paths:
        pixels=np.rint(coords.chart_to_pixels(path['chart'])).astype(np.int32)
        for run in segments(np.asarray(path['observed'],bool)):
            cv2.polylines(result,[pixels[run]],False,(0,210,160),2,cv2.LINE_AA)
    basis=np.rint([calibration['x_end'],calibration['origin'],calibration['y_end']]).astype(np.int32)
    cv2.polylines(result,[basis],False,(255,150,25),2,cv2.LINE_AA)
    for anchor in anchors:
        cv2.circle(result,tuple(np.rint(anchor['pixel']).astype(int)),5,(180,60,240),-1)
    for seed in band_seeds or []:
        cv2.circle(result,tuple(np.rint(seed['pixel']).astype(int)),6,(235,105,20),-1)
        cv2.circle(result,tuple(np.rint(seed['pixel']).astype(int)),6,(255,255,255),1,cv2.LINE_AA)
    return result
