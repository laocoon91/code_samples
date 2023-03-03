import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import math as m
import pandas as pd

def buildPolygon(buff_dist,z,subsource_z,fault_trace,strike,dip):
    """
    Build a polygon around the fault surface at a given depth    
    
    INPUT:
    buff_dist: size of buffer on either side of the fault trace (m)
    z: depth of interest (m)
    subsource_z: depth where fault trace values were calculated (m)
    fault_trace: x and y coordinates of fault trace (utm)
    strike: strike of fault (deg)
    dip: dip of fault (deg)
    
    OUTPUT:
    poly: set of points describing a polygon around the fault surface
    at the given depth z
    
    """
    bfd = buff_dist

    # Get direction of fault plane at z relative to fault plane at subsource_z
    dir = strike + 90.0 # you need down-dip direction to find the trace at each depth (perpindicular to strike)
    if(dir>=360):
        dir-=360
    if(dir <= 90.0): # convert from compass heading to cartesian degrees
        thet = 90.0 - dir
    else:
        thet = 360.0 - dir + 90.0

    # Convert angle measures to radians
    dip = dip * m.pi / 180.0
    thet = thet * m.pi / 180.0

    # Find xy location of fault trace at depth z
    for i in range(len(fault_trace)):
        x0,y0 = fault_trace[i,0],fault_trace[i,1]
        dz = z - subsource_z 
        dist = dz / m.tan(dip)
        dx,dy = dist * m.cos(thet), dist * m.sin(thet)
        x1,y1 = x0 + dx, y0 + dy
        if(i==0):
            pts = [x1,y1]
        else:
            pts = np.vstack((pts,[x1,y1]))

    # Calculate a set of points surrounding the fault trace at depth z
    # e.g., equidistant polygon surrounding line
    for i in range(len(fault_trace)):
        if(i==0): # Calculate polygon vertices at one end of the fault trace
            ax,ay = pts[i,0],pts[i,1]
            bx,by = pts[i+1,0],pts[i+1,1]
            ab = [bx-ax,by-ay]

            # Get angle from point of interest to adjacent point, relative to the x-axis
            hb = m.atan2(ab[1],ab[0])

            # Get heading to points on edge of polygon
            phi = hb + m.pi/2.0

            # Get coordinates of new point
            xx,yy = bfd * m.cos(phi) + ax, bfd * m.sin(phi) + ay
            xx2,yy2 = bfd * m.cos(phi+m.pi) + ax, bfd * m.sin(phi+m.pi) + ay

            # Get endcap of the buffer
            x_shift = bfd * m.cos(-1 * hb)
            y_shift = bfd * m.sin(-1 * hb)

            ndx,ndy = ax - x_shift, ay + y_shift
            nd_topx,nd_topy = xx - x_shift, yy + y_shift
            nd_botx,nd_boty = xx2 - x_shift, yy2 + y_shift

            # add points to polygon arrays
            poly1 = np.vstack(([ndx,ndy],[nd_topx,nd_topy],[xx,yy]))
            poly2 = np.vstack(([nd_botx,nd_boty],[xx2,yy2]))

        elif(i>0 and i<len(fault_trace)-1):
            ax,ay = pts[i-1,0],pts[i-1,1]
            bx,by = pts[i,0],pts[i,1]
            cx,cy = pts[i+1,0],pts[i+1,1]
            ba = [ax-bx,ay-by]
            bc = [cx-bx,cy-by]
            
            # Get angle from point of interest to adjacent points
            # relative to the x-axis
            ha = m.atan2(ba[1],ba[0])
            hc = m.atan2(bc[1],bc[0])

            # Find heading halfway between the two adjacent points
            thet = (ha + hc)/2.0-hc

            # Find the supplement to theta
            phi = m.pi - thet

            # Get length of line between the point of interest
            # and the polygon corner
            y = bfd / m.sin(phi)

            # Use heading and line length to find coordinates
            # of the polygon corner
            head = (ha + hc) / 2.0
            xx,yy = y * m.cos(head) + bx, y * m.sin(head) + by
            xx2,yy2 = y * m.cos(head+m.pi) + bx, y * m.sin(head+m.pi) + by

            # add points to polygon arrays
            poly1 = np.vstack((poly1,[xx,yy]))
            poly2 = np.vstack((poly2,[xx2,yy2]))

        else:
            bx,by = pts[i,0],pts[i,1]
            cx,cy = pts[i-1,0],pts[i-1,1]
            cb = [cx-bx,cy-by]

            # Get angle from point of interest to adjacent point
            # relative to the x-axis
            hc = m.atan2(cb[1],cb[0])
            
            # Get heading to new point
            phi = hc + 3*m.pi/2 

            # Get location of new points
            xx,yy = bfd * m.cos(phi) + bx, bfd * m.sin(phi) + by
            xx2,yy2 = bfd * m.cos(phi+m.pi) + bx, bfd * m.sin(phi+m.pi) + by

            # Get endcap of buffer
            x_shift,y_shift = bfd * m.cos(-1*hc),bfd * m.sin(-1*hc)
            ndx,ndy = bx - x_shift, by + y_shift
            nd_topx,nd_topy = xx - x_shift, yy + y_shift
            nd_botx,nd_boty = xx2 - x_shift, yy2 + y_shift

            poly1 = np.vstack((poly1,[xx,yy],[nd_topx,nd_topy]))
            poly2 = np.vstack((poly2,[xx2,yy2],[nd_botx,nd_boty],[ndx,ndy]))

    # Combine sides of polynomial
    poly = np.vstack((poly1,np.flipud(poly2)))
 
    return poly
