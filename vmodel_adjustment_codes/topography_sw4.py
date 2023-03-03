#!/usr/bin/env python3

import pdb
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import math as m
import pandas as pd

# modules for interpolation
from scipy.interpolate import griddata

"""

Set of subroutines used to adjust velocity model to consider topography.
The "main()" subroutine includes all the parameters you will need to
run the other subroutines, as well as an example implementation.

The topography surface used in this implementation references a 
PNW-wide topography model subset from NASA SRTM and NOAA Bathymetry
data. Local smoothing and adjustments have been applied in areas
where datasets did not match. 

The base velocity model is subset from the Stephenson et al.(2017)
CVM, which is saved in binary format. An adjustment
layer is determined from the topography dataset, and the CVM is
altered based on that layer. 

"""

__author__      = "Ian Stone"


def main():

    ####### CHANGE THESE!!! #######

    # Name of top layer velocity model file
    fnmp = "vp_16l1.bin"
    fnms = "vs_16l1.bin"

    # Path to topography ASCII files and file prefix
    topodir = "./new_toposurf/merged_topo_ascii/"
    topoprf = "pnw_topo_" # prefix to identify specific subset of ascii files

    # Name of output netcdf file
    outfn = "16l1_SWIF_1"

    # Define UTM bounds for model area
    xmin = 502800.00 # SWIF 65X60 model ('1' Signifier)
    xmax = 590700.00
    ymin = 5256670.00
    ymax = 5343460.00

    zmin = -270 
    zmax = 300  # zmin and zmax correspond to the lowest and highest elevation in your model area,
                  # rounded UP to the nearest factor of your chosen dz.
                  # (e.g., if your max and min elevations are 225 and -170, 
                  # and dz=30, zmax and zmin should be 240 and -150).
                  # If zmax is lower than the maximum elevation in your
                  # model region, all areas with topography higher than
                  # zmax will be shaved down to zmax.
    
    # Depth of tranistion to non-topographically adjusted vmodel
    # Must be equal to or (preferably) deeper than zmin
    ztrans = -300

    # Define the sampling of the final v-model, in meters
    dx = 30
    dy = 30
    dz = 30

    #########################

    ### Example Implementation : ###

#    # Option 1: Read in input velocity models
#    # You can comment this command out if you have already subset and saved the vmodels before
#    print("reading in velocity model from binary")
#    model_p = readBinary(binModelName=fnmp, zTop=0, zBot=1200,
#                 min_utme=xmin, max_utme=xmax,
#                 min_utmn=ymin, max_utmn=ymax,
#                 newModelName=outfn+'_Vp.nc')
#    model_s = readBinary(binModelName=fnms, zTop=0, zBot=1200,
#                 min_utme=xmin, max_utme=xmax,
#                 min_utmn=ymin, max_utmn=ymax,
#                 newModelName=outfn+'_Vs.nc')

    # Option 2: Load a pre-saved subset of the model
    model_p = xr.open_dataset('../output/'+outfn+'_unrotated_Vp.nc')
    model_s = xr.open_dataset('../output/'+outfn+'_unrotated_Vs.nc')

    dz0 = model_s.z[-1] - model_s.z[-2]     # Original model dz

    # Create arrays for X and Y coordinates of updated vmodel
    X = np.arange(xmin,xmax+dx,dx,dtype='int')
    Y = np.arange(ymin,ymax+dy,dy,dtype='int')

    # Read in topography and create adjustment mask
    print('Reading in topography')
#    # Option 1: Load a topo file for the first time; interpolate; save for later use
#    lyr = getTopoAdjustmentLayer(inpDir=topodir,fpref=topoprf,dz=dz,xvals=X,yvals=Y,zmax=zmax)
#    np.save('./datafiles/lyr_'+str(dx)+'m_'+outfn+'.npy', lyr)

    # Option 2: Load a pre-saved numpy array containing the interpolated topography
    lyr = np.load('./datafiles/lyr_'+str(dx)+'m_'+outfn+'.npy')

    print('adjusting velocity model for topography')
    ### Adjust velocity models according to topography
    model_topo_p = addTopo2Vmodel(model=model_p,lyr=lyr,zmax=zmax,zmin=zmin,ztrans=ztrans,dz=dz,dz0=dz0,xvals=X,yvals=Y,opt='p',outfn=outfn)
    model_topo_s = addTopo2Vmodel(model=model_s,lyr=lyr,zmax=zmax,zmin=zmin,ztrans=ztrans,dz=dz,dz0=dz0,xvals=X,yvals=Y,opt='s',outfn=outfn)

    # Saving .nc file is internal to addTopo2Vmodel
    print(model_topo_p)
    print(model_topo_s)

def getTopoAdjustmentLayer(inpDir="./new_toposurf/merged_topo_ascii/",fpref="pnw_topo_",dz=None,xvals=None,yvals=None,zmax=None):

    """
    Determine which locations in model region will need adjustment based on topographic surface

    Inputs:
    inpFile : ASCII file containing topographic surface. Format described below
    dz  : target spacing of the vmodel in the z direction (meters)
    xvals   : Array of values in the x direction at target spacing and dimension of vmodel (utm, e.g., meters)
    yvals   : Array of values in the y direction at target spacing and dimension of vmodel (utm, e.g., meters)
    zmax   : Set elevations above zmax equal to zmax (for flattening mountains)

    Output:
    lyr : 2D mask describing number of added/subtracted layers at each x-y point in model area

    """

    # Read topography dataset key and compare to bounds of velocity model
    tkey = np.loadtxt(inpDir+"TOPOGRAPHY_KEY.txt",comments='#')
    txmin = tkey[0]
    txmax = tkey[1]
    tymin = tkey[2]
    tymax = tkey[3]
    txno = int(tkey[4])
    tyno = int(tkey[5])
    dx = tkey[6]
    dy = tkey[7]
    tsmpx = int(tkey[8])
    tsmpy = int(tkey[9])

    vxmin = min(xvals)
    vxmax = max(xvals)
    vymin = min(yvals)
    vymax = max(yvals)

    # create grid
    [xv,yv] = np.meshgrid(xvals,yvals)
    topogrid = np.zeros_like(xv)

    # number of tiles in each direction of topo dataset
    noltsx = int(txno/tsmpx+1)
    noltsy = int(tyno/tsmpy+1)

    # Iterate through each ascii file to determine
    # wheter it shares area with vmodel
    for ii in range(1,noltsy+1):
        for jj in range(1,noltsx+1):
                # determine bounds of particular ascii file
                lxmin = txmin + float((jj-1)*tsmpx*dx)
                lxmax = lxmin + float(dx*tsmpx) - dx
                if(lxmax>txmax):
                        lxmax = txmax
                lymax = tymax - float((ii-1)*tsmpy*dy)-dy
                lymin = lymax - float(dy*tsmpy) + dy
                if(lymin<tymin):
                        lymin = tymin

                # determine whether ascii file shares
                # area with vmodel
                # Downloads data as a Pandas DataFrame
                if(lxmin<=vxmin and lxmax>vxmin or lxmin>=vxmin and lxmax<=vxmax or lxmin<=vxmax and lxmax>vxmax or lxmin<=vxmin and lxmax>=vxmax):
                        if(lymin<=vymin and lymax>vymin or lymin>=vymin and lymax<=vymax or lymin<=vymax and lymax>vymax or lymin<=vymin and lymax>=vymax):
                                asciino = (jj+(ii-1)*noltsx)
                                with open(inpDir+fpref+str(asciino)+".asc") as myfile:
                                    head = [next(myfile) for x in range(6)]
                                nx = int(head[0].split()[1])
                                ny = int(head[1].split()[1])
                                lxmin = float(head[2].split()[1])
                                lymin = float(head[3].split()[1])
                                lxmax = lxmin + (nx-1) * dx
                                lymax = lymin + (ny-1) * dy

                                lx = np.arange(lxmin,lxmax+dx,dx)
                                ly = np.arange(lymin,lymax+dy,dy)
                                ly = np.flipud(ly)
                                [lxs,lys] = np.meshgrid(lx,ly)
                                tdat = pd.read_csv(inpDir+fpref+str(asciino)+".asc",sep=' ',header=None,skiprows=6)
                                tdat = tdat.dropna(axis='columns') # <- remove the weird NaN column at the start of the table

                                # Here, I flatten the 2D data, save it to a new DataFrame so
                                # I can set the coordinates as the indices, and then convert
                                # to a XArray, which provides faster intepolation than scipy's griddata.
                                model = pd.DataFrame(np.column_stack((lxs.flatten(),lys.flatten(),tdat.values.flatten())), columns=['utme','utmn','topo'])
                                model = model.set_index(['utme','utmn'])
                                modelxr = model.to_xarray()

                                xsub = xvals[(xvals>=lxmin-dx/2.0) & (xvals<lxmax+dx/2.0)]
                                xbool = np.argwhere((xvals>=lxmin-dx/2.0) & (xvals<lxmax+dx/2.0))
                                ysub = yvals[(yvals>=lymin-dy/2.0) & (yvals<lymax+dy/2.0)]
                                ybool = np.argwhere((yvals>=lymin-dy/2.0) & (yvals<lymax+dy/2.0))

                                [xsubv,ysubv] = np.meshgrid(xsub,ysub)
                                lint = modelxr.interp(utme=xsub,utmn=ysub,method="nearest",kwargs={"fill_value": "extrapolate"})
                                topogrid[ybool[0][0]:ybool[-1][0]+1,xbool[0][0]:xbool[-1][0]+1] = lint.topo.values.T

    # If any topo points exceed zmax, shave them down...
    print('Shaving down topography to ' + str(zmax))
    maxgrid = np.ones_like(topogrid)*zmax
    topogrid = np.where(topogrid > maxgrid, maxgrid, topogrid)

    # At each point on the topographic grid, determine how many 
    # layers to add/subtract based on the chosen dz value
    print('Determining topographic adjustment layer')        
    lyr = np.zeros(topogrid.shape)
    nx = topogrid.shape[1]
    ny = topogrid.shape[0]
    for i in range(nx):
        for j in range(ny):
            if(topogrid[j,i] >= dz):
                nlyr = int(m.floor(topogrid[j,i]/dz)) 
                lyr[j,i] = nlyr
            elif(topogrid[j,i] < 0):
                nlyr = int(m.floor(topogrid[j,i]/dz))
                lyr[j,i] = nlyr

    return lyr

def myround(x, base=5):
    return base * round(x/base)

def addTopo2Vmodel(model=None,lyr=None,zmax=None,zmin=None,ztrans=None,dz=None,dz0=None,xvals=None,yvals=None,opt='s',outfn=None): 

    """
    Adjust the velocity model to account for topography.
    Attempts to maintain uniform thickness of surface layer across model (e.g., values between 0-100m depth).
    For positive topography >100m in height, internal velocity is equal to velocity at 100m depth.
    For negative topography, velocity in the 100m below the topographic surface is replaced by 
        velocity from the original 0-100m depth section. 
    Points in the "air" (i.e., above the topographic surface but within the range spanned by zmin and zmax)
        are set equal to the 0 m velocity value. (For SW4, set to -999.)
    Water velocity is not considered. 

    Inputs:
    model   : velocity data, formatted as an xarray dataset (created using the readBinary() subroutine
    lyr     : 2D mask describing number of added/subtracted layers at each x-y point in model area, 
            (created using the getTopoAdjustmentLayer() subroutine)
    zmax    : the highest elevation in your model area, rounded up to the nearest factor of dz (meters)
    zmin    : greatest depth in your model area that you would like to adjust (meters),
            (should be at least a couple hundred feet below lowest elevation)
    ztrans  : depth below which vmodel will not be affected by topographic adjustment
            (added so that I could keep the original z values the same below zmin when changing dz values)
    dz      : target spacing of the vmodel in the z direction (meters)
    dz0     : original spacing of the vmodel in the z direction (meters)
    xvals   : Array of values in the x direction at target spacing and dimension of vmodel (utm, e.g., meters)
    yvals   : Array of values in the y direction at target spacing and dimension of vmodel (utm, e.g., meters)
    opt     : Specifies P- or S-wave velocity model

    Outputs:
    model_adjusted  : velocity data adjusted to consider topography, formatted as an xarray dataset

    """

    # Determine how many layers will be adjusted above and below sea level
    upmax = max(map(max,lyr)) # max num. layers above sea level
    upmin = min(map(min,lyr)) # max num. layers below sea level
    top = upmax*dz

    # Interpolate dataset to new spacing
    print('Interpolate to new spacing')
    znew = np.concatenate((np.arange(0,-1*zmin+dz,dz),np.arange(-1*ztrans,1300,dz0)))
    #model = model.interp(utme=xvals,utmn=yvals,z=znew,method="nearest",kwargs={"fill_value": None}) # applies nearest neighbors interp.
    model = model.interp(utme=xvals,utmn=yvals,z=znew,method="linear",kwargs={"fill_value": "extrapolate"}) # applies linear interp.

    # Subset 0 m and ~100 m layer
    print('Extract layers')
    lay0 = model.sel(z=0)
    dfac = m.floor(100/dz)
    ddep = dfac*dz
    lay1 = model.sel(z=ddep)

    lay0 = lay0.copy(deep=True)
    lay1 = lay1.copy(deep=True)

    # Set velocity values in layers between dz and dz*dfac equal to lay1
    for i in range(1,dfac):
        if opt == 'p':
            model.vp.values[:,:,i] = lay1.vp.values[:,:]
        else:
            model.vs.values[:,:,i] = lay1.vs.values[:,:]

    # Build data array for topography above sea level
    print('Build array for topo above sea level')
    ztop = np.arange(dz,zmax+dz,dz,dtype='int')
    ztop = -1*ztop  # Positive topography is "negative" in this coordinate system

    # SW4 requires any grid cell that is 'air' be set to -999.
    asl = np.ones((len(xvals),len(yvals),len(ztop)))*-999.

    # Define velocities above sea level 
    lvl_cnt = int(upmax)
    print("lvl_cnt: ",lvl_cnt)
    for i in range(lvl_cnt)[::-1]:
        if opt == 'p':
            asl[:,:,i] = np.where((lyr[:,:].T==i) | (lyr[:,:].T==i+1), lay0.vp[:,:].values, asl[:,:,i])
            asl[:,:,i] = np.where(lyr[:,:].T>i+1, lay1.vp[:,:].values, asl[:,:,i])
        else:
            asl[:,:,i] = np.where((lyr[:,:].T==i) | (lyr[:,:].T==i+1), lay0.vs[:,:].values, asl[:,:,i])
            asl[:,:,i] = np.where(lyr[:,:].T>i+1, lay1.vs[:,:].values, asl[:,:,i])

    # set state of layer between elevations of 0 and dz (layer 0 of asl)
    if opt == 'p':
        asl[:,:,0] = np.where(lyr[:,:].T>1, lay1.vp[:,:].values, asl[:,:,0])
        asl[:,:,0] = np.where((lyr[:,:].T==0)|(lyr[:,:].T==1), lay0.vp[:,:].values, asl[:,:,0])
    else:
        asl[:,:,0] = np.where(lyr[:,:].T>1, lay1.vs[:,:].values, asl[:,:,0])
        asl[:,:,0] = np.where((lyr[:,:].T==0)|(lyr[:,:].T==1), lay0.vs[:,:].values, asl[:,:,0])

    # Turn above sea level variable into xarray dataset
    if opt == 'p':
        asl_xar = xr.Dataset(
            {
                'vp':(['utme','utmn','z'],asl)
            }, 
            coords={
                'utme':(['utme'],xvals),
                'utmn':(['utmn'],yvals),
                'z':(['z'],ztop),
            },
        )
    else:
        asl_xar = xr.Dataset(
            {
                'vs':(['utme','utmn','z'],asl)
            }, 
            coords={
                'utme':(['utme'],xvals),
                'utmn':(['utmn'],yvals),
                'z':(['z'],ztop),
            },
        )

    # Adjust velocity below sea level
    print('Adjusting velocity model below sea level')
    if(zmin<0.):
        ndwn = int(-1*zmin/dz)+1
        for i in range(0,ndwn):
            if opt == 'p':
                if(i==0):
                    model.vp.values[:,:,0] = xr.where(lyr[:,:].T>0, lay1.vp.values[:,:], model.vp.values[:,:,0])
                    model.vp.values[:,:,1] = xr.where(lyr[:,:].T>0, lay1.vp.values[:,:], model.vp.values[:,:,1])
                model.vp.values[:,:,i] = xr.where(lyr[:,:].T<-1*i-1, -999., model.vp.values[:,:,i])
                model.vp.values[:,:,i] = xr.where(lyr[:,:].T==-1*i-1, lay0.vp.values[:,:], model.vp.values[:,:,i])
                model.vp.values[:,:,i] = xr.where(lyr[:,:].T==-1*i, lay0.vp.values[:,:], model.vp.values[:,:,i])
                model.vp.values[:,:,i] = xr.where(lyr[:,:].T==-1*i+1, lay1.vp.values[:,:], model.vp.values[:,:,i])
            else:
                if(i==0):
                    model.vs.values[:,:,0] = xr.where(lyr[:,:].T>0, lay1.vs.values[:,:], model.vs.values[:,:,0])
                    model.vs.values[:,:,1] = xr.where(lyr[:,:].T>0, lay1.vs.values[:,:], model.vs.values[:,:,1])
                model.vs.values[:,:,i] = xr.where(lyr[:,:].T<-1*i-1, -999., model.vs.values[:,:,i])
                model.vs.values[:,:,i] = xr.where(lyr[:,:].T==-1*i-1, lay0.vs.values[:,:], model.vs.values[:,:,i])
                model.vs.values[:,:,i] = xr.where(lyr[:,:].T==-1*i, lay0.vs.values[:,:], model.vs.values[:,:,i])
                model.vs.values[:,:,i] = xr.where(lyr[:,:].T==-1*i+1, lay1.vs.values[:,:], model.vs.values[:,:,i])
        
    else:
        if opt =='p':
            model.vp.values[:,:,1] = lay1.vp.values[:,:]
            model.vp.values[:,:,0] = np.where(lyr[:,:].T>0, lay1.vp.values[:,:], model.vp.values[:,:,0])
        else:
            model.vs.values[:,:,1] = lay1.vs.values[:,:]
            model.vs.values[:,:,0] = np.where(lyr[:,:].T>0, lay1.vs.values[:,:], model.vs.values[:,:,0])
    
    # Combine above- and below-sea-level portions 
    print('Combining top and bottom components of adjusted velocity model')
    asl_xar = asl_xar.sortby('z',ascending=True)
    model_adjusted = xr.concat([asl_xar,model],dim='z')

    # Save adjusted model
    if opt == 'p':
        model_adjusted.to_netcdf('../output/'+outfn+'_vp.nc')
    else:
        model_adjusted.to_netcdf('../output/'+outfn+'_vs.nc')

    return model_adjusted

def readBinary(zTop=0, zBot=1200,
               min_utme=None, max_utme=None,
               min_utmn=None, max_utmn=None,
               newModelName='subModel.nc',
               binModelName='',
               saveFull='N'):
    """
    Read in 3-D CVM from binary files.

    This function only works with one binary file at a time.
    Input parameters are used to specify bounds of a sub-model,
    if only a small chunk of the full CVM is needed.

    Offshore water is set to V=1e20 in binary files.

    All UTM coordinates are given for Zone 10T.

    For Puget Lowland: zTop=0, zBot=200,
               min_utme=446700, max_utme=567100,
               min_utmn=5194300, max_utmn=5316700,
               modelName='subModel_0_200m_PL.nc'

    Parameters:
    zTop: Top of sub-model [m]
    zBot: Bottom of sub-model [m]
    min_utme: Minimum easting [m]
    max_utme: Maximum easting [m]
    min_utmn: Minimum northing [m]
    max_utmn: Maximum northing [m]

    Returns:
    subModelxr: Sub-model in xarray format

    """

    # The following lines describe the size of the binary file values.
    #
    # Layer 1 0-1200m depth
    # 3271 in EW direction
    # 5367 in NS direction
    # 13 in z
    # dx=dy= 200m,  dz= 100m
    #
    if 'l1' in binModelName:
        nx = 3271; ny = 5367; nz = 13
        dx = 200;  dy = 200;  dz = 100
        zmin = 0; zmax = 1200
    #
    # Layer 2 1500-9900m depth
    # 2181 in EW
    # 3578 in NS
    # 29 in z
    # dx=dy=dz=300m
    #
    if 'l2' in binModelName:
        nx = 2181; ny = 3578; nz = 29
        dx = 300;  dy = 300;  dz = 300
        zmin = 1500; zmax = 9900
    #
    # Layer 3 10800-59400m depth
    # 727 in EW
    # 1193 NS
    # 55 in z
    # dx=dy=dz=900m
    #
    if 'l3' in binModelName:
        nx = 727; ny = 1193; nz = 55
        dx = 900; dy = 900;  dz = 900
        zmin = 10800; zmax = 59400
    #
    # The SW corner of the velocity model is -10800m East, 4467300m N Zone 10.
    SWcornerFull = [-10800, 4467300]
    #

    # Read in binary file
    v = np.fromfile(binModelName, dtype='<f4')

    # Generate arrays for x, y, & z locations
    z = np.linspace(zmin, zmax, nz)
    z = np.repeat(z, (np.ones(len(z))*nx*ny).astype(int))
    z = z[::-1]             # Reverse array

    y = np.linspace(SWcornerFull[1], SWcornerFull[1]+ny*dy, ny, endpoint=False)
    y = np.repeat(y, (np.ones(len(y))*nx).astype(int))
    y = np.tile(y, nz)     # Repeat array for each depth

    x = np.linspace(SWcornerFull[0], SWcornerFull[0]+nx*dx, nx, endpoint=False)
    x = np.tile(x, ny*nz)

    # Convert CVM to dataframe
    model = pd.DataFrame(np.column_stack((x,y,z,v)), columns=['utme','utmn','z',(binModelName.split('/'))[1].split('_16')[0]])

    if saveFull == 'N' or saveFull == 'n':
        # Subset model (speed things up)
        if min_utme==None:
            # Use full horizontal model extent
            subModel = model[(model["z"] >= zTop) & (model["z"] <= zBot)]
        else:
            subModel = model[(model["z"] >= zTop) & (model["z"] <= zBot)
                             & (model["utme"] >= min_utme) & (model["utme"] <= max_utme)
                             & (model["utmn"] >= min_utmn) & (model["utmn"] <= max_utmn)]
        subModel = subModel.set_index(['utme','utmn','z'])    # Set these parameters as coordinates

        # Convert to xarray (slow)
        subModelxr = subModel.to_xarray()

        # Save xarray for faster reload later
        subModelxr.to_netcdf('../output/' + newModelName)

        return subModelxr

        # This can be VERY slow (~ 1 hour)
    if saveFull == 'Y' or saveFull == 'y':
        model = model.set_index(['utme','utmn','z'])
        modelxr = model.to_xarray()
        modelxr.to_netcdf('../output/' + (binModelName.split('/'))[1].split('.bin')[0]+'.nc')

        return modelxr


if __name__ == '__main__':
        main()
