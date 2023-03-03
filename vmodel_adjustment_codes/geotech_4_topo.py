#!/usr/bin/env python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import math as m
import pandas as pd

# modules for location determination
import cartopy as cart
import cartopy.crs as ccrs
from matplotlib.patches import Path, PathPatch
from pyproj import Proj
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.prepared import prep

# modules for interpolation
from scipy.interpolate import griddata

"""

Given a velocity model amended with surface topography, add a geotechnical
layer to the upper few hundred meters. Adjustments are based on the
functional forms developed by a. grant for the M9 ShakeMap paper. 
Constants are based on PNW regional profiles from the Ahdi et al. (2017) dataset.

Inputs and outputs should be in netcdf format.

NOTE: The order in which you apply the geotech subroutines is important.
General order should be:
1. addPugetLowlandGradient
2. addFillAlluvium
3. addRockGradient
4. addRandomness

"""

__author__      = "Ian Stone"

def main():

    ### CHANGE THESE!!! ###

    # name of netcdf files containing Vp and Vs models
    fvp = 'subModel_Vp_SWIF.nc'
    fvs = 'subModel_Vs_SWIF.nc'

    #######################

    fvpPref = fvp.split('.')
    fvsPref = fvs.split('.')

    modelTrueVp = xr.open_dataset(fvp)
    modelTrue = xr.open_dataset(fvs)

    print("Adding Puget Lowland Gradient")
    addPugetLowlandGradient(modelTrue, modelTrueVp)

    print("Adding Alluvium")
    addFillAlluvium('../output/'+fvpPref+'_gradPL.nc', '../output/'+fvsPref+'_gradPL.nc')	

    print("Adding Rock Gradient")
    addRockGradient('../output/'+fvpPref+'_gradPL_FA.nc','../output/'+fvsPref+'_gradPL_FA.nc')

    print("Adding Randomness")    
    addRandomness('../output/'+fvpPref+'_gradPL_FA_gradRock.nc','../output/'+fvsPref+'_gradPL_FA_gradRock.nc')

def addPugetLowlandGradient(modelTrue, modelTrueVp):

    """
    
    Add gradient to sites within the Puget Lowland

    This uses the functional form developed by a. grant
    for the M9 ShakeMap paper. Constants are based on
    Puget Lowland profiles from the Ahdi et al. (2017) dataset.
    Uses the USGS slope-based Vs30 as an input parameter.

    We use this new function to set Vs until the Vs
    of the Stephenson model is exceeded. (This is generally ~100 m.)
    Then, switch back to Stephenson Vs values. Profiles are valid
    to deeper depths than non-Puget Lowland sites.
    Updated to also modify Vp using Vp/Vs=2.5

    Saves resulting models as netCDF files.

    Parameters:
    modelTrue, modelTrueVp: Vs and Vp Sub-models to modify

    Returns:
    None

    """

    # Deep copy new object (prevents alteration of original)
    model = modelTrue.copy(deep=True)
    modelVp = modelTrueVp.copy(deep=True)

    # Define matrices describing the z index where the topographic surface is encountered
    # and what the VS is at that surface.
    surfVS = np.ones_like(model.vs.values[:,:,0])*-1.0
    surfind = np.ones_like(model.vs.values[:,:,0],dtype='int')*-1
    uno = np.ones_like(model.vs.values[:,:,0])
    for i in range(len(model.z.values)):
        surfind[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfind[:,:]==-1)),uno*i,surfind[:,:])
        surfVS[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfVS[:,:]==-1.0)),model.vs.values[:,:,i],surfVS[:,:])

    # Find sites were 500 m/s< Vs_surface <700
    # All Puget Lowland sites are ~600 m/s before adding randomness
    # Indices to apply new Vs(z)
    ix, iy = np.where(((surfVS > 500) & (surfVS < 700)) |
        ((model.utme > 507540) & (model.utme < 561604) & (model.utmn > 5254066) & (model.utmn < 5275503)))

    # Import new Vs30 (considers slope of bathymetry)
    fvs = '../output/vs30_bathy1.nc'
    q = xr.open_dataset(fvs)
    qinterp = q.interp(utmn=model.utmn,utme=model.utme)
    vs30 = qinterp.vs30.values

    # Define a dz array.
    # This describes the increase in depth between subsequent z indices.
    # Normally, the Vs_puget relation is depth dependent relative to z=0.
    # However with topography, this relation must be considered relative to the ground surface at each XY point.
    # Since spacing in the z-direction is not uniform (e.g., 10m to 100m at depth),
    # the spacing of z-values in the Vs_puget relation may vary between XY points.
    # This array helps to address this disparity when calculating gradient Vs_puget values.
    dzmat = np.zeros_like(model.z.values)
    dzmat = dzmat[1:]
    for i in range(len(model.z.values)-1):
        dzmat[i] = model.z.values[i+1]-model.z.values[i]
    print("dzmat: ",dzmat)

    # # Amend Puget Lowland sites
    # # Use a.g. function from 0-1200 m; array with locations at all depths
    print("model.z: ",model.z.values)
    interpStart = 1200; iizStart = np.where(model.z == interpStart)[0][0]
    ixr = np.tile(ix, iizStart+1) # "tile" repeats sequentially (e.g., 123123123). "repeat" repeats by index value (e.g., 111222333)
    iyr = np.tile(iy, iizStart+1)
    izr = np.repeat(np.arange(iizStart+1), len(ix))

    # Use a.g. function from 0-1200 m
    vs_agrant = np.zeros_like(model.vs.values)
    for i in range(len(ix)):
        # Figure out what the z values are for each xy point
        zloc = np.zeros((len(model.z.values)-surfind[ix[i],iy[i]]))
        zloc[0]=0
        for j in range(2,len(zloc)): 
            zloc[j] = zloc[j-1]+dzmat[surfind[ix[i],iy[i]]+j-1]
        vs_agrant[ix[i],iy[i],surfind[ix[i],iy[i]]+1:] = Vs_puget(vs30.T[ix[i],iy[i]], zloc[1:])
        vs_agrant[ix[i],iy[i],surfind[ix[i],iy[i]]] = vs_agrant[ix[i],iy[i],surfind[ix[i],iy[i]]+1]

    modelTemp = model.copy(deep=True)

    # Compare model.vs.values and vs_agrant; keep lowest Vs values
    model.vs.values[ixr, iyr, izr] = np.minimum(model.vs.values[ixr,iyr,izr], vs_agrant[ixr,iyr,izr])
    modelVp.vp.values[ixr, iyr, izr] =  np.where(((vs_agrant[ixr,iyr,izr] < modelTemp.vs.values[ixr,iyr,izr]) & \
        (model.vs.values[ixr,iyr,izr] <= 900)), model.vs.values[ixr, iyr, izr] * 2.500, modelVp.vp.values[ixr, iyr, izr])
    modelVp.vp.values[ixr, iyr, izr] =  np.where(((vs_agrant[ixr,iyr,izr] < modelTemp.vs.values[ixr,iyr,izr]) & \
        (model.vs.values[ixr,iyr,izr] > 900) & (model.vs.values[ixr,iyr,izr] <= 933)), model.vs.values[ixr, iyr, izr] * 2.325, modelVp.vp.values[ixr, iyr, izr])
    modelVp.vp.values[ixr, iyr, izr] =  np.where(((vs_agrant[ixr,iyr,izr] < modelTemp.vs.values[ixr,iyr,izr]) & \
        (model.vs.values[ixr,iyr,izr] > 933) & (model.vs.values[ixr,iyr,izr] <= 966)), model.vs.values[ixr, iyr, izr] * 2.250, modelVp.vp.values[ixr, iyr, izr])
    modelVp.vp.values[ixr, iyr, izr] =  np.where(((vs_agrant[ixr,iyr,izr] < modelTemp.vs.values[ixr,iyr,izr]) & \
        (model.vs.values[ixr,iyr,izr] > 966) & (model.vs.values[ixr,iyr,izr] <= 1000)), model.vs.values[ixr, iyr, izr] * 2.125, modelVp.vp.values[ixr, iyr, izr])
    modelVp.vp.values[ixr, iyr, izr] =  np.where(((vs_agrant[ixr,iyr,izr] < modelTemp.vs.values[ixr,iyr,izr]) & \
        (model.vs.values[ixr,iyr,izr] > 1000)), model.vs.values[ixr, iyr, izr] * 2.000, modelVp.vp.values[ixr, iyr, izr])

    # Save new model
    model.to_netcdf('../output/' + model.encoding['source'].split('/')[-1].split('.')[0] + '_gradPL.nc')
    modelVp.to_netcdf('../output/' + modelVp.encoding['source'].split('/')[-1].split('.')[0] + '_gradPL.nc')

def addRockGradient(modelTrue, modelTrueVp):

    """
    Add gradient to sites outside of the Puget Lowland

    This uses the functional form developed by a. grant
    for the M9 ShakeMap paper. Constants are based on
    non-Puget Lowland profiles from the Ahdi et al. (2017) dataset.
    Uses the USGS slope-based Vs30 as an input parameter.

    We use this new function to set Vs from 0-50 m,
    then linearly interpolate between 50 m and 500 m
    (i.e., with Vs @ 500 m set by the Stephenson model).
    Updated to also modify Vp using Vp/Vs=sqrt(3)=1.73

    Parameters:
    model: Sub-model to modify

    Returns:
    None

    """

    # Open dataset
    modelTrue = xr.open_dataset(modelTrue)
    modelTrueVp = xr.open_dataset(modelTrueVp)

    # Deep copy new object (prevents alteration of original)
    model = modelTrue.copy(deep=True)
    modelVp = modelTrueVp.copy(deep=True)

    # Define matrices describing the z index where the topographic surface is encountered
    # and what the VS is at that surface.
    surfVS = np.ones_like(model.vs.values[:,:,0])*-1.0
    surfind = np.ones_like(model.vs.values[:,:,0],dtype='int')*-1
    uno = np.ones_like(model.vs.values[:,:,0])
    for i in range(len(model.z.values)):
        surfind[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfind[:,:]==-1)),uno*i,surfind[:,:])
        surfVS[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfVS[:,:]==-1.0)),model.vs.values[:,:,i],surfVS[:,:])

    # Find sites were 700 m/s<= Vs_surface <1e19
    # Indices to apply new Vs(z)
    ix, iy = np.where((surfVS >= 700) & (surfVS < 1e19))

    # Import new Vs30 (considers slope of bathymetry)
    fvs = '../output/vs30_bathy1.nc'
    q = xr.open_dataset(fvs)
    qinterp = q.interp(utmn=model.utmn,utme=model.utme)
    vs30 = qinterp.vs30.values

    # Define a dz array.
    # This describes the increase in depth between subsequent z indices.
    # Normally, the Vs_puget relation is depth dependent relative to z=0.
    # However with topography, this relation must be considered relative to the ground surface at each XY point.
    # Since spacing in the z-direction is not uniform (e.g., 10m to 100m at depth),
    # the spacing of z-values in the Vs_puget relation may vary between XY points.
    # This array helps to address this disparity when calculating gradient Vs_puget values.
    dzmat = np.zeros_like(model.z.values)
    dzmat = dzmat[1:]
    for i in range(len(model.z.values)-1):
        dzmat[i] = model.z.values[i+1]-model.z.values[i]

    # Amend non-Lowland sites
    # Find non-zero z values closest to 50m and 500m
    ddd1 = 100000
    ddd2 = 100000
    for i in range(len(model.z.values)):
        if((abs(model.z.values[i]-50)<ddd1) & (model.z.values[i]>0)):
            interpStart = model.z.values[i]
            ddd1 = abs(model.z.values[i]-50)
        if(abs(model.z.values[i]-500<ddd2)):
            interpEnd = model.z.values[i]
            ddd2 = abs(model.z.values[i]-500)
    # Slightly worried the vectorized version is unstable... reverted to a loop
    inds_50m = np.zeros_like(ix) # index where depth below surface exceeds 50m
    inds_500m = np.zeros_like(ix) # index where depth below surface exceeds 500m
    modelTemp = model.copy(deep=True)
    modelVpTemp = modelVp.copy(deep=True)
    for i in range(len(ix)):
        zloc = [0]
        zloc2 = [0]
        dnew = 0
        dnew2 = 0
        inds_50m[i] = surfind[ix[i],iy[i]]
        inds_500m[i] = surfind[ix[i],iy[i]]
        for j in range(1,len(model.z.values)):
            dnew += dzmat[surfind[ix[i],iy[i]]+j-1]
            if(dnew <= interpStart):
                zloc = np.append(zloc,dnew)
                inds_50m[i] += 1
            if((dnew<=interpEnd) & (dnew>interpStart)):
                dnew2 += dzmat[surfind[ix[i],iy[i]]+j-1]
                zloc2 = np.append(zloc2,dnew2)
            if(dnew<=interpEnd):
                inds_500m[i] += 1
            else:
                break
        model.vs.values[ix[i],iy[i],surfind[ix[i],iy[i]]:inds_50m[i]+1] = Vs_puget(vs30.T[ix[i],iy[i]], zloc)
        model.vs.values[ix[i],iy[i],surfind[ix[i],iy[i]]+1:inds_50m[i]+1] = model.vs.values[ix[i],iy[i],surfind[ix[i],iy[i]]:inds_50m[i]]
        val_50m_vs = model.vs.values[ix[i],iy[i],inds_50m[i]]
        model.vs.values[ix[i],iy[i],inds_50m[i]+1:inds_500m[i]+1] = np.full_like(zloc2[1:],val_50m_vs) + \
                                   zloc2[1:] * (modelTemp.vs.values[ix[i],iy[i],inds_500m[i]] - val_50m_vs) / (model.z.values[inds_500m[i]] - model.z.values[inds_50m[i]])
        modelVp.vp.values[ix[i],iy[i],surfind[ix[i],iy[i]]:inds_50m[i]+1] = model.vs.values[ix[i],iy[i],surfind[ix[i],iy[i]]:inds_50m[i]+1] * np.sqrt(3)
        val_50m_vp = modelVp.vp.values[ix[i],iy[i],inds_50m[i]]
        modelVp.vp.values[ix[i],iy[i],inds_50m[i]+1:inds_500m[i]+1] = np.full_like(zloc2[1:],val_50m_vp) + \
                                   zloc2[1:] * (modelVpTemp.vp.values[ix[i],iy[i],inds_500m[i]] - val_50m_vp) / (model.z.values[inds_500m[i]] - model.z.values[inds_50m[i]])
      
    # Save new model
    model.to_netcdf('../output/' + model.encoding['source'].split('/')[-1].split('.')[0] + '_gradRock.nc')
    modelVp.to_netcdf('../output/' + modelVp.encoding['source'].split('/')[-1].split('.')[0] + '_gradRock.nc')

def addFillAlluvium(modelTrue, modelTrueVp):
    """
    Add Seattle fill and alluvium to CVM.

    This function uses a file specifying depth to Holocene fill &
    alluvium from K. Troost, modified by A. Frankel.
    Allows for a function that specifies Vs profile with depth.
    Updated to also modify Vp using Vp/Vs=2.5

    Parameters:
    model: Sub-model to modify

    Returns:
    None

    """
    # Open dataset
    modelTrue = xr.open_dataset(modelTrue)
    modelTrueVp = xr.open_dataset(modelTrueVp)

    # Deep copy new object (prevents alteration of original)
    model = modelTrue.copy(deep=True)
    modelVp = modelTrueVp.copy(deep=True)

    dz = model.z.values[1]-model.z.values[0]

    # Load file containing depth to fill & alluvium (in feet)
    ft2m = 0.3048       # Feet to meters
    df = pd.read_table('./datafiles/dtg0201_frankel.asc', delim_whitespace=True,
                       names=['lat','lon','depth_ft'])
    # Convert latitude/longitude to UTM
    p  = Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)
    utme, utmn = p(df.lon.values, df.lat.values, inverse=False)     # ~280 m horizontal spacing

    # Fill -9999s with NaNs & convert to meters
    depth_m = np.where(df.depth_ft == '-9999', np.nan, df.depth_ft*ft2m)

    # Interpolate fill & alluvium data onto CVM grid (approx. same resolution)
    xi, yi = np.meshgrid(np.linspace(min(model.utme),max(model.utme),len(model.utme)),
                         np.linspace(min(model.utmn),max(model.utmn),len(model.utmn)))
    zi = griddata((utme, utmn), depth_m, (xi, yi), method='linear')

    # Fill offshore sites with NaNs; this must be done on interpolated grid!
    result = isOnLand(xi, yi, p)
    zi = np.where(result == False, np.nan, zi)
    # Original file contains small artifacts; replace thicknesses < 5m with NaNs.
    # (Note: this was <1m in the original code, but the rounding step below sends all values <=dz/2 to zero.)
    zi = np.where(zi <=dz/2, np.nan, zi) 
    # Manually remove Magnolia sites (non-Duwamish)
    zi = np.where(xi < 545000, np.nan, zi)

    # Add new Vs profiles
    indices = np.argwhere(~np.isnan(zi))

    for index in indices:
        zrng = dz * round(zi[index[0],index[1]]/dz) # Round to nearest factor of dz 
            # If dz=30, values >15 round to 30; values <15 round to 0
        ztop = -100000000
        for jj in range(len(model.z)):  # find the z value where you encounter the ground surface
            if(model.vs.values[index[1],index[0],jj]>0):
                ztop = model.z.values[jj]
                break
        zset = model.z[(model.z <= ztop+zrng) & (model.z >= ztop)] # extract z values between the surface and bottom of alluvium. NB: values above sea level are negative. 
        zinds = np.argwhere((model.z.values <= ztop+zrng) & (model.z.values >= ztop)) # extract indices associated with zset variable
        zinds.flatten()
        zvals = zset - np.ones_like(zset)*ztop
        model.vs.values[index[1],index[0],zinds] = np.reshape(SEA_Qal(zvals),(-1,1))   # zi(n,e); model.vs(e,n,z)
        modelVp.vp.values[index[1],index[0],zinds] = model.vs.values[index[1],index[0],zinds]*2.5
        zinds2 = zinds + np.ones_like(zinds) # comment out if NOT using buffer layer at surface
        model.vs.values[index[1],index[0],zinds2] = model.vs.values[index[1],index[0],zinds] # comment out if NOT using buffer layer at surface
        modelVp.vp.values[index[1],index[0],zinds2] = modelVp.vp.values[index[1],index[0],zinds] # comment out if NOT using buffer layer at surface

    model.to_netcdf('../output/'+ model.encoding['source'].split('/')[-1].split('.')[0]
                    +'_FA.nc')
    modelVp.to_netcdf('../output/'+ modelVp.encoding['source'].split('/')[-1].split('.')[0]
                    +'_FA.nc')

    return None

def addRandomness(modelTrue, modelTrueVp):

    """
    Add randomness to Puget Lowland

    See Frankel et al., 2018 (M9 Project) We used a 3D random
    fluctuation based on a fractal variation with a horizontal
    correlation distance of 5.0 km and a vertical correlation
    distance of 2.5 km, with a standard deviation of 5%.
    These correlation distances were chosen so that the velocity
    fluctuations occurred over a wide length scale. Sedimentary
    layering would tend to make the horizontal correlation distance
    larger than the vertical. The 5% standard deviation of velocity
    is within the variations observed in boreholes in the Los
    Angeles basin (see Thelen et al., 2006).

    You will need the file with the random perturbations: toprand10
    This is a C binary file. The dimensions are 2000 x 2000 x 13.

    Also modifies Vp using a Vp/Vs ratio of 2.5
    Requires using the near-original version of the model
    to isolate the Puget Lowland points (i.e., this is too hard
    to do once the gradient is in place)

    Parameters:
    model: Sub-model to modify

    Returns:
    None

    """

    # Open dataset
    modelTrue = xr.open_dataset(modelTrue)

    # Deep copy new object (prevents alteration of original)
    model = modelTrue.copy(deep=True)

    # Find range of z values present in velocity model
    minz = min(model.z.values)
    maxz = max(model.z.values)

    # Read in binary file containing random perturbation
    # File contrains ~80M points, only use first 52M (2000x2000x13)
    vrand_dx, vrand_dy, vrand_dz = 200, 200, 100
    vrand_nx, vrand_ny, vrand_nz = 2000, 2000, 13
    vrand_SWcorner = [300000, 5167300] #[411000, 5167300] # [410960, 5167300]    # Long Beach, WA
    vrand = np.fromfile('./datafiles/toprand10', dtype='f4')[0:vrand_nx*vrand_ny*vrand_nz]
    print('Done loading vrand')

    # Interpolate vrand onto current model spacing
    vrand3d = vrand.reshape((vrand_nx, vrand_ny, vrand_nz), order='F')
    vrand_x = np.arange(0, vrand_nx*vrand_dx, vrand_dx)
    vrand_y = np.arange(0, vrand_ny*vrand_dy, vrand_dy)
    vrand_z = np.arange(0, vrand_nz*vrand_dz, vrand_dz)
    points = (vrand_x, vrand_y, vrand_z)

    # Expand randomness dataset to include full set of z values.
    # For above-sea-level, randomness values are mirrored upward
    # relative to the top surface of the matrix.
    # This action is repeated for every 1200m interval
    # For values below 1200m, values are mirrored downward
    # relative to the bottom surface of the matrix
    if((minz<0) | (maxz>1200)):
        rand0 = vrand3d
        vrandz0 = vrand_z
        # get number of times to duplicate randomness matrix
        nup = m.ceil(abs(minz)/1200.0)
        ndown = m.ceil(maxz/1200.0)-1
        if(nup>0):
            rand_flip = rand0
            for i in range(nup):
                rand_flip = np.flip(rand_flip,2) # flip matrix along z axis
                vrand3d = np.concatenate((rand_flip[:,:,:-1],vrand3d),axis=2)
                zvals = (vrandz0 + np.ones_like(vrandz0) * vrandz0[-1] * i) * -1 # above sea level is negative in this model
                zvals = np.flipud(zvals)
                vrand_z = np.concatenate((zvals[:-1],vrand_z))
        if(ndown>0):
            rand_flip = rand0
            for i in range(ndown):
                rand_flip = np.flip(rand_flip,2) # flip matrix along z axis
                vrand3d = np.concatenate((vrandz0,rand_flip[1:,:,:]),axis=2) 
                zvals = (vrandz0 + np.ones_like(vrandz0) * vrandz0[-1] * (i+1)) # below sea level is positive in this model
                vrand_z = np.concatenate((vrand_z,zvals[1:]))
		
    # Shift X and Y arrays to be in proper UTM coordinates
    vrand_x = vrand_x + np.ones_like(vrand_x) * vrand_SWcorner[0]
    vrand_y = vrand_y + np.ones_like(vrand_y) * vrand_SWcorner[1]

    # convert randomness matrix to XArray DataArray, then to XArray Dataset
    randxr = xr.DataArray(vrand3d,coords=[vrand_x,vrand_y,vrand_z],dims=["utme","utmn","z"])
    randxr = xr.Dataset({"randx" : randxr})
    print(randxr.z.values)

    # Interpolate randomness matrix to same sampling as vmodel
    vrand3d_interp = randxr.interp_like(model)

    # Define matrices describing the z index where the topographic surface is encountered
    # and what the VS is at that surface.
    surfVS = np.ones_like(model.vs.values[:,:,0])*-1.0
    surfind = np.ones_like(model.vs.values[:,:,0],dtype='int')*-1
    uno = np.ones_like(model.vs.values[:,:,0])
    for i in range(len(model.z.values)):
        surfind[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfind[:,:]==-1)),uno*i,surfind[:,:])
        surfVS[:,:] = np.where(((model.vs.values[:,:,i]>0) & (surfVS[:,:]==-1.0)),model.vs.values[:,:,i],surfVS[:,:])

    # Find model locations with Vs < 650 m/s at surface
    # Use fill & alluvium version of the model to identify Puget Lowland
    xi, yi = np.where(surfVS<650)

    # Model coordinates to change
    print('Changing values')
    model.vs.values[xi,yi,:] = np.where((model.vs.values[xi,yi,:]>0),model.vs.values[xi,yi,:]+model.vs.values[xi,yi,:]*vrand3d_interp.randx.values[xi,yi,:],model.vs.values[xi,yi,:])

    # Save new model
    model.to_netcdf('../output/' + model.encoding['source'].split('/')[-1].split('.')[0]+'_RAND.nc')

    return None

def SEA_Qal(z):
    """
    Shallow Vs profile for Quaternary alluvium in Seattle

    Vs(z) as ln of depth with static offset (Vs0) and constant skew (B*z)
    Based on 23 profiles of Quaternary alluvium in the Seattle area (Ahdi et al.)
    Written by alex grant 8/3/2020

    Parameters:
    z: Array of depths in meters (topographic surface = 0)

    Returns:
    Vsz: Array of Vs at corresponding depths

    """

    Vs0 = 101.44259911
    B = 1.45718457
    C = 28.88778375

    Vsz = np.zeros_like(z)

    j = 0
    for i in z:
        if i == 0:
            # Set surface Vs to Vs0 (ignore log 0s)
            Vsz[j] = Vs0
        else:
            Vsz[j] = Vs0 +  B*i + C*np.log(i)
        j+=1

    return Vsz

def isOnLand2(xi, yi, p):
    """
    Determines whether a point is onland or offshore

    Parameters:
    xi: Easting locations
    yi: Northing locations
    p: Projection for converting between UTM and lat/lon

    Returns:
    result: Array of booleans; True = 'on land'; False = 'offshore'
    """

    # Convert grid points to lat/lon arrays
    lon, lat = p(xi.ravel(), yi.ravel(), inverse=True)

    # Pull Land area polygons from Cartopy
    land_10m = cart.feature.NaturalEarthFeature('physical','land','10m')
    land_polygons = list(land_10m.geometries())

    # Convert point array to list of Point objects
    points = [Point(point) for point in zip(lon, lat)]
    # keep a numpy array version of the points for processing later
    point_array=np.array(list(zip(lon,lat)))

    # Prepare the land polygons for comparison
    land_polygons_prep = [prep(land_polygon) for land_polygon in land_polygons]

    # Identify points falling within the land polygons
    land = []
    for land_polygon in land_polygons_prep:
        land.extend([tuple(point.coords)[0] for point in filter(land_polygon.covers, points)])
    land = np.array(land)

    # Create boolean list establishing whether or not each lat/lon coordinate is on land
    result = np.isin(point_array,land)
    #isin only works for individual values in the point_array (not pairs of points)
    #Multiplying the columns of result will return whether a pair of lat/lon points is in the Land area:
    result = np.multiply(result[:,0],result[:,1])

    # Include Harbor Island and Port using a polygon
    # (This is necessary because these features are not included in cartopy's
    #  "natural earth features" and won't show up on the map otherwise).
    polygon = [[-122.322785, 47.598963],
        [-122.339846, 47.598963],
        [-122.343128, 47.591386],
        [-122.357871, 47.586727],
        [-122.358123, 47.584573],
        [-122.382221, 47.584573],
        [-122.382221, 47.554944],
        [-122.322785, 47.554944],
        [-122.322785, 47.598963]]
    
    portPath = mpltPath.Path(polygon)
    insidePort = portPath.contains_points(point_array)
    result = np.where(insidePort==1,insidePort,result)

    result = result.reshape(xi.shape)

    return result

def isInLowland(lon, lat):
    """
    Determines whether a point is within a SMALL PORTION OF the Puget Lowland

    Parameters:
    xi: Easting locations
    yi: Northing locations
    p: Projection for converting between UTM and lat/lon

    Returns:
    result: Array of booleans; True = 'in Lowland'
    """
    point = Point(lon, lat)

    lons_vect = [-122.88, -122.88, -122.18, -122.18]
    lats_vect = [47.44, 47.63, 47.63, 47.44]
    lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
    polygon = Polygon(lons_lats_vect) # create polygon

    return polygon.contains(point)

def Vs_puget(Vs30, z):
    Vs0 = -7.89782794 + 0.49847935*Vs30
    B = 1.73336036 + 0.01301348*Vs30
    C = -53.40169168 +   0.23105325 * Vs30

    Vsz = np.zeros_like(z)

    j = 0
    for i in z:
        if i == 0:
            # Set surface Vs to Vs0 (avoid log 0s)
            Vsz[j] = Vs0
        else:
            Vsz[j] = Vs0 + B*i + C*np.log(i)
            if Vsz[j] < Vsz[j-1] and j > 1:
                Vsz[j] = Vsz[j-1] + (Vsz[j-2] - Vsz[j-1])
        j+=1

    return Vsz




if __name__ == '__main__':
        main()
