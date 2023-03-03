"""
Calculates Spectral Acceleration data from Specfem seismograms.
This version is designed specifically to run in parallel on 
the USGS' Denali Supercomputer. Because of the large number
of seismograms processed (>10000), running in parallel
reduces processing time by several orders of magnitude.

Utilizes pyrotd (instead of your own rotD50 method, as in calcCGOF2.py).
to calculate the rotated SA

For SA, uses Newmarks: Linear System, assuming Linear Acceleration Method
From Chopra, Page 177

INPUTS:
- Specfem seismogram files
- Specfem stations file

OUTPUTS:
- ASCII files with SA at each station at a select range of periods

"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from scipy import signal as s
import pyrotd
import time
import multiprocessing

# -------------------------------------
# CHANGE THESE!!!

# Name of output spectral acceleration file directory
dirname="/caldera/projects/usgs/hazards/ehp/istone/denali_gnu/swif_M7_1_1_topo/OUTPUT_FILES_swif_M7_1_1_topo/SA_swif_M7_1_1_topo/"

# path to seismogram records
dirt='/caldera/projects/usgs/hazards/ehp/istone/denali_gnu/swif_M7_1_1_topo/OUTPUT_FILES_swif_M7_1_1_topo/seismograms'

# Stations file
fstat = "./DATA/STATIONS_FILTERED"

# sample rate of synthetics (Hz)
sdt = 1.0/(0.0003*50)

# Frequency min and max for filtration of data (Hz)
fmin=0.05
fmax=3.2

# Range of periods T for spectral Response (s)
Tar=[0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.5,10]

# Damping Ratio
Zeta = 0.05
# -------------------------------------

chans=['FXZ','FXY','FXX']
comp='semv'

# Read in stations file
statArray = np.loadtxt(fstat,dtype='str')
sta = statArray[:,0]
net = statArray[:,1]

def multiprocessing_func(i):
    # Load seismograms
    tstN="%s/%s.%s.%s.%s" % (dirt,net[i],sta[i],chans[1],comp)
    tstE="%s/%s.%s.%s.%s" % (dirt,net[i],sta[i],chans[2],comp)
    
    dattN=np.loadtxt(tstN)
    dattE=np.loadtxt(tstE)
   
    # Filter seismograms to appropriate range 
    sost = s.butter(5,[fmin,fmax],'bandpass',fs=sdt,output='sos')
    ydattN=s.sosfilt(sost,dattN[:,1])
    ydattE=s.sosfilt(sost,dattE[:,1])

    # Convert seismograms from velocity (m/s) to acceleration (G)
    g=9.8
    ydtaN=deriv(ydattN,1.0/sdt)/g
    ydtaE=deriv(ydattE,1.0/sdt)/g
    
    # Use pyrotd to calculate the rotated PSA
    satop = pyrotd.calc_rotated_spec_accels(1.0/sdt,ydtaN,ydtaE,np.divide(1,Tar),Zeta,percentiles=[50])
    
    # Save array to file
    sas=satop['spec_accel']
    sat=np.divide(1,satop['osc_freq'])
    fstr=dirname+str(sta[i])
    np.savetxt(fstr,np.column_stack((sat,sas)),fmt='%5.2f %14.10f')

def deriv(fun,dt):
    # takes first, central derivative of function fun with uniform sampling dt (s).
    # returns numerical derivative aa
    aa=np.zeros(fun.shape)
    for i in range(1,len(fun)-1):
        aa[i]=(fun[i+1]-fun[i-1])/2/dt
    # assign values to first and last indices
    aa[0]=aa[1]
    aa[len(fun)-1]=aa[len(fun)-2]

    return aa

if __name__ == '__main__':

    starttime = time.time()
    processes = []
    for i in range(0,len(sta)):
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()
        if(i%500==0):
            for process in processes:
                process.join()
            processes=[]        

    print('That took {} seconds'.format(time.time() - starttime))

