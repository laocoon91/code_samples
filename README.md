# code_samples
Selection of coding examples from projects I am working on.

Codes in this directory are meant to demonstrate proficiencies in various languages (mainly python and shell script) and their associated packages. Below I briefly describe the contents of each directory, as well as the main packages used. 

Machine_learning_workflow
These scripts are used for running a PhaseNet detection of earthquakes in PNSN data. Data are downloaded as miniseed files from IRIS (downloadMseed.ipynb); processed for input into PhaseNet (mseedProcess4PhaseNet_PNSNData.ipynb); fed into PhaseNet for P and S phase detection (runPhaseNet_multistation.ipynb). The resulting triggers are fed into the REAL association algorithm for event association (runREAL_phasenet.ipynb). Packages used include:
Python:
-numpy
-obspy
-tensorflow
-shutil and subprocess

Parallel_computing_codes
These scripts are ones I’ve written to aid in running code in parallel computing settings. This includes a specialized python script for Spectral Acceleration calculation (calcCGOF3.py and run_calcCGOF3.bash), as well as a bash script used to build a CUBIT mesh for use in SPECFEM3D (run_usgs_SWIF_nopo.bash). Packages used include:
Python:
-numpy
-scipy
-multiprocessing

Vmodel_adjustment_codes
These scripts adjust the Cascadia Velocity Model (Stephenson et al., 2017) to consider near-surface geotechnical gradients (geotech_4_topo.py), surface topography (topography_sw4.py), and low velocity fault zones (addFaultZone.ipynb and fault_zone_functions.py). These handle very large netCDF files and rely heavily upon xarray/numpy for data manipulation. Packages used include:
Python:
-numpy
-xarray
-pandas
-cartopy
-pyproj
-shapely
-scipy
-matplotlib

Ian Stone, March 2022
![image](https://user-images.githubusercontent.com/16617064/222844637-14b999cf-afd3-4306-b323-c2dd61f1a686.png)
