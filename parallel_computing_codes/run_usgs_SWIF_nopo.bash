#!/bin/bash

#SBATCH -p t1small
#SBATCH --ntasks-per-node=24
#SBATCH -t 01:00:00
#SBATCH --array=0-11%1

#SBATCH --output=OUTPUT_FILES/%j.o
#SBATCH --job-name=run_puget_scaling

umask 0022

set -x

module purge
module load Trelis
module load lang/Python/2.7.12-pic-intel-2016b
python --version

cd $SLURM_SUBMIT_DIR 
RDIR=$SLURM_SUBMIT_DIR 

# USER PARAMETERS (ALSO -t ABOVE FOR NUMBER OF NODES)
CORES=24
FTAG=usgs_SWIF_nopo

tstart=$(( $SLURM_ARRAY_TASK_ID * $CORES ))
tend=$(( ( $SLURM_ARRAY_TASK_ID + 1 ) * $CORES - 1 ))

#echo "$SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "tstart: $tstart"
echo "tend: $tend"

# do NOT delete the run directories, since the job is launched onto multiple nodes
mkdir -p $FTAG

# copy files into output directory
cp $RDIR/$FTAG.cfg $FTAG

for ii in $( seq $tstart $tend); do
$RDIR/GEOCUBIT.py --mesh --build_volume --cfg="$RDIR/$FTAG.cfg" --id_proc=$ii > $FTAG/$ii.log &
done
wait

##---------------------------
## MERGE

cd $RDIR/$FTAG

# get the number of cores/slices from the configuration file
nxi=$(awk '{if (/number_processor_xi/) print $2}' FS='=' ./${FTAG}.cfg)
neta=$(awk '{if (/number_processor_eta/) print $2}' FS='=' ./${FTAG}.cfg)
nproc=$[nxi*neta-1]
echo 'nxi , neta , nproc = ' $nxi ' , ' $neta ' , '$nproc
$RDIR/GEOCUBIT.py --collect --merge --meshfiles=mesh_vol_*.e --cpux=$nxi --cpuy=$neta
sleep 2s
$RDIR/GEOCUBIT.py --export2SPECFEM3D --meshfiles=$RDIR/$FTAG/TOTALMESH_MERGED.e

echo "job finished"
