#!/bin/bash
#COBALT -n 1
#COBALT -q gpu_v100_smx2
#COBALT -A Performance
#COBALT -t 2:00:00

# Activate python - first Taylor's JLSE Horovod build
source /gpfs/jlse-fs0/projects/datascience/parton/mlcuda/mconda/setup.sh
# Then my package additions
conda activate /home/rmaulik/link_dir/rmaulik/DayMet/daymet_gpu_env

cd src/
python ROM.py --geo_data tmax --comp cae --train_mode --train_space --win 7
cd ..
