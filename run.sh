#!/bin/bash
#COBALT -n 1
#COBALT -q gpu_v100_smx2
#COBALT -A Performance
#COBALT -t 2:00:00

# Activate python
source activate daymet_env
cd src/
python ROM.py --geo_data tmax --comp cae --train_mode --train_space --win 7
cd ..
source deactivate
