#!/bin/sh
### ------------- specify queue name ----------------
#BSUB -q c02516
### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"
### ------------- specify job name ----------------
#BSUB -J eval_prop
### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 08:00

#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

source .venv/bin/activate
python evaluate_proposals.py