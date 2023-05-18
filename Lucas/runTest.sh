#!/bin/sh
#SBATCH -N 1
pip3 install -r requirements.txt
python3 testPython.py > /mnt/beegfs/bulk/stripe/lm865/tests/Fluzao.txt