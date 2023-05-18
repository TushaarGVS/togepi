#!/bin/sh
#SBATCH -N 1
pip install -r requirements.txt
python3 testPython.py