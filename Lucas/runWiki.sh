#!/bin/sh
#SBATCH -N 1
pip3 install -r requirements.txt
pip3 install -r req2.txt
python3 installation.py