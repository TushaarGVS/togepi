#!/bin/sh
#SBATCH -N 1
pip3 install -r requirements.txt
python3 testPython.py > /Fluzao.txt