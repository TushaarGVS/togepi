#!/bin/sh
#SBATCH -N 1
#SBATCH --mem=16000
pip3 install torchinfo
python3 timeTest.py