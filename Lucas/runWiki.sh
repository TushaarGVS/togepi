#!/bin/sh
#SBATCH -N 1
pip3 install -r requirements.txt
echo "out of first requirements"
pip3 install -r req2.txt
echo "out of second requirements"
python3 installation.py
echo "installation ran"