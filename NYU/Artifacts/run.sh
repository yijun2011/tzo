#!/usr/bin/bash
echo "Activating tensorflow environment ..."
source ./tensorflow/bin/activate
echo "Launching $1 ..."
dd=$(date +"%Y%m%d_%H%M%S")
cp $1 ../Models; nohup python3 $1 > ../Models/Log_${dd}.txt 2>&1 &
