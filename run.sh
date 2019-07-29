#!/bin/bash

## BASE
for MIS in 50
do
for T in 0.3
do
python3 main.py --ver "${MIS}-${T}" --train 1 --gpu 0 --min_instances_slice $MIS --threshold $T
done
done
