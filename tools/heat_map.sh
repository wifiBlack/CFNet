#! /bin/bash

python heat_map.py\
 --data-dir /home/wufan/Datasets/_CLCD\
 --gpu 3\
 --batch-size 16\
 --num-workers 8\
 --checkpoint checkpoints/cl-art.pth