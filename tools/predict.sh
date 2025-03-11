#! /bin/bash
python predict.py\
 --data-dir /home/wufan/Datasets/SYSU-CD\
 --gpu 3\
 --batch-size 32\
 --num-workers 16\
 --checkpoint checkpoints/sysu-art.pth\
 --pred-dir /home/wufan/Predictions/SYSU-pred\