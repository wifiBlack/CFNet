#! /bin/bash
python test.py\
 --data-dir /home/wufan/Datasets/_CLCD\
 --gpu 0\
 --batch-size 8\
 --num-workers 8\
 --checkpoint logs_1_1_cl/run_0000/epoch41.pth