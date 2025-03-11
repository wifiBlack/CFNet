#! /bin/bash
python run.py --data-dir /home/wufan/Datasets/_CLCD\
              --log-dir logs_1_1_cl\
              --gpu 0\
              --epochs 50\
              --batch-size 8\
              --num-workers 8\
              --lr 0.00001\
              --checkpoint logs_1_1_cl/run_0000/epoch41.pth