#! /bin/bash
python run.py --data-dir /home/wufan/Datasets/_CLCD\
              --log-dir logs\
              --gpu 0\
              --epochs 100\
              --batch-size 8\
              --num-workers 8\
              --lr 0.00001\
              --checkpoint logs/run_0000/epoch41.pth