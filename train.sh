#! /bin/bash
python run.py --data-dir CLCD-processed \
              --log-dir logs_clcd \
              --gpu 0 \
              --epochs 100 \
              --batch-size 32 \
              --num-workers 16 \
              --lr 0.0005 \