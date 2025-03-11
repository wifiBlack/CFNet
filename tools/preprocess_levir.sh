#! /bin/bash
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/train/A /home/wufan/Datasets/LEVIR-CD-processed/train/A
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/train/B /home/wufan/Datasets/LEVIR-CD-processed/train/B
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/train/label /home/wufan/Datasets/LEVIR-CD-processed/train/label
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/val/A /home/wufan/Datasets/LEVIR-CD-processed/val/A
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/val/B /home/wufan/Datasets/LEVIR-CD-processed/val/B
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/val/label /home/wufan/Datasets/LEVIR-CD-processed/val/label
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/test/A /home/wufan/Datasets/LEVIR-CD-processed/test/A
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/test/B /home/wufan/Datasets/LEVIR-CD-processed/test/B
python split_images_with_overlap.py /home/wufan/Datasets/LEVIR-CD/test/label /home/wufan/Datasets/LEVIR-CD-processed/test/label
bash list_all.sh /home/wufan/Datasets/LEVIR-CD-processed/train/A /home/wufan/Datasets/LEVIR-CD-processed/list/train.txt
bash list_all.sh /home/wufan/Datasets/LEVIR-CD-processed/val/A /home/wufan/Datasets/LEVIR-CD-processed/list/val.txt
bash list_all.sh /home/wufan/Datasets/LEVIR-CD-processed/test/A /home/wufan/Datasets/LEVIR-CD-processed/list/test.txt