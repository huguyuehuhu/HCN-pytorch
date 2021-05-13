#!/bin/bash
python main.py  --dataset_dir ./data0/SBU-RGB-D/FOLD_1/ --mode test  --load true --model_name HCN --dataset_name SBU-RGB-D --num 01
python main.py  --dataset_dir ./data0/SBU-RGB-D/FOLD_2/ --mode test  --load true --model_name HCN --dataset_name SBU-RGB-D --num 02
python main.py  --dataset_dir ./data0/SBU-RGB-D/FOLD_3/ --mode test --load true --model_name HCN --dataset_name SBU-RGB-D --num 03
python main.py  --dataset_dir ./data0/SBU-RGB-D/FOLD_4/ --mode test  --load true --model_name HCN --dataset_name SBU-RGB-D --num 04
python main.py  --dataset_dir ./data0/SBU-RGB-D/FOLD_5/ --mode test  --load true --model_name HCN --dataset_name SBU-RGB-D --num 05
