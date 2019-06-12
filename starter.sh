#!/bin/sh


module load python3
module load cuda/10.0

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 Trainer.py
