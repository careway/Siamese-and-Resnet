#!/bin/sh


module load python3
module load cuda/10.0

CUDA_VISIBLE_DEVICES=1 python3 TrSiamese.py
