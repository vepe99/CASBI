#!/bin/sh
torchrun --standalone --nproc_per_node=2  distributed_training.py 1 1