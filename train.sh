#!/usr/bin/env bash

python3 -B -u main.py \
	--train-root ~/Datasets/LS3D-W/300VW-3D/Trainset \
        --val-root ~/Datasets/LS3D-W/300W-Testset-3D \
	--log-root ~/Experiments/ \
	--image-size 256 \
	--num-workers 4 \
	--lr 1e-3 \
	--lr-patience 50 \
	--epoch 100 \
	--cuda \
	--batch-size 32 \
	--track \
