#!/usr/bin/env bash

python3 -B -u main.py \
	--train-root ~/Datasets/LS3D-W/300VW-3D \
        --val-root ~/Datasets/LS3D-W/300W-Testset-3D \
	--log-root ~/Experiments/ \
	--image-size 256 \
	--num-workers 2 \
	--lr 1e-3 \
	--lr-patience 100 \
	--epoch 300 \
	--cuda \
	--batch-size 32 \
	--track \
