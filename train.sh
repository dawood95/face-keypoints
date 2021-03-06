#!/usr/bin/env bash

python3 -B -u main.py \
	--train-root ~/Datasets/LS3D-W/300VW-3D \
	--val-root ~/Datasets/LS3D-W/300W-Testset-3D \
	--log-root ~/Experiments/ \
	--image-size 320 \
	--num-workers 8 \
	--lr 1e-4 \
	--weight-decay 1e-5\
	--lr-patience 10 \
	--epoch 100 \
	--cuda \
	--batch-size 48 \
	--track \
