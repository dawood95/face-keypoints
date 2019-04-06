#!/usr/bin/env bash

python3 -B -u main.py \
	--train-root ~/Datasets/LS3D-W/300VW-3D/Trainset \
        --val-root ~/Datasets/LS3D-W/300VW-3D/CatA \
	--log-root ~/Experiments/ \
	--image-size 320 \
	--num-workers 4 \
	--lr 1e-4 \
	--weight-decay 1e-5\
	--lr-patience 50 \
	--epoch 100 \
	--cuda \
	--batch-size 64 \
	--track \
