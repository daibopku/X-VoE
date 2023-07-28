#!/usr/bin/env bash

# download dataset
bash scripts/download_dataset.sh

# download checkpoint
bash scripts/download_pretrained_model.sh 

# prepare image dataset for pretrain perception model
python -m scripts.make_video_data

# pretrain perception model
python -m train.train_perception

# train xpl model
# python -m train.train_xpl

# train plato model
# python -m train.train_plato

# eval xpl model
python -m eval.eval_xpl

# eval plato model
python -m eval.eval_plato