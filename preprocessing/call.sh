#!/bin/bash

python preprocessing/preprocess_dataset.py -ppcfg /home/jseia/Desktop/thesis/code/stroke-seg/preprocessing/cfg_files/preprocessing_cfg_aisd.yml
# python preprocessing/preprocess_dataset.py -ppcfg /home/jseia/Desktop/thesis/code/stroke-seg/preprocessing/cfg_files/preprocessing_cfg_apis.yml
python preprocessing/preprocess_dataset.py -ppcfg /home/jseia/Desktop/thesis/code/stroke-seg/preprocessing/cfg_files/preprocessing_cfg_apis_no_label.yml
python preprocessing/preprocess_dataset.py -ppcfg /home/jseia/Desktop/thesis/code/stroke-seg/preprocessing/cfg_files/preprocessing_cfg_tbi.yml