#!/bin/bash

export nnUNet_raw='<PATH_TO_PROJECT>/nnUnet/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed='<PATH_TO_PROJECT>/nnUnet/nnunetv2/preprocessed'
export nnUNet_results='<PATH_TO_PROJECT>/nnUnet/nnunetv2/nnUNet_trained_models'

python '<PATH_TO_PROJECT>/nnUnet/ssl/main_ssl.py' \
    -cfg '<PATH_TO_PROJECT>/nnUnet/ssl/cfg_files/example.yml'
