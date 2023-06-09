#!/bin/bash

# set environmental variables
export OMP_NUM_THREADS=1
export nnUNet_raw='<some_PATH>/stroke-seg/nnUnet/nnunetv2/nnUNet_raw'
export nnUNet_preprocessed='<some_PATH>/stroke-seg/nnUnet/nnunetv2/preprocessed'
export nnUNet_results='<some_PATH>/stroke-seg/nnUnet/nnunetv2/nnUNet_trained_models'

# adjust the training cfg file accordingly
cp '<nnUNet_preprocessed_PATH>/Dataset046_AIS/nnunet_cfg_fr_0.yml' \
    '<nnUNet_preprocessed_PATH>/Dataset046_AIS/nnunet_cfg.yml'

# run training
nnUNetv2_train Dataset046_AIS 3d_fullres 0 -tr nnUNetTrainerCfg -p nnUNetPlansSSL \
    -pretrained_weights '<pretrain_model_PATH>/ssl_checkpoint.pth'

# run predictions over the validation set
splits="<splits_PATH>/raw_splits_final_full.json"
out_path="<out_PATH>"
model_path="<model_PATH>"
for i in best
do
    nnUNetv2_predict_from_modelfolder -i "${splits}" -o "${out_path}/validation_${i}" -m "${model_path}" \
        -f 0 --save_probabilities -chk "checkpoint_${i}.pth" -npp 3 -nps 3 -device 'cuda'
done

# rename the results path
mv '<nnUNet_results_PATH>/Dataset046_AIS' \
    '<nnUNet_results_PATH>/Dataset046_AIS_example'