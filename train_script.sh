#!/usr/bin/env bash
# bash scripts/finetune_real.sh "cat" /home/rlpo/custom-diffusion/data/cat real_reg/cat cat_cd finetune_addtoken.yaml /home/rlpo/custom-diffusion/sd-v1-4.ckpt

# Paste your log folder here for generating samples
logdir=logs/2023-10-24T22-15-38_joe-sdv4

# Linear
# bash scripts/finetune_real.sh "person" ../../datasets/diffusion_finetuning_data/joe politicians joe finetune_addtoken.yaml ../../datasets/sd-v1-4.ckpt
bash scripts/finetune_gen.sh "person" ../../datasets/diffusion_finetuning_data/joe politicians joe finetune_addtoken.yaml ../../datasets/sd-v1-4.ckpt

# LoRA
# bash scripts/finetune_real.sh "person" ../../datasets/diffusion_finetuning_data/joe politicians joe finetune_addtoken_lora.yaml ../../datasets/sd-v1-4.ckpt 

# LoRSA
# bash scripts/finetune_real.sh "person" ../../datasets/diffusion_finetuning_data/joe politicians joe finetune_addtoken_lorsa.yaml ../../datasets/sd-v1-4.ckpt 


# python src/get_deltas.py --path $logdir/checkpoints/last.ckpt --newtoken 1 
# python sample.py --prompt "<new1> person eating an ice cream cone" --delta_ckpt $logdir/checkpoints/last.ckpt --ckpt ../../datasets/sd-v1-4.ckpt
