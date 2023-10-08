#!/usr/bin/env bash

bash scripts/finetune_real.sh "cat" /home/rlpo/custom-diffusion/data/cat real_reg/samples_cat  cat_lora_r30 finetune_addtoken_lora.yaml /home/rlpo/custom-diffusion/sd-v1-4.ckpt
