#!/usr/bin/env bash

bash scripts/finetune_real.sh "cat" data/cat real_reg/samples_cat  cat_lora_r200 finetune_addtoken_lora.yaml sd-v1-4.ckpt

# bash scripts/finetune_real.sh "barn" data/barn real_reg/samples_barn  barn finetune_addtoken.yaml sd-v1-4.ckpt

# bash scripts/finetune_real.sh "dog" data/dog real_reg/samples_dog  dog finetune_addtoken.yaml sd-v1-4.ckpt

# bash scripts/finetune_real.sh "teddy bear" data/teddybear real_reg/samples_teddybear teddybear finetune_addtoken.yaml sd-v1-4.ckpt

# bash scripts/finetune_real.sh "tortoise plushy" data/tortoise_plushy real_reg/samples_tortoise_plushy tortoise_plushy finetune_addtoken.yaml sd-v1-4.ckpt

# bash scripts/finetune_real.sh "wooden pot" data/wooden_pot real_reg/samples_wooden_pot wooden_pot finetune_addtoken.yaml sd-v1-4.ckpt
