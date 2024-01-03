#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ./src/THUDM/chatglm3-6b/ \
    --dataset medical \
    --template chatglm3_raw \
    --finetuning_type lora \
    --lora_target query_key_value \
    --output_dir output/ \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-4 \
    --num_train_epochs 2.0 \
    --plot_loss True \
    --fp16 True

