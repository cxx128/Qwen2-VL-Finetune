#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

# You can use 2B instead of 7B
MODEL_NAME="/mnt/afs/chenxiaoxuan/hf_home/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/data/train_data_1130_part_format.json \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/debug_1215 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --min_pixels $((512 * 28 * 28)) \
    --max_pixels $((1024 * 28 * 28)) \
    --max_seq_length 2048 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --dataloader_num_workers 4
    #--world_size 8 \
    #--image_folder /mnt/afs/chenxiaoxuan/Projects_2024_cxx/data_image_draw_taikai \