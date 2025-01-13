cd /mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune #为什么这句话没用


#!/bin/bash

# source  ~/env.sh
DIR=`pwd`
#--video_data  /work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/e_dataset/video_test_50frames.json \
MODEL_NAME="/mnt/afs/chenxiaoxuan/hf_home/Qwen2-VL-7B-Instruct"
python src/training/check_data.py \
    --deepspeed /mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen_352k_0923\
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_pixels $((1440 * 28 * 28)) \
    --max_seq_length 2800 \
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
    --save_steps 5 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 \
    --data_path /mnt/afs/chenxiaoxuan/modality_alignment/Qwen2-VL-Finetune/data/train_data_1130_part_format.json \