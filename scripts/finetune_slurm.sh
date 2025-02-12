#!/bin/bash

# source  ~/env.sh

ndcus=${1:-8} 
model=${2:-blip2}

echo The model is $model

jobname=${3:-test}
echo The jobname is $jobname

output=./log
GPUS_PER_NODE=${ndcus:-4}
if [ $GPUS_PER_NODE -ge 4 ]; then
  GPUS_PER_NODE=4
fi

time=$(date "+%Y%m%d-%H%M%S")

echo The output dict is ${output}

# mkdir -p ${output}/${model}/${jobname}

work_dir=${output}/${model}/${jobname}/${time}
mkdir -p $work_dir

partition=xahdnormal
#partition=xahdtest

DIR=`pwd`

# MODEL_NAME="/work/home/acehekbmzh/data/hf_home/Qwen2-VL-7B-Instruct"
MODEL_NAME="/work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/output/qwen_351k_0925_fp32"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

#fp16/bf16 offload ds_config:/work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/scripts/zero3_offload.json
#fp16/bf16 ds_config:/work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/scripts/zero3.json
#fp32 ds_config:/work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/e_dataset/ds_config_zero3_fp32.json


#可调节图片大小：
#--min_pixels $((256 * 28 * 28)) \
#--max_pixels $((2048 * 28 * 28)) \
#    --video_data  /work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/e_dataset/video_test_50frames.json \
set -x 
srun -p ${partition} -n $ndcus -N 8 --gres=dcu:${GPUS_PER_NODE} -o $work_dir/exp.log  --job-name=${jobname}  \
--cpus-per-task=8  --kill-on-bad-exit=1 --ntasks-per-node=${GPUS_PER_NODE} \
python src/training/train_slurm.py \
    --deepspeed /work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/e_dataset/ds_config_zero3_fp32.json \
    --model_id $MODEL_NAME \
    --data_path /work/home/acehekbmzh/wjh/Qwen2-VL-Finetune/e_dataset/hongwai_cal_1129.json \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger True \
    --bf16 False \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir output/hongwai_cal_1129 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_pixels $((1024 * 28 * 28)) \
    --max_seq_length 2048 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 8 \
    --save_total_limit 1 \
    --dataloader_num_workers 4 