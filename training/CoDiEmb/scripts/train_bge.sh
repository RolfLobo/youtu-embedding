#!/bin/bash
set -ex

BASE_DIR=$BASE_DIR  # needs to be modified according to the actual situation
IR_DATA_PATH=$BASE_DIR/CoDiEmb/data/ir_example_data
STS_DATA_PATH=$BASE_DIR/CoDiEmb/data/sts_example_data
export HF_DATASETS_CACHE=$IR_DATA_PATH/cache
export WANDB_DISABLED=true

model_name_or_path=$BASE_DIR/model/bge-large-zh-v1.5
OUTPUT_PATH=$BASE_DIR/output/bge_model

mkdir -p $OUTPUT_PATH
LOG_PATH=$OUTPUT_PATH
mkdir -p $LOG_PATH/logs

PROJECT_PATH=$BASE_DIR/CoDiEmb
DS_PATH=$PROJECT_PATH/scripts/deepspeed_config_fp32_zero1.json
cd $PROJECT_PATH

LAUNCHER="python3 -m torch.distributed.run \
    --nnodes $HOST_NUM \
    --node_rank $INDEX \
    --nproc_per_node $HOST_GPU_NUM \
    --master_addr $CHIEF_IP \
    --master_port 29500 \
    train/run.py \
    --deepspeed ${DS_PATH}"

export CMD=" \
    --output_dir ${OUTPUT_PATH} \
    --model_name_or_path ${model_name_or_path} \
    --cache_dir ${HF_DATASETS_CACHE} \
    --seed 49 \
    --ir_train_data ${IR_DATA_PATH} \
    --sts_train_data ${STS_DATA_PATH} \
    --learning_rate 5e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --ir_per_device_batch_size 64 \
    --sts_per_device_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --ir_negatives_cross_device \
    --dataloader_drop_last \
    --data_sampler dynamic \
    --normalized \
    --temperature 0.02 \
    --multi_layer_loss \
    --positive_group_size 2 \
    --negative_group_size 4 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --logging_steps 10 \
    --pooling_method cls \
    --attn bbcc \
    --attn_implementation eager \
    --save_strategy "epoch" \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --skip_filter_too_long_instruction"
set +e

sh -c "$LAUNCHER $CMD" 2>&1 | tee $LOG_PATH/logs/${INDEX}.log
