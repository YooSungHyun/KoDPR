#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=false
export WANDB_PROJECT="kodpr"
export WANDB_ENTITY="bart_tadev"
export WANDB_NAME="klue-unique-256"
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"

deepspeed --include localhost:0,1,2,3 --master_port 61000 ./ds_train.py \
    --transformers_model_name="team-lucid/deberta-v3-base-korean" \
    --train_datasets_path="./raw_data/train/total_preproc.json" \
    --eval_datasets_path="./raw_data/dev/total_preproc.json" \
    --output_dir=ds_outputs/ \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=256 \
    --accumulate_grad_batches=1 \
    --max_epochs=150 \
    --learning_rate=1e-5 \
    --weight_decay=0.0001 \
    --warmup_ratio=0.01 \
    --div_factor=10 \
    --final_div_factor=10 \
    --dataloader_drop_last=False \
    --sampler_shuffle=True \
    --log_every_n=10 \
    --deepspeed_config=ds_config/zero2.json \
    --metric_on_cpu=false