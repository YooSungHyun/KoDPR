#!/bin/bash
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=false
export WANDB_PROJECT="kodpr"
export WANDB_ENTITY="bart_tadev"
export WANDB_NAME="korquad-klue-bm25-127+128"
export HUGGINGFACE_HUB_CACHE="./.cache"
export HF_DATASETS_CACHE="./.cache"

deepspeed --include localhost:0,1,2,3 --master_port 61000 ./ds_bm25_train.py \
    --transformers_model_name="team-lucid/deberta-v3-base-korean" \
    --train_datasets_path="./raw_data/train/200k_preproc_bm25idx.json" \
    --indices_path="./raw_data/train/200k_preproc_bm25_sampler_indices.json" \
    --eval_datasets_path="./raw_data/dev/total_preproc.json" \
    --output_dir=ds_outputs/ \
    --seed=42 \
    --num_workers=12 \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --accumulate_grad_batches=1 \
    --max_epochs=200 \
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