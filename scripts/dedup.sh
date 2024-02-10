#!/bin/bash
python -m text_dedup.minhash \
  --local \
  --path "./raw_data/train/korquad_1.0_processed" \
  --split "all" \
  --cache_dir "./cache" \
  --output "./raw_data/train/korquad_1.0_processed_dedup" \
  --column "context" \
  --batch_size 10000