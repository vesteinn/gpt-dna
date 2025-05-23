#!/bin/bash


# Launch the distributed training job
python -m torch.distributed.launch \
  --nproc_per_node=4 \
    /data/scratch/vesteinn/dna/transformers/examples/pytorch/language-modeling/run_clm.py \
  --model_type gpt2 \
  --config_overrides="vocab_size=5,n_positions=1024" \
  --tokenizer_name ./dna_tokenizer \
  --train_file ./processed_dna_data/train.txt \
  --validation_file ./processed_dna_data/validation.txt \
  --do_train \
  --do_eval \
  --fp16 \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 3 \
  --eval_steps 5000 \
  --eval_strategy steps \
  --load_best_model_at_end True \
  --metric_for_best_model "loss" \
  --greater_is_better False \
  --num_train_epochs 10 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --learning_rate 3e-4 \
  --warmup_steps 1000 \
  --output_dir ./dna_model \
  --report_to wandb \
  --overwrite_output_dir \
  --logging_steps 100 \
  --block_size 1024 \
  --dataloader_num_workers 4 \
  --ddp_find_unused_parameters False \
  --torch_dtype bfloat16 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --logging_first_step \
  --keep_linebreaks False