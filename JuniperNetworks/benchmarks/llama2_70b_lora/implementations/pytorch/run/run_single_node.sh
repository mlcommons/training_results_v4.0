export CODE=/path/to/code
export MODEL=/path/to/model
export DATA=/path/to/data
export RESULTS=/path/to/results directory


accelerate launch --config_file $CODE/configs/ds_config.yaml \
scripts/train.py \
--dataset_path $DATA \
--model_path $MODEL \
--max_seq_len 8192 \
--bf16 True \
--logging_steps 10 \
--eval_steps 12 \
--output_dir $RESULTS \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type "cosine" \
--learning_rate 3e-4 \
--weight_decay 0.0001 \
--warmup_ratio 0.0 \
--max_grad_norm 0.3 \
--use_gradient_checkpointing True \
--target_eval_loss 0.925 \
--use_peft_lora True \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.1 \
--max_steps 1024 \
--use_flash_attn \
--lora_target_modules "qkv_proj,o_proj"