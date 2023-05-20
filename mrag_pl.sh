CUDA_VISIBLE_DEVICES=0 python src/tevatron/mrag/train_pl.py \
  --output_dir save/520_constrain_l0_and_negative \
  --model_name_or_path  model_nq_dpr_new \
  --fid_path nq_reader_base \
  --corpus Tevatron/wikipedia-nq-corpus \
  --train_dataset model_nq_dpr_new/eval/fid.nq.train.fast.jsonl \
  --val_dataset model_nq_dpr_new/eval/fid.nq.dev.fast.jsonl \
  --learning_rate 1e-5 \
  --learning_rate_alpha 1e-2 \
  --n_context 100 \
  --val_n_context 100 \
  --validate_k 10 \
  --train_batch_size 1 \
  --eps 0.5 \
  --alpha 1.0 \
  --alpha_up_constrain 1000 \
  --bias 0.0 \
  --warmup_step 2000 \
  --max_epochs 1 \
  --log_every_n_steps 50 \
  --val_check_interval 1000 \
  --limit_val_batches 0.5 \
  --precision 32 \
  --t5_base_tokenizer_path model_t5_base \
  --untie_encoder True \
  --sample_num 1 \
  --temperature 1.0 \
  --negative_ratio 0.5
  # --only_validate True



