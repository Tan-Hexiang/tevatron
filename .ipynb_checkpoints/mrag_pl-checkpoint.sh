CUDA_VISIBLE_DEVICES=0 python src/tevatron/mrag/train_pl.py \
  --output_dir save/56_1 \
  --model_name_or_path  model_nq_dpr_new \
  --fid_path nq_reader_base \
  --corpus Tevatron/wikipedia-nq-corpus \
  --train_dataset model_nq_dpr_new/eval/fid.nq.train.fast.jsonl \
  --val_dataset model_nq_dpr_new/eval/fid.nq.dev.fast.jsonl \
  --learning_rate 1e-5 \
  --learning_rate_alpha 1e-3 \
  --n_context 100 \
  --val_n_context 100 \
  --validate_k 10 \
  --train_batch_size 1 \
  --eps 1.2 \
  --alpha 10.0 \
  --alpha_up_constrain 300 \
  --bias 8.0 \
  --warmup_step 7000 \
  --max_epochs 5 \
  --log_every_n_steps 50 \
  --val_check_interval 6000 \
  --limit_val_batches 1.0 \
  --precision 16 \
  --t5_base_tokenizer_path model_t5_base \
  --untie_encoder True \
  --sample_num 1
  # --only_validate True

