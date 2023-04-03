CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.launch --nproc_per_node=1 -m tevatron.mrag.train \
  --output_dir model_nq \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_name Tevatron/wikipedia-nq \
  --fp16 \
  --per_device_train_batch_size 32 \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --negatives_x_device \
  --overwrite_output_dir >log/dpr_nq_train.log 2>&1 &