CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 -m tevatron.mrag.train \
  --corpus Tevatron/wikipedia-nq-corpus \
  --output_dir  /data/tanhexiang/tevatron/412 \
  --model_name_or_path  /data/tanhexiang/tevatron/model_nq \
  --save_steps 20000 \
  --dataset_name /data/tanhexiang/tevatron/data_nq/result100/fid.nq.train.jsonl \
  --fp16 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-5 \
  --dpr_query_len 32 \
  --dpr_passage_len 156 \
  --num_train_epochs 4 \
  --logging_steps 500 \
  --logging_first_step \
  --overwrite_output_dir \
  --alpha 0.05