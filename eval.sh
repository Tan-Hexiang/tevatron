
for s in $(seq -f "%02g" 0 19)
do
CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
  --output_dir=49/eval/embs \
  --model_name_or_path 49 \
  --fp16 \
  --per_device_eval_batch_size 16 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus_emb.$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done