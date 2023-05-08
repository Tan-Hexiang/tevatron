# # kl train model_trivia with adist
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 -m tevatron.fidkd.train \
  --output_dir model_kl/trivia_adist \
  --model_name_or_path model_trivia \
  --corpus Tevatron/wikipedia-trivia-corpus \
  --save_steps 20000 \
  --dataset_name data_trivia/adist_score/trivia_train_adist_100/dataset_wscores.jsonl \
  --depth 100 \
  --fp16 \
  --per_device_train_batch_size 1 \
  --positive_passage_no_shuffle \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --negatives_x_device



# # json to jsonl(adist)
# python -m tevatron.utils.format.convert_json_dir_to_jsonl --json_dir data_trivia/adist_score/trivia_train_adist_100