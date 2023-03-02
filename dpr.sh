### Encode Corpus  $ENCODE_DIR
# mkdir temp
# for s in $(seq -f "%02g" 0 19)
# do
# python -m tevatron.driver.encode \
#   --output_dir=temp \
#   --model_name_or_path model_nq \
#   --fp16 \
#   --per_device_eval_batch_size 156 \
#   --dataset_name Tevatron/wikipedia-nq-corpus \
#   --encoded_save_path corpus_emb.$s.pkl \
#   --encode_num_shard 20 \
#   --encode_shard_index $s
# done

# sharded search
INTERMEDIATE_DIR=intermediate_nq_train
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
  python -m tevatron.faiss_retriever \
  --query_reps data/query_embs/nq_train.pkl \
  --passage_reps data/embs/corpus_emb.${s}.pkl \
  --depth 1000 \
  --save_ranking_to ${INTERMEDIATE_DIR}/${s}
done

# python -m tevatron.faiss_retriever.reducer \
# --score_dir ${INTERMEDIATE_DIR} \
# --query data/query_embs/nq_train.pkl \
# --save_ranking_to run.nq.train.txt

# encode train query
# CUDA_VISIBLE_DEVICES=0 nohup python -m tevatron.driver.encode \
#   --output_dir=data/query_embs \
#   --model_name_or_path model_nq \
#   --fp16 \
#   --per_device_eval_batch_size 156 \
#   --dataset_name Tevatron/wikipedia-nq/train \
#   --encoded_save_path nq_train.pkl \
#   --encode_is_qry > nq_train.log 2>&1 &