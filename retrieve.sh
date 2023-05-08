model=model_nq_dpr_new
model_path=model_nq_dpr_new/eval
# # mkdir $model_path
# # mkdir $model_path/embs
# # mkdir $model_path/query_embs


# # Encode dev query
CUDA_VISIBLE_DEVICES=3 python -m tevatron.driver.encode \
  --output_dir $model_path/query_embs \
  --model_name_or_path $model \
  --fp16 \
  --per_device_eval_batch_size 2048 \
  --dataset_name Tevatron/wikipedia-nq/dev \
  --encoded_save_path $model_path/query_embs/nq_dev.pkl \
  --encode_is_qry

# # Encode train query
CUDA_VISIBLE_DEVICES=3 python -m tevatron.driver.encode \
  --output_dir $model_path/query_embs \
  --model_name_or_path $model \
  --fp16 \
  --per_device_eval_batch_size 2048 \
  --dataset_name Tevatron/wikipedia-nq/train \
  --encoded_save_path $model_path/query_embs/nq_train.pkl \
  --encode_is_qry

# # dev
# # # 分块检索
INTERMEDIATE_DIR=${model_path}/intermediate_dev
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
  CUDA_VISIBLE_DEVICES=3 python -m tevatron.faiss_retriever \
  --query_reps ${model_path}/query_embs/nq_dev.pkl \
  --passage_reps ${model_path}/embs/corpus_emb.${s}.pkl \
  --depth 100 \
  --save_ranking_to ${INTERMEDIATE_DIR}/${s}
done
# # # 合并
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query ${model_path}/query_embs/nq_dev.pkl \
--save_ranking_to ${model_path}/run.nq.dev.txt

# # train
# # # 分块检索
INTERMEDIATE_DIR=${model_path}/intermediate_train
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
  CUDA_VISIBLE_DEVICES=3 python -m tevatron.faiss_retriever \
  --query_reps ${model_path}/query_embs/nq_train.pkl \
  --passage_reps ${model_path}/embs/corpus_emb.${s}.pkl \
  --depth 100 \
  --save_ranking_to ${INTERMEDIATE_DIR}/${s}
done
# # # 合并
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query ${model_path}/query_embs/nq_train.pkl \
--save_ranking_to ${model_path}/run.nq.train.txt


# # 转化为fid需要的格式
# python -m tevatron.utils.format.convert_result_to_fid \
# --input ${model_path}/run.nq.dev.txt \
# --output ${model_path}/fid.nq.dev.jsonl \
# --dataset_name Tevatron/wikipedia-nq/dev \
# --depth 100 \
# --save_ctxs_text \
# --corpus_name Tevatron/wikipedia-nq-corpus

# python -m tevatron.utils.format.convert_result_to_fid \
# --input ${model_path}/run.nq.train.txt \
# --output ${model_path}/fid.nq.train.jsonl \
# --dataset_name Tevatron/wikipedia-nq/train \
# --depth 100 \
# --save_ctxs_text \
# --corpus_name Tevatron/wikipedia-nq-corpus

# fast convert
python -m tevatron.utils.format.convert_result_to_fid_fast \
--input ${model_path}/run.nq.train.txt \
--output ${model_path}/fid.nq.train.fast.jsonl \
--dataset_name Tevatron/wikipedia-nq/train \
--depth 100 \
--save_ctxs_text \
--corpus_name Tevatron/wikipedia-nq-corpus

python -m tevatron.utils.format.convert_result_to_fid_fast \
--input ${model_path}/run.nq.dev.txt \
--output ${model_path}/fid.nq.dev.fast.jsonl \
--dataset_name Tevatron/wikipedia-nq/dev \
--depth 100 \
--save_ctxs_text \
--corpus_name Tevatron/wikipedia-nq-corpus

# # 得到EM结果
CUDA_VISIBLE_DEVICES=3 python -m tevatron.mrag.test_reader \
        --model_path nq_reader_base \
        --eval_data ${model_path}/fid.nq.dev.fast.jsonl \
        --per_gpu_batch_size 6 \
        --n_context 100 \
        --name test_reader \
        --checkpoint_dir ${model_path}
