model=model_nq_dpr_new
model_path=model_nq_dpr_new/eval_32
# mkdir $model_path
# mkdir $model_path/embs
# mkdir $model_path/query_embs
# # Encode corpus
# for s in $(seq -f "%02g" 15 19)
# do
# CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
#   --output_dir $model_path/embs \
#   --model_name_or_path $model \
#   --per_device_eval_batch_size 2048 \
#   --dataset_name Tevatron/wikipedia-nq-corpus \
#   --encoded_save_path $model_path/embs/corpus_emb.$s.pkl \
#   --encode_num_shard 20 \
#   --encode_shard_index $s \
# #   --fp16 \
# done

# # Encode query
CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
  --output_dir $model_path/query_embs \
  --model_name_or_path $model \
  --per_device_eval_batch_size 2048 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path $model_path/query_embs/nq_test.pkl \
  --encode_is_qry \
#   --fp16 \

# # 分块检索
INTERMEDIATE_DIR=${model_path}/intermediate
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
  CUDA_VISIBLE_DEVICES=1 python -m tevatron.faiss_retriever \
  --query_reps ${model_path}/query_embs/nq_test.pkl \
  --passage_reps ${model_path}/embs/corpus_emb.${s}.pkl \
  --depth 100 \
  --save_ranking_to ${INTERMEDIATE_DIR}/${s}
done
# # 合并
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query ${model_path}/query_embs/nq_test.pkl \
--save_ranking_to ${model_path}/run.nq.test.txt

# # 转换数据格式
python -m tevatron.utils.format.convert_result_to_trec \
              --input ${model_path}/run.nq.test.txt \
              --output ${model_path}/run.nq.test.trec

python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input ${model_path}/run.nq.test.trec \
              --output ${model_path}/run.nq.test.json

# # 得到检索结果
python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval ${model_path}/run.nq.test.json \
                --topk 1 5 20 100


# 转化为fid需要的格式,注意！这里用的test集合！！！！
python -m tevatron.utils.format.convert_result_to_fid_fast \
--input ${model_path}/run.nq.test.txt \
--output ${model_path}/fid.nq.test.jsonl \
--dataset_name Tevatron/wikipedia-nq/test \
--depth 100 \
--save_ctxs_text \
--corpus_name Tevatron/wikipedia-nq-corpus

# 得到EM结果
CUDA_VISIBLE_DEVICES=1 python -m tevatron.mrag.test_reader \
        --model_path nq_reader_base \
        --eval_data ${model_path}/fid.nq.test.jsonl \
        --per_gpu_batch_size 6 \
        --n_context 100 \
        --name test_reader \
        --checkpoint_dir ${model_path}
