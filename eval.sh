
# for s in $(seq -f "%02g" 0 19)
# do
# CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
#   --output_dir=49/eval/embs \
#   --model_name_or_path 49 \
#   --fp16 \
#   --per_device_eval_batch_size 16 \
#   --dataset_name Tevatron/wikipedia-nq-corpus \
#   --encoded_save_path corpus_emb.$s.pkl \
#   --encode_num_shard 20 \
#   --encode_shard_index $s
# done


# for s in $(seq -f "%02g" 0 9)
# do
# CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
#   --output_dir=embs \
#   --model_name_or_path 49 \
#   --fp16 \
#   --per_device_eval_batch_size 2048 \
#   --dataset_name Tevatron/wikipedia-nq-corpus \
#   --encoded_save_path corpus_emb.$s.pkl \
#   --encode_num_shard 20 \
#   --encode_shard_index $s
# done

# CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
#   --output_dir=49/eval/query_embs \
#   --model_name_or_path 49 \
#   --fp16 \
#   --per_device_eval_batch_size 2048 \
#   --dataset_name Tevatron/wikipedia-nq/test \
#   --encoded_save_path 49/eval/query_embs/nq_test.pkl \
#   --encode_is_qry

INTERMEDIATE_DIR=49/eval/intermediate
# mkdir ${INTERMEDIATE_DIR}
# for s in $(seq -f "%02g" 0 19)
# do
#   CUDA_VISIBLE_DEVICES=0 python -m tevatron.faiss_retriever \
#   --query_reps 49/eval/query_embs/nq_test.pkl \
#   --passage_reps 49/eval/embs/corpus_emb.${s}.pkl \
#   --depth 100 \
#   --save_ranking_to ${INTERMEDIATE_DIR}/${s}
# done

# python -m tevatron.faiss_retriever.reducer \
# --score_dir ${INTERMEDIATE_DIR} \
# --query 49/eval/query_embs/nq_test.pkl \
# --save_ranking_to 49/eval/run.nq.test.txt

# python -m tevatron.utils.format.convert_result_to_trec \
#               --input 49/eval/run.nq.test.txt \
#               --output 49/eval/run.nq.test.trec

# python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
#               --topics dpr-nq-test \
#               --index wikipedia-dpr \
#               --input 49/eval/run.nq.test.trec \
#               --output 49/eval/run.nq.test.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval 49/eval/run.nq.test.json \
                --topk 1 5 20 100
