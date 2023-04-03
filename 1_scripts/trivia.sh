# 编码corpus
# ENCODE_DIR=embs
# mkdir $ENCODE_DIR
# for s in $(seq -f "%02g" 0 19)
# do
# CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
#   --output_dir=embs \
#   --model_name_or_path model_trivia \
#   --fp16 \
#   --per_device_eval_batch_size 156 \
#   --dataset_name Tevatron/wikipedia-trivia-corpus \
#   --encoded_save_path data_trivia/embs/corpus_emb.$s.pkl \
#   --encode_num_shard 20 \
#   --encode_shard_index $s
# done

# 编码query
# for split in test dev train
# do
#     CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
#     --output_dir=data_trivia/query_embs \
#     --model_name_or_path model_trivia \
#     --fp16 \
#     --per_device_eval_batch_size 312 \
#     --dataset_name Tevatron/wikipedia-trivia/${split} \
#     --encoded_save_path data_trivia/query_embs/trivia_${split}.pkl \
#     --encode_is_qry
# done
# 分块检索
# 检索
for split in test dev train
do
    # sharded search
    # INTERMEDIATE_DIR=intermediate_trivia_${split}
    # mkdir ${INTERMEDIATE_DIR}
    # for s in $(seq -f "%02g" 0 19)
    # do
    #     CUDA_VISIBLE_DEVICES=1 python -m tevatron.faiss_retriever \
    #     --query_reps data_trivia/query_embs/trivia_${split}.pkl \
    #     --passage_reps data_trivia/embs/corpus_emb.${s}.pkl \
    #     --depth 100 \
    #     --save_ranking_to ${INTERMEDIATE_DIR}/${s}
    # done

    # python -m tevatron.faiss_retriever.reducer \
    # --score_dir ${INTERMEDIATE_DIR} \
    # --query data_trivia/query_embs/trivia_${split}.pkl \
    # --save_ranking_to data_trivia/result100/run.trivia.${split}.txt

    # 转换格式 
    python -m tevatron.utils.format.convert_result_to_fid \
    --input data_trivia/result100/run.trivia.${split}.txt \
    --output data_trivia/result100/run.trivia.${split}.jsonl \
    --dataset_name Tevatron/wikipedia-trivia/${split} \
    --depth 100
    # --save_ctxs_text \
    # --corpus_name Tevatron/wikipedia-nq-corpus
done


# 测试
# python -m tevatron.utils.format.convert_result_to_trec \
#               --input run.trivia.test.txt \
#               --output run.trivia.test.trec

# python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
#               --topics dpr-trivia-test \
#               --index wikipedia-dpr \
#               --input run.trivia.test.trec \
#               --output run.trivia.test.json

# nohup python -m pyserini.eval.evaluate_dpr_retrieval \
#                 --retrieval run.trivia.test.json \
#                 --topk 20 100 &




