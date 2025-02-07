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

# encode query
# CUDA_VISIBLE_DEVICES=1 python -m tevatron.driver.encode \
#   --output_dir=data/query_embs \
#   --model_name_or_path model_nq \
#   --fp16 \
#   --per_device_eval_batch_size 156 \
#   --dataset_name Tevatron/wikipedia-nq/dev \
#   --encoded_save_path nq_dev.pkl \
#   --encode_is_qry

# 检索
for split in dev train test
do
    # sharded search
    # INTERMEDIATE_DIR=intermediate_nq_${split}
    # mkdir ${INTERMEDIATE_DIR}
    # for s in $(seq -f "%02g" 0 19)
    # do
    #     CUDA_VISIBLE_DEVICES=1 python -m tevatron.faiss_retriever \
    #     --query_reps data_nq/query_embs/nq_${split}.pkl \
    #     --passage_reps data_nq/embs/corpus_emb.${s}.pkl \
    #     --depth 100 \
    #     --save_ranking_to ${INTERMEDIATE_DIR}/${s}
    # done

    # python -m tevatron.faiss_retriever.reducer \
    # --score_dir ${INTERMEDIATE_DIR} \
    # --query data_nq/query_embs/nq_${split}.pkl \
    # --save_ranking_to data_nq/result100/run.nq.${split}.txt

    python -m tevatron.utils.format.convert_result_to_fid \
    --input data_nq/result100/run.nq.${split}.txt \
    --output data_nq/result100/fid.nq.${split}.jsonl \
    --dataset_name Tevatron/wikipedia-nq/${split} \
    --depth 100
    # --save_ctxs_text \
    # --corpus_name Tevatron/wikipedia-nq-corpus
done

# encode train query

# for split in dev test train
# do
#     CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
#     --output_dir=data_nq/query_embs \
#     --model_name_or_path model_nq \
#     --fp16 \
#     --per_device_eval_batch_size 156 \
#     --dataset_name Tevatron/wikipedia-nq/${split} \
#     --encoded_save_path data_nq/query_embs/nq_${split}.pkl \
#     --encode_is_qry
# done
# python -m tevatron.utils.format.convert_result_to_trec \
#               --input run.nq.train.txt \
#               --output run.nq.train.trec

# python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
#               --topics dpr-nq-train \
#               --index wikipedia-dpr \
#               --input run.nq.train.trec \
#               --output run.nq.train.json

# nohup python -m pyserini.eval.evaluate_dpr_retrieval \
#                 --retrieval run.nq.train.json \
#                 --topk 20 100 &


# python -m tevatron.utils.format.convert_result_to_fid \
# --input data_nq/result/run.nq.dev.txt \
# --output data_nq/result/fid.nq.dev.json \
# --dataset_nam Tevatron/wikipedia-nq/dev \
# --depth 1000
# --save_ctxs_text \
# --corpus_name Tevatron/wikipedia-nq-corpus