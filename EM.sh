model=""
model_path=$model/eval
CUDA_VISIBLE_DEVICES=1 python src/tevatron/mrag/test_reader.py \
        --model_path nq_reader_base \
        --eval_data  \
        --per_gpu_batch_size 2 \
        --n_context 100 \
        --name eval \
        --checkpoint_dir $model \
        --sort_by_score