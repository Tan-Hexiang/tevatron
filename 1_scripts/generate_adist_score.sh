conda activate fid

CUDA_VISIBLE_DEVICES=0 nohup python /data/tanhexaing/CFQA/test_reader.py \
        --model_path ../pretrained_models/nq_reader_base \
        --eval_data  \
        --per_gpu_batch_size 1 \
        --n_context 32 \
        --name ADist_32_train \
        --checkpoint_dir ../实验结果/1102score_32passages \
        --write_crossattention_scores \
        --write_results > /data/tanhexiang/FiD/实验结果/1102score_32passages/log/ADist_score_32_train_run.log 2>&1 &
