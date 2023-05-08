# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import transformers
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import tqdm

import tevatron.fid_src.slurm
import tevatron.fid_src.util
from tevatron.fid_src.options import Options
import tevatron.fid_src.data
import tevatron.fid_src.evaluation
import tevatron.fid_src.model

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
    with torch.no_grad():
        with tqdm.tqdm(dataloader) as pbar:
            for i, batch in enumerate(pbar):
                (idx, _, _, context_ids, context_mask) = batch # 中间两个为target_ids, target_mask, 预测时需要忽略
                if opt.write_crossattention_scores:
                    model.reset_score_storage()

                outputs = model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=50,
                )

                if opt.write_crossattention_scores:
                    crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

                for k, o in enumerate(outputs):
                    ans = tokenizer.decode(o, skip_special_tokens=True)
                    example = dataset.data[idx[k]]
                    if 'answers' in example:
                        score = tevatron.fid_src.evaluation.ems(ans, example['answers'])
                        exactmatch.append(score)

                    if opt.write_results:
                        fw.write(str(example['id']) + "\t" + ans + '\n')
                    if opt.write_crossattention_scores:
                        for j in range(context_ids.size(1)):
                            example['ctxs'][j]['score'] = crossattention_scores[k, j].item()

                    total += 1
                if (i + 1) % opt.eval_print_freq == 0:
                    log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                    if len(exactmatch) == 0:
                        log += '| no answer to compute scores'
                    else:
                        log += f' | average = {np.mean(exactmatch):.3f}'
                    logger.warning(log)

    logger.warning(f'Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = tevatron.fid_src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    options.add_experiment_options()
    opt = options.parse()
    tevatron.fid_src.slurm.init_distributed_mode(opt)
    tevatron.fid_src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = tevatron.fid_src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)


    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base', return_dict=False)

    collator_function = tevatron.fid_src.data.Collator(opt.text_maxlength, tokenizer)
    eval_examples = tevatron.fid_src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
        score_key = opt.score_key
    )
    eval_dataset = tevatron.fid_src.data.Dataset(
        eval_examples, 
        opt.n_context, 
        sort_by_score=opt.sort_by_score,
        passages_source_path=opt.passages_source_path
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=1, 
        collate_fn=collator_function
    )
    
    model_class = tevatron.fid_src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'EM {100*exactmatch:.2f}, Total number of example {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        tevatron.fid_src.util.write_output(glob_path, write_path) 
    if opt.write_crossattention_scores:
        tevatron.fid_src.util.save_distributed_dataset(eval_dataset.data, opt)

