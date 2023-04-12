import logging
import os
import sys

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    T5Tokenizer,
)
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.MRetriever import MDenseModel,mrag
from tevatron.mrag.data import HFMTrainDataset, MTrainCollator, MTrainDataset
from tevatron.mrag.Mtrainer import MTrainer
from tevatron.mrag.arguments import MModelArguments, MTrainArguments, MDataArguments

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((MModelArguments, MDataArguments, MTrainArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: MModelArguments
        data_args: MDataArguments
        training_args: MTrainArguments


    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    # Setup logging
    log_path = training_args.output_dir+"/run.log"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.FileHandler(log_path, mode='w'),
                              stream_handler]
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logging.info("loading fid and dpr tokenizer")
    fid_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    dpr_tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    logging.info("loading MdenseModel")
    model = MDenseModel.build(
        model_args,
        training_args,
        config=config,
    )

    logging.info("loading dataset")
    train_dataset = HFMTrainDataset(dpr_tokenizer=dpr_tokenizer,
                                    fid_tokenizer=fid_tokenizer,
                                    data_args=data_args,
                                    )
    if training_args.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = MTrainDataset(data_args, train_dataset.process(), dpr_tokenizer)
    if training_args.local_rank == 0:
        print("Loading results from main process")
        torch.distributed.barrier()
    logging.info("dataset load completed!")
    logging.info("Column_names of dataset:")
    logging.info("{}".format(train_dataset.train_data.column_names))

    # prepare fid 
    fid = FiDT5.from_pretrained(training_args.fid_path)
    # prepare mrag model
    m = mrag(fid, model, n_context=data_args.n_context, alpha=training_args.alpha)
    # debug model 
    logging.info("Model detail:")
    for name, p in m.named_parameters():
        if p.requires_grad:
            # print("requires_grad: {}".format(name))
            logging.info("requires_grad: {} {}".format(name, p.shape))
        else:
            logging.info("close grad of {}".format(name))
    # lookahead
    trainer = MTrainer(
        model=m,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=MTrainCollator(
            in_batch_negative=data_args.batch_negative
        )
    )
    train_dataset.trainer = trainer

    trainer.save_model()
    trainer.train()  # TODO: resume training
    trainer.save_model()
    if trainer.is_world_process_zero():
        dpr_tokenizer.save_pretrained(training_args.output_dir)
        # fid_tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
