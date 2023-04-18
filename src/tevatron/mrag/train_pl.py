import logging
import os
import sys
from lightning.pytorch import loggers as pl_loggers
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    T5Tokenizer,
)
from transformers import (
    get_constant_schedule_with_warmup,
)
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.MRetriever import MDenseModel,mrag
from tevatron.mrag.data import HFMTrainDataset, MTrainCollator, MTrainDataset
from tevatron.mrag.Mtrainer import MTrainer
from tevatron.mrag.arguments import MModelArguments, MTrainArguments, MDataArguments
from tevatron.mrag.lookahead import LookaheadRMSprop
from tevatron.mrag.mrag_pl import MaskRetrievalAugmentGeneration

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((MModelArguments, MDataArguments, MTrainArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, hparams = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, hparams = parser.parse_args_into_dataclasses()
        model_args: MModelArguments
        data_args: MDataArguments
        hparams: MTrainArguments
    
    set_seed(hparams.seed)

    # prepare model
    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    mdense = MDenseModel.build(
        model_args,
        hparams,
        config=config,
    )
    fid = FiDT5.from_pretrained(hparams.fid_path)
    m = mrag(fid, mdense, n_context=data_args.n_context, eps=hparams.eps)
    # tokenizer
    proxies = {'http': 'http://10.130.3.188:1087'}
    print("Using proxy {}".format(proxies))
    fid_tokenizer = T5Tokenizer.from_pretrained('t5-base', proxies=proxies)
    dpr_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, proxies=proxies)
    # prepare dataset
    train_dataset = HFMTrainDataset(data_path=data_args.train_dataset,
                                    dpr_tokenizer=dpr_tokenizer,
                                    fid_tokenizer=fid_tokenizer,
                                    data_args=data_args,
                                    )
    val_dataset = HFMTrainDataset(data_path=data_args.val_dataset,
                                    dpr_tokenizer=dpr_tokenizer,
                                    fid_tokenizer=fid_tokenizer,
                                    data_args=data_args,
                                    )
    if hparams.local_rank > 0:
        print("Waiting for main process to perform the mapping")
        torch.distributed.barrier()
    train_dataset = MTrainDataset(data_args, train_dataset.process(), dpr_tokenizer)
    val_dataset = MTrainDataset(data_args, val_dataset.process(), dpr_tokenizer)
    
    # get model
    model = MaskRetrievalAugmentGeneration(m, 
                                           train_dataset=train_dataset, 
                                           hparams=hparams, 
                                           val_dataset=val_dataset)
    # log
    if not os.path.exists(hparams.output_dir):
        os.makedirs(hparams.output_dir, exist_ok=True)
    if not os.path.exists(hparams.output_dir+"/logs/"):
        os.makedirs(hparams.output_dir+"/logs/", exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.output_dir+"/logs/")
    trainer = pl.Trainer(
    accelerator="auto", strategy="auto", devices="auto",
    logger=tb_logger,
    default_root_dir=hparams.output_dir,
    precision='32',
    max_epochs=2,
    log_every_n_steps=5,
    enable_checkpointing=True,
    )
    trainer.fit(model)  

    


if __name__ == "__main__":
    main()
