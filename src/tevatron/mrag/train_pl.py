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
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import (
    get_constant_schedule_with_warmup,
)
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.MRetriever import MDenseModel,mrag
from tevatron.mrag.data_ import HFMTrainDataset, MTrainCollator, MTrainDataset
from tevatron.mrag.Mtrainer import MTrainer
from tevatron.mrag.arguments import MModelArguments, MTrainArguments, MDataArguments
from tevatron.mrag.lookahead import LookaheadRMSprop
from tevatron.mrag.mrag_pl import MaskRetrievalAugmentGeneration

logger = logging.getLogger(__name__)


def main():
    torch.set_float32_matmul_precision('medium')
    parser = HfArgumentParser((MModelArguments, MDataArguments, MTrainArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, hparams = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, hparams = parser.parse_args_into_dataclasses()
        model_args: MModelArguments
        data_args: MDataArguments
        hparams: MTrainArguments
    
    set_seed(hparams.seed)
    
    # mkdir
    if not os.path.exists(hparams.output_dir):
        os.makedirs(hparams.output_dir, exist_ok=True)
    if not os.path.exists(hparams.output_dir+"/logs/"):
        os.makedirs(hparams.output_dir+"/logs/", exist_ok=True)
    
    # Setup logging
    log_path = hparams.output_dir+"/run.log"
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        # if os.environ.get('LOCAL_RANK') in [-1, 0, 1] else logging.WARN
        level=logging.DEBUG,
        handlers=[logging.FileHandler(log_path, mode='w'),
                              stream_handler]
    )
    
    # get model
    model = MaskRetrievalAugmentGeneration(
                                           hparams=hparams, 
                                           data_args=data_args, 
                                           model_args=model_args)
    # compiled_model = torch.compile(model)
    # debug model
    for name, p in model.named_parameters():
        if p.requires_grad:
            # print("requires_grad: {}".format(name))
            logging.debug("requires_grad: {} {}".format(name, p.shape))
        else:
            logging.debug("close grad of {}".format(name))
    # compiled_model = torch.compile(model)
   
    # logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.output_dir)
    # , strategy='auto'
    trainer = pl.Trainer(
    accelerator="gpu", devices=1,
    logger=tb_logger,

    default_root_dir=hparams.output_dir,
    precision=hparams.precision,
    # precision="bf16",
    max_epochs=hparams.max_epochs,
    log_every_n_steps=hparams.log_every_n_steps,
    limit_val_batches=hparams.limit_val_batches,
    val_check_interval=hparams.val_check_interval,

    enable_checkpointing=True,
    # enable_progress_bar=False,
    # barebones=True,
    num_sanity_val_steps=2,
    profiler=False,
    )

    logging.info("ALL information saved in {}".format(hparams.output_dir))
    if hparams.only_validate:
        logging.info("Only validate!")
        trainer.validate(model)
    else:
        logging.info("Begin train!")
        trainer.fit(model)  

    


if __name__ == "__main__":
    main()
