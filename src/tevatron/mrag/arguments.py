from dataclasses import dataclass, field
from typing import Optional
from tevatron.arguments import ModelArguments, DataArguments
from transformers import TrainingArguments

@dataclass
class MDataArguments(DataArguments):
    train_dataset:  str = field(
        default="/data/tanhexiang/tevatron/data_nq/result100/fid.nq.train.jsonl", 
        metadata={"help":"default datasetname/local path"}
    )
    val_dataset:  str = field(
        default="/data/tanhexiang/tevatron/tevatron/data_nq/result100/fid.nq.dev.jsonl", 
        metadata={"help":"default datasetname/local path"}
    )
    corpus:  str = field(
        default=None, metadata={"help": "corpus name. used to find 'text' with id"}
    )
    n_context:  int = field(
        default=100, metadata={"help": "ctxs num"}
    )
    val_n_context: int = field(default=10)
    negative_ratio: float = field(default=0)

    # tokenizer
    fid_question_prefix:  str = field(
        default='question:', metadata={"help": ""}
    )
    fid_title_prefix:  str = field(
        default='title:', metadata={"help": ""}
    )
    fid_passage_prefix:  str = field(
        default='context:', metadata={"help": ""}
    )
    fid_passage_len:  int = field(
        default=200, metadata={"help":"max len"}
    )
    fid_target_len:  int = field(
        default=20, metadata={"help": "max len"}
    )
    dpr_query_len: int = field(
        default=32,
        metadata={
            "help": "DPR The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    dpr_passage_len: int = field(
        default=128,
        metadata={
            "help": "DPR The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    batch_negative:  bool = field(
        default=False, metadata={"help": "whether batch negative"}
    )
@dataclass
class MModelArguments(ModelArguments):
    placeholder:  bool = field(
        default=True, metadata={"help": "learnable mask vector"}
    )    

@dataclass
class MTrainArguments():
    output_dir: str = field(
        default="/data/tanhexiang/tevatron/tevatron/mrag_output", metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    sample_num :int = field(default=1)
    
    t5_base_tokenizer_path: str=field(default='t5-base')
    # 模型参数
    eps: float = field(
        default=1.0
    )
    alpha:float = field(default=15.0,metadata={"help":"初始化的拉格朗日乘子"})
    alpha_up_constrain:float = field(default=2000.0)
    bias:float = field(default=5.0)
    temperature: float=field(default=4.0)
    # 学习率
    learning_rate: float =field(default=1e-5)
    learning_rate_alpha: float = field(default=1e-4)
    train_batch_size: int = field(default=1)
    warmup_step: int=field(default=6000)
    validate_k: int=field(default=10)
    # 训练相关参数
    max_epochs :int=field(default=2)
    log_every_n_steps :int=field(default=50)
    val_check_interval :int=field(default=500,metadata={"help":"多少步validate一次"})
    limit_val_batches :float=field(default=200,metadata={"help":"验证跑多少步，最多800多"})
    precision :str=field(default='32',metadata={"help":"  '32','16','bf16'  "})
    # 其他参数
    fid_path: str = field(
        default="nq_reader_base"
    )
    seed : int =field(default=1)
    dataloader_num_workers:int = field(default=32)
    only_validate: bool=field(default=False)

    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})

