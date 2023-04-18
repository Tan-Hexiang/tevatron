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
    model_name_or_path: str =field(
        default="/data/tanhexiang/tevatron/model_nq"
    )

@dataclass
class MTrainArguments(TrainingArguments):
    eps: float = field(
        default=1.0
    )
    learning_rate: float =field(default=1e-5)
    learning_rate_alpha: float = field(default=1e-4)
    output_dir: str = field(
        default="/data/tanhexiang/tevatron/tevatron/mrag_output", metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    fid_path: str = field(
        default="nq_reader_base"
    )
    dataloader_num_workers:int = field(default=32)
    warmup_ratio: float = field(default=0.1)
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})

