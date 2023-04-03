from dataclasses import dataclass, field
from typing import Optional
from tevatron.arguments import ModelArguments, TevatronTrainingArguments, DataArguments


@dataclass
class MDataArguments(DataArguments):
    corpus: Optional[str] = field(
        default=None, metadata={"help": "corpus name. used to find 'text' with id"}
    )
    n_context: Optional[int] = field(
        default=100, metadata={"help": "ctxs num"}
    )
    # tokenizer
    fid_question_prefix: Optional[str] = field(
        default='question:', metadata={"help": ""}
    )
    fid_title_prefix: Optional[str] = field(
        default='title:', metadata={"help": ""}
    )
    fid_passage_prefix: Optional[str] = field(
        default='context:', metadata={"help": ""}
    )
    fid_passage_len: Optional[int] = field(
        default=200, metadata={"help":"max len"}
    )
    fid_target_len: Optional[int] = field(
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
    batch_negative: Optional[bool] = field(
        default=False, metadata={"help": "whether batch negative"}
    )
@dataclass
class MModelArguments(ModelArguments):
    placeholder: Optional[bool] = field(
        default=True, metadata={"help": "learnable mask vector"}
    )

@dataclass
class MArguments:
    alpha: Optional[float] = field(
        defualt=0.1, metadata={"help": "super params for balancing two loss"}
    )
