from dataclasses import dataclass, field
from typing import Optional
from tevatron.arguments import ModelArguments, TevatronTrainingArguments, DataArguments


@dataclass
class KLDataArguments(DataArguments):
    corpus: Optional[str] = field(
        default=None, metadata={"help": "corpus name. used to find 'text' with id"}
    )
    key_name: Optional[str] = field(
        default='score', metadata={"help":"key name of ctxs"}
    )
    depth: Optional[int] = field(
        default=100, metadata={"help": "ctxs num"}
    )
    batch_negative: Optional[bool] = field(
        default=False, metadata={"help":"whether batch negative"}
    )
