from dataclasses import dataclass, field
from typing import Optional
from tevatron.arguments import ModelArguments, TevatronTrainingArguments, DataArguments


@dataclass
class KLDataArguments(DataArguments):
    corpus: str = field(
        default=None, metadata={"help": "corpus name. used to find 'text' with id"}
    )
    key_name: str = field(
        default='score', metadata={"help":"key name of ctxs"}
    )
    depth: int = field(
        default=100, metadata={"help": "ctxs num"}
    )
    batch_negative: bool = field(
        default=False, metadata={"help":"whether batch negative"}
    )
