import random
from dataclasses import dataclass
from typing import List, Tuple
import torch

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from tevatron.arguments import DataArguments

import logging
logger = logging.getLogger(__name__)


class KLPreProcessor:
    def __init__(self, tokenizer, key_name,corpus=None, depth=100, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator
        self.corpus = corpus
        self.key_name = key_name
        self.corpus = load_dataset(corpus)['train']

    def __call__(self, example):
        question = self.tokenizer.encode(example['question'],
                                      add_special_tokens=False,
                                      max_length=self.query_max_length,
                                      truncation=True)
        passages = []
        scores = []
        for ctxs in example['ctxs']:
            scores.append(ctxs[self.key_name])

            if 'text' not in ctxs:
                ctxs['text'] = self.corpus[int(ctxs['id'])-1]['text']
                ctxs['title'] = self.corpus[int(ctxs['id'])-1]['title']
            text = ctxs['title'] + self.separator + ctxs['text']
            passages.append(self.tokenizer.encode(text,
                                                   add_special_tokens=False,
                                                   max_length=self.text_max_length,
                                                   truncation=True))
            
        return {
            'question': question,
            'passages': passages,
            'scores': scores
            }


class HFKLTrainDataset:
    def __init__(self, tokenizer: PreTrainedTokenizer,
                       data_args: DataArguments):
        self.dataset = load_dataset('json',data_files=data_args.dataset_name)['train']
        # print(self.dataset)
        self.preprocessor = KLPreProcessor
        self.tokenizer = tokenizer
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = ' '
        self.corpus = data_args.corpus
        self.depth =data_args.depth
        self.key_name = data_args.key_name

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor(self.tokenizer, self.key_name, self.corpus, self.depth, self.q_max_len, self.p_max_len, self.separator),
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
        return self.dataset


class KLTrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer
    ):
        self.train_data = dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_example(self, text_encoding: List[int], is_query=False):
        item = self.tokenizer.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding="max_length",
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return item.input_ids


    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], List[BatchEncoding]]:
        group = self.train_data[item]
        question = group['question']
        passages = group['passages']
        scores = group['scores']
        
        encoded_question = self.create_example(question, is_query=True) # dim
        encoded_passages = []
        for passage in passages:
            encoded_passages.append(self.create_example(passage))
        encoded_passages = torch.stack(encoded_passages,dim=0)  # n_passages, dim
        return {
            'question': encoded_question,   # tensor: dim
            'passages': encoded_passages,   # tensor: n_passages, dim
            'scores': torch.tensor(scores)  # tensor: n_passages
        }


@dataclass
class KLTrainCollator:
    def __init__(self, in_batch_negative:bool) -> None:
        self.in_batch_negative = in_batch_negative
    def __call__(self, batch):
        q = [x['question'] for x in batch]
        q = torch.stack(q,dim=0)    # b,len
        p = [x['passages'] for x in batch]
        p = torch.stack(p,dim=0)    # b,n,len
        s = [x['scores'] for x in batch]
        s = torch.stack(s,dim=0)    # b,n
        logging.debug("batch shape: q {}, p {}, s {}".format(q.shape,p.shape,s.shape))
        return q,p,s
