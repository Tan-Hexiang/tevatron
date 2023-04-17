import random
from dataclasses import dataclass
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from transformers import  PreTrainedTokenizer, BatchEncoding

from tevatron.mrag.arguments import MDataArguments
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)


class MPreProcessor:
    def __init__(self, dpr_tokenizer, fid_tokenizer, data_args: MDataArguments):
        self.data_args = data_args
        self.dpr_tokenizer = dpr_tokenizer
        self.fid_tokenizer = fid_tokenizer
        self.corpus = load_dataset(data_args.corpus)['train']

    def __call__(self, example):

        dpr_question = self.dpr_tokenizer.encode(example['question'],
                                                 add_special_tokens=False,
                                                 pad_to_max_length=True,
                                                 max_length=self.data_args.dpr_query_len,
                                                 truncation=True)
        # random sample an answer from answers list to train
        answer = random.choice(example['answers']) + ' </s>'
        fid_answer = self.fid_tokenizer.encode_plus(answer,
                                        max_length=self.data_args.fid_target_len if self.data_args.fid_target_len > 0 else None,
                                        pad_to_max_length=True,
                                        return_tensors='pt',
                                        truncation=True if self.data_args.fid_target_len > 0 else False,
                                        )
        
        fid_answer_ids = fid_answer['input_ids']
        fid_answer_mask = fid_answer['attention_mask'].bool()
        fid_answer_ids = fid_answer_ids.masked_fill(~fid_answer_mask, -100)
        # logging.info("fid_answer_ids type {}".format(type(fid_answer_ids)))

        dpr_passages = []
        fid_passages = []
        for ctxs in example['ctxs'][:self.data_args.n_context]:
            # 匹配正文
            if 'text' not in ctxs:
                ctxs['text'] = self.corpus[int(ctxs['id']) - 1]['text']
                ctxs['title'] = self.corpus[int(ctxs['id']) - 1]['title']
            # dpr
            text = ctxs['title'] + ' ' + ctxs['text']
            dpr_passages.append(self.dpr_tokenizer.encode(text,
                                                      pad_to_max_length=True,
                                                      add_special_tokens=False,
                                                      max_length=self.data_args.dpr_passage_len,
                                                      truncation=True))
            # fid
            f = self.data_args.fid_title_prefix + " {} " + self.data_args.fid_passage_prefix + " {}"
            fid_question = self.data_args.fid_question_prefix + ' ' + example['question']
            fid_passage = f.format(ctxs['title'], ctxs['text'])
            fid_passage = fid_question + ' ' + fid_passage
            fid_passages.append(fid_passage)
        # fid tokenizer
        # logging.info("fid passages details: {}".format(fid_passages))
        fid_passages = self.fid_tokenizer.batch_encode_plus(
            fid_passages,
            max_length=self.data_args.fid_passage_len,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        # logging.info("fid passages tokenized details: {}".format(fid_passages))
        
        fid_passage_ids = fid_passages['input_ids'][None]
        fid_passage_mask = fid_passages['attention_mask'][None]
        # logging.info("fid passages ids details: {}".format(fid_passage_ids))
        # logging.info("fid passages mask details: {}".format(fid_passage_mask))
        # logging.info("fid_passage_ids type {}".format(type(fid_passage_ids)))
        # logging.info("fid_passage_mask type {}".format(type(fid_passage_mask)))

        return {
            'dpr_question': dpr_question,
            'dpr_passages': dpr_passages,
            'fid_answer_ids': fid_answer_ids,
            # 'fid_answer_mask': fid_answer_mask,
            'fid_passage_ids': fid_passage_ids,
            'fid_passage_mask': fid_passage_mask
        }


class HFMTrainDataset:
    def __init__(self, dpr_tokenizer: PreTrainedTokenizer,
                 fid_tokenizer: PreTrainedTokenizer,
                 data_path,
                 data_args: MDataArguments):
        self.data_args = data_args
        self.dataset = load_dataset('json', data_files=data_path)['train']
        logging.info("Datasets 前10个数据   {}".format(self.dataset[:10]))
        self.dpr_tokenizer = dpr_tokenizer
        self.fid_tokenizer = fid_tokenizer
        self.proc_num = data_args.dataset_proc_num
        self.preprocessor = MPreProcessor(self.dpr_tokenizer, self.fid_tokenizer, data_args=self.data_args)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            # map does not return tensor type https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
            self.dataset = self.dataset.map(
                self.preprocessor,
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running dpr/fid tokenizer on train dataset",
                load_from_cache_file=True,
            )
        return self.dataset


class MTrainDataset(Dataset):
    def __init__(
            self,
            data_args: MDataArguments,
            dataset,
            dpr_tokenizer: PreTrainedTokenizer
    ):
        self.train_data = dataset
        self.dpr_tokenizer = dpr_tokenizer
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_example(self, text_encoding: List[int], is_query=False):
        item = self.dpr_tokenizer.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.dpr_query_len if is_query else self.data_args.dpr_passage_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        return item.input_ids

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        group = self.train_data[item]
        question = group['dpr_question']
        passages = group['dpr_passages']

        encoded_question = self.create_example(question, is_query=True)
        encoded_passages = []
        for passage in passages:
            encoded_passages.append(self.create_example(passage))
        encoded_passages = torch.stack(encoded_passages, dim=0)  # n_passages, dim
        return {
            'dpr_question': encoded_question,  # dim
            'dpr_passages': encoded_passages,  # n_passages, dim
            'fid_answer_ids': torch.Tensor(group['fid_answer_ids'][0]).type(torch.LongTensor),
            # 'fid_answer_mask': fid_answer_mask,
            'fid_passage_ids': torch.Tensor(group['fid_passage_ids'][0]).type(torch.LongTensor),
            'fid_passage_mask': torch.Tensor(group['fid_passage_mask'][0]).type(torch.FloatTensor)
        }


@dataclass
class MTrainCollator:
    def __init__(self, in_batch_negative: bool) -> None:
        self.in_batch_negative = in_batch_negative

    def __call__(self, batch):
        q = [x['dpr_question'] for x in batch]
        q = torch.stack(q, dim=0)  # b*len
        p = [x['dpr_passages'] for x in batch]
        p = torch.stack(p, dim=0)  # b*n*len
        fid_a = [x['fid_answer_ids'] for x in batch]
        fid_a = torch.stack(fid_a, dim=0)
        fid_p_ids = [x['fid_passage_ids'] for x in batch]
        fid_p_ids = torch.stack(fid_p_ids, dim=0)
        fid_p_mask = [x['fid_passage_mask'] for x in batch]
        fid_p_mask = torch.stack(fid_p_mask, dim=0)

        return {
            'dpr_question': q,  # bsz, len
            'dpr_passages': p,  # bsz, n_context, len
            'fid_answer_ids': fid_a,  # bsz, len
            # 'fid_answer_mask': fid_answer_mask,
            'fid_passage_ids': fid_p_ids,  # bsz, n_context, len
            'fid_passage_mask': fid_p_mask
        }
