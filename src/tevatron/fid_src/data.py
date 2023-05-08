# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# from memory_profiler import profile
from asyncio.log import logger
from glob import glob
import torch
import random
import json
import numpy as np
from tevatron.fid_src.jsonl import load_all_jsonl
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 do_softmax = False, 
                 sort_by_score = False,
                 passages_source_path = None):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        # 是否对score softmax
        self.softmax = torch.nn.Softmax(dim=-1)
        self.do_softmax = do_softmax
        # 不要进行排序，很多实验需要验证score的顺序影响
        # if sort_by_score:
            # self.sort_data()
        if passages_source_path is not None:
            self.passages_source = pd.read_csv(passages_source_path, sep='\t')
    
    def get_text_from_id(self, id:int):
        # dataframe编码是从0开始的，id从1开始，所以index = id-1；可以输出例子确认一下
        return self.passages_source.at[id-1,'text'],self.passages_source.at[id-1,'title']

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            # python3以后删除了haskey方法
            # if not example['ctxs'][0].has_key('text'):
            if not 'text' in example['ctxs'][0]:
                for i in range(self.n_context):
                    # id字段默认是string类型，需要转换为int
                    # dpr_result id是以wiki:开头的
                    id = example['ctxs'][i]['id']
                    if id[0:5] == 'wiki:':
                        id = id.replace('wiki:','')
                    id = int(id)
                    example['ctxs'][i]['text'],example['ctxs'][i]['title'] = self.get_text_from_id(id)

            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            if self.do_softmax:
                scores = self.softmax(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
            # debug
            # print("***----------------------------***")
            # print("question : {}".format(question))
            # print("passages {}".format(len(passages)))
            # print("score {} {}".format(scores.shape,scores))
            # for i in range(len(passages)): 
            #     print("passages {} : {}".format(i,passages[i]))
            #     print("example['ctxs'][i]['ADist_score']: {}",example['ctxs'][i]['ADist_score'])
            #     print("example['ctxs'][i]['score']: {}",example['ctxs'][i]['score'])
            #     print("**")
            # exit()
        else:
            passages, scores = None, None
        
        # print("***----------------------------***")
        # print("question : {}".format(question))
        # print("passages : {}".format(passages[0]))
        # exit()

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    # def sort_data(self):
    #     if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
    #         return
    #     for ex in self.data:
    #         ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)
        ## passage包括question and passages
        return (index, target_ids, target_mask, passage_ids, passage_masks)

def sort_example(ex):
    if not 'score' in ex['ctxs'][0]:
        return
    ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

# @profile(precision=4,stream=open('/data/tanhexiang/CF_QA/nohup/memory_profiler_load_data.log','w+'))
def load_data(data_path=None, score_key = 'score', global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        # data = open(data_path, 'r')
        data = load_all_jsonl(data_path)
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    logger.warning("*---------------------------------------*")
    logger.warning("Using score_keys: {}".format(score_key))
    logger.warning("*---------------------------------------*")
    for k, example in enumerate(data):  #enumerate可以加file object/json将其转换为k,v的迭代器
        if global_rank > -1 and not k%world_size==global_rank:
            continue
        # if data_path is not None and data_path.endswith('.jsonl'):
            # example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        # 将使用的对应的score放进'score'中，后面dataset和collator都适用score作为关键词
        if 'ctxs' in example:
            for c in example['ctxs']:
                if str(score_key) in c:
                    c['score'] = c[str(score_key)]
                else:
                    c['score'] = 0
        else:
            example['ctxs'] = []
        examples.append(example)
    ## egrave: is this needed?
    # if data_path is not None and data_path.endswith('.jsonl'):
        # data.close()
    logger.info("successfully load data, len {}".format(len(examples)))
    logger.info("world_size:{}, global_rank:{}".format(world_size,global_rank))
    return examples

class RetrieverCollator(object):
    def __init__(self, tokenizer , passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask

class RerankCollator(object):
    def __init__(self, tokenizer, maxlength=200) -> None:
        self.tokenizer = tokenizer
        self.maxlength = maxlength
    def __call__(self, batch):
        # scores: batch*n_context  
        scores = [k['scores'] for k in batch]
        scores = torch.stack(scores,dim=0)
        bsz = scores.shape[0]
        n_context = scores.shape[1]
        index = []
        question = []
        passages = []
        for example in batch:
            for k,p in enumerate(example['passages']):
                # 一个question/index对应n_context个scores/passages
                question.append(example['question'])
                index.append(example['index'])
                passages.append(p)
        # tokenizer 
        text = self.tokenizer(
            question, passages,
            padding="max_length",
            return_tensors='pt',
            truncation=True,
            max_length = self.maxlength
        )

        text_ids = text['input_ids']
        text_mask = text['attention_mask'].bool()
        index = torch.tensor(index)
        text_ids = text_ids.view(bsz, n_context, -1)
        text_mask = text_mask.view(bsz, n_context, -1)
        index = index.view(bsz, n_context)
        # debug
        # logger.info("index {}, scores {}, text_ids {}, text_mask {}".format(index.shape, scores.shape, text_ids.shape, text_mask.shape))
        # exit()
        return (index, scores, text_ids, text_mask)
        
