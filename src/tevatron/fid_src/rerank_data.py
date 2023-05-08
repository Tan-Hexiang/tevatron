import json
import torch
from asyncio.log import logger
import logging

def load_data_for_rerank(file, title_prefix='title:', passage_prefix='context:'):
    with open(file, 'r') as f:
        data = json.load(f)
    # index question answers passage_id passage score
    passage_form = title_prefix+"{} "+passage_prefix+"{}"
    index = 0
    new_data = []
    logger.info("Question lines: {}".format(len(data)))
    logger.info("Question {}: Positive ctxs lines: {}".format(1000, len(data[1000]['positive_ctxs'])))
    logger.info("Question {}: Negative ctxs lines: {}".format(1000, len(data[1000]['negative_ctxs'])))
    for example in data:
        for p in example['positive_ctxs']:
            line = {}
            line['index'] = index
            index = index+1
            line['question'] = example['question']
            line['answer'] = example['answers']
            line['passage_id'] = p['passage_id']
            line['passage'] = passage_form.format(p['title'],p['text'])
            line['score'] = 1
            new_data.append(line)
        for p in example['negative_ctxs']:
            line = {}
            line['index'] = index
            index = index+1
            line['question'] = example['question']
            line['answer'] = example['answers']
            line['passage_id'] = p['passage_id']
            line['passage'] = passage_form.format(p['title'],p['text'])
            line['score'] = 0
            new_data.append(line)
    logger.info("Load data {} lines".format(len(new_data)))    
    logger.info("First line: {}".format(new_data[0]))
    return new_data

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data
                ):
        # index question answers passage_id passage score
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class RerankCollator_for_positive(object):
    def __init__(self, tokenizer, 
                text_maxlength=200):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength

    def __call__(self, batch):
        index = torch.Tensor([x['index'] for x in batch])
        question = [x['question']  for x in batch]
        passage = [x['passage'] for x in batch]
        score = torch.Tensor([x['score']  for x in batch])
        text = self.tokenizer(
            question, passage,
            padding="max_length",
            return_tensors='pt',
            truncation=True,
            max_length = self.text_maxlength
        )
        text_ids = text['input_ids']
        text_mask = text['attention_mask'].bool()
        # logger.info("Shape: index {}, text_ids {}, text_mask {}, score {} ".format(index.shape, text_ids.shape, text_mask.shape, score.shape))
        return (index, text_ids, text_mask, score)
        



if __name__=='__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    data= load_data_for_rerank("/data/tanhexiang/CF_QA/data/retriever/nq-dev.json")
    # dataset= Dataset(data)
    # for k,example in enumerate(dataset):
    #     print(example['answers'])
    #     exit(0)