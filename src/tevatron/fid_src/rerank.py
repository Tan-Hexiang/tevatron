import types
import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
import logging

class Reranker(transformers.BertForSequenceClassification):
    def __init__(self, config, loss_mode='kl') -> None:
        super().__init__(config)
        self.loss_mode = loss_mode
        logging.info("Rerank loss mode: {}".format(self.loss_mode))
        self.loss_fct = torch.nn.NLLLoss()    
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask=None, labels=None, loss_mode=None,print_info=False, i=None, **kwargs):
        '''
        labels: bsz
        input_ids: bsz*len
        attention_mask: bsz*len
        output: 
            loss: 1 dim tensor
            logits: bsz
        '''
        self.loss_mode = loss_mode

        if self.loss_mode == 'ce' :
            (loss,output) = super().forward(input_ids, attention_mask,labels=labels.to(torch.long),**kwargs)
            # 模型是num_labels=2，https://github.com/huggingface/transformers/issues/580 只取第二个作为最后的分数
            output = self.softmax(output)
            logits = output[:,1]
        elif self.loss_mode == 'kl' :
            bsz, n_context = input_ids.shape[0], input_ids.shape[1]
            (output,) = super().forward(input_ids.view(bsz*n_context,-1), attention_mask.view(bsz*n_context,-1), **kwargs)
            output = self.softmax(output)
            logits = output[:,1].view(bsz,n_context)
            if labels==None:
                loss = None
            else:
                loss = self.kldivloss(logits,labels)
        else:
            logging.error("Unexpected loss mode:{}".format(self.loss_mode))
            exit()

        if print_info:
            logging.info("output :{}".format(output))
            logging.info("output shape {}".format(output.shape))
            logging.info("logits : {}".format(logits))
            logging.info("labels : {}".format(labels))
            logging.info("loss : {}".format(loss))
            # exit()

        if i !=None and i%1000==0:
            logging.info("----------------------index {}------------------------".format(i))
            logging.info("loss {}, label {}, logits {}".format(loss.shape,labels.shape,logits.shape))
            logging.info("loss {}, label {}, logits {}".format(loss,labels,logits))

        return loss,logits

    
    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        loss_fct = torch.nn.KLDivLoss()
        return loss_fct(score, gold_score)
