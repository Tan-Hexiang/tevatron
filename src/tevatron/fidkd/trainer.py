import os
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import nn
import logging


logger = logging.getLogger(__name__)

class KLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(KLTrainer, self).__init__(*args, **kwargs)
        self.loss_fct = torch.nn.KLDivLoss()

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)

    def compute_loss(self, model, inputs):
        question, passages, gold_scores = inputs
        bsz, n_passages, dim = passages.size()
        question_rep = model.encode_query(question)
        passages_rep = model.encode_passage(passages.view(bsz*n_passages,-1))
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_passages, -1)
        )
        loss = self.kldivloss(sim, gold_scores)
        return loss

