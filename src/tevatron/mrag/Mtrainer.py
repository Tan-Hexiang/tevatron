import os
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.setter import fid_setter
from tevatron.mrag.distributions import RectifiedStreched, BinaryConcrete

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import nn
import logging


logger = logging.getLogger(__name__)

class MTrainer(Trainer):
    def __init__(self, fid: FiDT5, n_context:int, alpha:float, *args, **kwargs):
        # freeze reader
        self.fid = fid
        for params in fid.parameters():
            params.requires_grad = False
        self.n_context = n_context
        self.alpha = alpha

        super(MTrainer, self).__init__(*args, **kwargs)

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

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs 应该是q,p,(labels, context_ids, context_mask)
        question, passages, (labels, context_ids, context_mask) = inputs
        bsz, n_passages, dim = passages.size()
        question_rep = model.encode_query(question)
        passages_rep = model.encode_passage(passages.view(bsz*n_passages,-1))
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_passages, -1)
        )

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.0,
        )
        # bsz, b_passages
        gates = dist.rsample()
        expected_L0 = dist.log_expected_L0()
        # scalar
        loss_L0 = expected_L0.sum(-1).mean(-1)

        loss_ans, logits = fid_setter(
            model=self.fid,
            passage_ids=context_ids,
            passage_masks=context_mask,
            target_ids=labels,
            n_context=self.n_context,
            gates=gates,
            placeholder=model.placeholder
        )
        loss = loss_ans + self.alpha*loss_L0
        return loss

