import os
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.setter import fid_setter
from tevatron.mrag.distributions import RectifiedStreched, BinaryConcrete
from tevatron.mrag.plot_gradient import plot_grad_flow
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import nn
import logging


logger = logging.getLogger(__name__)

class MTrainer(Trainer):
    def __init__(self, n_context:int, alpha:float, *args, **kwargs):
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
        dpr_q, dpr_p = inputs['dpr_question'], inputs['dpr_passages']
        fid_a, fid_p_ids, fid_p_mask = inputs['fid_answer_ids'], inputs['fid_passage_ids'], inputs['fid_passage_mask']
        bsz, n_context, dim = dpr_p.size()
        question_rep = model.module.mdense.encode_query({"input_ids":dpr_q})
        passages_rep = model.module.mdense.encode_passage({"input_ids":dpr_p.view(bsz*n_context,-1)})
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_context, -1)
        )
        # 控制初始score范围
        # logging.info("sim: {}".format(str(sim)))
        # sim = model.module.f(sim)* model.module.max_activation + model.module.bias

        # minmax normalization then constrain to (-5,5)
        origin_sim = sim
        MIN, MAX = 99, 132
        sim = 10 * ( (sim - MIN)/(MAX - MIN) - 0.5 )
        # logging.info("constrained sim: {}".format(str(sim)))


        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.0,
        )
        # bsz, b_passages
        gates = dist.rsample()
        expected_l0 = dist.log_expected_L0().exp()
        # logging.info("gates : {}".format(str(gates)))
        # logging.info("l0: {}".format(expected_l0))
        # scalar
        loss_l0 = expected_l0.sum(-1).mean(-1)

        loss_ans, logits = fid_setter(
            model=model.module.fid,
            passage_ids=fid_p_ids,
            passage_masks=fid_p_mask,
            target_ids=fid_a,
            n_context=self.n_context,
            gates=gates,
            placeholder=model.module.mdense.placeholder
        )
        loss = loss_ans + self.alpha*loss_l0
        self.log({"loss_ans": float(loss_ans), "loss":float(loss)})
        self.log({"loss_l0":float(loss_l0)})
        self.log({"mean sim":float(torch.mean(origin_sim))})
        self.log({"max sim":float(torch.max(origin_sim))})
        self.log({"min sim":float(torch.min(origin_sim))})
        self.log({"mean gates":float(torch.mean(gates))})
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if (self.state.global_step % 200 == 0 or self.state.global_step<10) and self.state.global_step < 5000 :
            # logging.info("print gradient png")
            # plot_grad_flow(model.module.named_parameters(), self.args.output_dir+"/gradient", self.state.global_step)

            logging.info("Step {}  Gradient".format(self.state.global_step))
            logging.info(" loss gradient {}  is_leaf {}".format(loss.grad, loss.is_leaf))
            for n, p in model.module.named_parameters():
                # para = p.to('cpu')
                logging.info(str(n))
                logging.info(str(p.requires_grad))
                logging.info(str(p.grad))
        return loss
        