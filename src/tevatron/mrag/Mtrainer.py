import os
from typing import Dict, List, Tuple, Optional, Any, Union

from transformers.trainer import Trainer
from tevatron.mrag.fid import FiDT5

from tevatron.mrag.plot_gradient import plot_grad_flow
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch import nn
import logging


logger = logging.getLogger(__name__)

class MTrainer(Trainer):
    def __init__(self, *args, **kwargs):

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
        loss, loss_ans, loss_l0, origin_sim, sim, gates = model(inputs)
        self.log({
                "loss_ans": float(loss_ans), 
                "loss_step":float(loss),
                "loss_l0":float(loss_l0),
                "mean sim":float(torch.mean(origin_sim)),
                "max sim":float(torch.max(origin_sim)),
                "min sim":float(torch.min(origin_sim)),
                "mean logits":float(torch.mean(sim)),
                "mean gates":float(torch.mean(gates))
                })
        return loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        loss = super().training_step(model, inputs)
        if (self.state.global_step % 50 == 0 or self.state.global_step<10)  :
        #     # logging.info("print gradient png")
        #     # plot_grad_flow(model.module.named_parameters(), self.args.output_dir+"/gradient", self.state.global_step)

            logging.info("Step {}  Gradient".format(self.state.global_step))
            logging.info(" loss gradient {}  is_leaf {}".format(loss.grad, loss.is_leaf))
            for n, p in model.module.named_parameters():
                logging.info(str(n))
                logging.info(str(p.requires_grad))
                logging.info(str(p.grad))
                # if str(n) == 'mdense.lm_q.encoder.layer.0.attention.output.LayerNorm.weight':
                if( p.requires_grad == True and p.grad is not None and 
                   str(n)=='mdense.lm_q.encoder.layer.0.attention.output.LayerNorm.weight'
                ):
                    self.log({str(n):float(torch.mean(p.grad))})
        return loss
        