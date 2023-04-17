import pytorch_lightning as pl
import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer
)
from tevatron.mrag.lookahead import LookaheadRMSprop
from tevatron.mrag.data import MTrainCollator
from torch.utils.data import DataLoader

class MaskRetrievalAugmentGeneration(pl.LightningModule):
    def __init__(self, model, train_dataset, hparams, val_dataset = None):
        super().__init__()
        self.hparams = hparams
        # dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # collactor
        self.data_collator = MTrainCollator(in_batch_negative=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.hparams.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.val_dataset:

            return DataLoader(
                self.val_dataset,
                batch_size=self.hparams.train_batch_size,
                collate_fn=self.data_collator,
                drop_last=True,
                num_workers=self.hparams.dataloader_num_workers,
                shuffle=True,
            )

    def training_step(self, batch, batch_idx):
        loss, loss_ans, loss_l0, origin_sim, sim, gates = self.model(batch)
        self.log_dict({
                "loss_ans": float(loss_ans), 
                "loss_step":float(loss),
                "loss_l0":float(loss_l0),
                "mean sim":float(torch.mean(origin_sim)),
                "max sim":float(torch.max(origin_sim)),
                "min sim":float(torch.min(origin_sim)),
                "mean logits":float(torch.mean(sim)),
                "mean gates":float(torch.mean(gates))
                }, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_ans, loss_l0, _, _, _ = self.model(batch)
        metrics = {
                "val_loss_ans": float(loss_ans), 
                "val_loss_step":float(loss),
                "val_loss_l0":float(loss_l0),
                }
        self.log_dict(metrics, logger=True)
        return metrics

    def configure_optimizers(self):
        optimizers = [
            LookaheadRMSprop(
                params=[
                    {
                        "params": self.gate.g_hat.parameters(),
                        "lr": self.hparams.learning_rate,
                    },
                    {
                        "params": self.gate.placeholder.parameters()
                        if isinstance(self.gate.placeholder, torch.nn.ParameterList)
                        else [self.gate.placeholder],
                        "lr": self.hparams.learning_rate_placeholder,
                    },
                ],
                centered=True,
            ),
            LookaheadRMSprop(
                params=[self.alpha]
                if isinstance(self.alpha, torch.Tensor)
                else self.alpha.parameters(),
                lr=self.hparams.learning_rate_alpha,
            ),
        ]

        schedulers = [
            {
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], 24 * 50),
                "interval": "step",
            },
            get_constant_schedule(optimizers[1]),
        ]
        return optimizers, schedulers

    def optimizer_step(
        self,
        current_epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None

        elif optimizer_idx == 1:
            # 优化alpha时反转梯度，对应lagrange relaxation中的max min
            for i in range(len(self.alpha)):
                if self.alpha[i].grad:
                    self.alpha[i].grad *= -1

            optimizer.step()
            optimizer.zero_grad()
            for g in optimizer.param_groups:
                for p in g["params"]:
                    p.grad = None
            # 保证在0--200之间
            for i in range(len(self.alpha)):
                self.alpha[i].data = torch.where(
                    self.alpha[i].data < 0,
                    torch.full_like(self.alpha[i].data, 0),
                    self.alpha[i].data,
                )
                self.alpha[i].data = torch.where(
                    self.alpha[i].data > 200,
                    torch.full_like(self.alpha[i].data, 200),
                    self.alpha[i].data,
                )
