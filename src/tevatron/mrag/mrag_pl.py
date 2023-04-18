import lightning.pytorch as pl
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
from tevatron.mrag.MRetriever import mrag

class MaskRetrievalAugmentGeneration(pl.LightningModule):
    def __init__(self, model:mrag, train_dataset, hparams, val_dataset = None):
        super().__init__()
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False
        self.model = model
        self.params = hparams
        # dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # collactor
        self.data_collator = MTrainCollator(in_batch_negative=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.train_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.params.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.val_dataset:

            return DataLoader(
                self.val_dataset,
                batch_size=self.params.train_batch_size,
                collate_fn=self.data_collator,
                drop_last=True,
                num_workers=self.params.dataloader_num_workers,
                shuffle=True,
            )

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        loss, loss_ans, loss_l0, alpha, origin_sim, sim, gates = self.model(batch)
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        self.manual_backward(loss)
        # alpha梯度上升
        self.model.alpha.grad *= -1
        # clip gradients
        self.clip_gradients(optimizers[0], gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(optimizers[1], gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        optimizers[0].step()
        optimizers[1].step()

        self.constrain_alpha()
        
        # multiple schedulers
        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

        
        
        self.log_dict({
                "loss_ans": float(loss_ans), 
                "loss_step":float(loss),
                "loss_l0":float(loss_l0),
                "alpha": float(self.model.alpha),
                "alpha grad": float(self.model.alpha.grad),
                "mean sim":float(torch.mean(origin_sim)),
                "max sim":float(torch.max(origin_sim)),
                "min sim":float(torch.min(origin_sim)),
                "mean logits":float(torch.mean(sim)),
                "mean gates":float(torch.mean(gates))
                }, logger=True, prog_bar=True)
        
        for n, p in self.model.mdense.named_parameters():
            if p.requires_grad == True and p.grad is not None:
                self.log_dict({str(n):float(torch.mean(p.grad))}, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, loss_ans, loss_l0, alpha, _, _, _ = self.model(batch)
        metrics = {
                "val_loss_ans": float(loss_ans), 
                "val_loss_step":float(loss),
                "val_loss_l0":float(loss_l0),
                }
        self.log_dict(metrics, logger=True, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                params=[
                    {
                        "params": self.model.mdense.parameters(),
                        "lr": self.params.learning_rate,
                    },
                    {
                        "params": [self.model.bias],
                        "lr": self.params.learning_rate,
                    },
                ]
            ),
            torch.optim.Adam(
                params=[self.model.alpha],
                lr=self.params.learning_rate_alpha,
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

    def constrain_alpha(self):
            # 保证在0--200之间
            self.model.alpha.data = torch.where(
                self.model.alpha.data < 0,
                torch.full_like(self.model.alpha.data, 0),
                self.model.alpha.data,
            )
            self.model.alpha.data = torch.where(
                self.model.alpha.data > 200,
                torch.full_like(self.model.alpha.data, 200),
                self.model.alpha.data,
            )
