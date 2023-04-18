import lightning.pytorch as pl
import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer
)
import os
from tevatron.mrag.data import HFMTrainDataset, MTrainCollator, MTrainDataset
from tevatron.mrag.lookahead import LookaheadRMSprop
from tevatron.mrag.data import MTrainCollator
from torch.utils.data import DataLoader
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.MRetriever import MDenseModel
import logging
from tevatron.mrag.setter import fid_setter
from tevatron.mrag.distributions import RectifiedStreched, BinaryConcrete

class MaskRetrievalAugmentGeneration(pl.LightningModule):
    def __init__(self, hparams, data_args, model_args):
        super().__init__()
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        self.automatic_optimization = False
        self.params = hparams
        # dataset
        self.data_args = data_args
        self.model_args = model_args
        # collactor
        self.data_collator = MTrainCollator(in_batch_negative=False)
        # init mrag model
        self.fid = FiDT5.from_pretrained(hparams.fid_path)
        # freeze fid
        for params in self.fid.parameters():
            params.requires_grad = False
        #  create mdense
        config = AutoConfig.from_pretrained(self.model_args.model_name_or_path, num_labels=1, cache_dir=model_args.cache_dir,)
        self.mdense = MDenseModel.build(model_args, hparams, config=config)
        # other parameters
        self.n_context = self.data_args.n_context
        self.eps = self.params.eps
        self.alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(5.0), requires_grad=True)
    
    def forward(self, inputs):
        # inputs 应该是q,p,(labels, context_ids, context_mask)
        dpr_q, dpr_p = inputs['dpr_question'], inputs['dpr_passages']
        fid_a, fid_p_ids, fid_p_mask = inputs['fid_answer_ids'], inputs['fid_passage_ids'], inputs['fid_passage_mask']
        bsz, n_context, dim = dpr_p.size()
        question_rep = self.mdense.encode_query({"input_ids":dpr_q})
        passages_rep = self.mdense.encode_passage({"input_ids":dpr_p.view(bsz*n_context,-1)})
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_context, -1)
        )
        # 控制初始score范围
        origin_sim = sim

        # minmax normalization then constrain to (-10,10)
        # MIN, MAX = 99, 132
        MIN, MAX = torch.min(sim), torch.max(sim)
        sim = 20 * ( (sim - MIN)/(MAX - MIN + 0.000001) - 0.5 ) + self.bias
        # logging.info("constrained sim: {}".format(str(sim)))
        # sim = self.f(sim)* self.max_activation + self.bias_out

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.0,
        )
        # bsz, b_passages
        gates = dist.rsample()
        expected_l0 = dist.log_expected_L0().exp()
        # scalar
        loss_l0 = expected_l0.sum(-1).mean(-1)

        loss_ans, logits = fid_setter(
            model=self.fid,
            passage_ids=fid_p_ids,
            passage_masks=fid_p_mask,
            target_ids=fid_a,
            n_context=self.n_context,
            gates=gates,
            placeholder=self.mdense.placeholder
        )
        loss = self.alpha*(loss_ans-self.eps) + loss_l0
        # loss = loss_ans
        return loss, loss_ans, loss_l0, self.alpha, origin_sim, sim, gates
    
    def setup(self, stage):
        # proxies = {'http': 'http://10.130.3.188:1087'}
        # print("Using proxy {}".format(proxies))
        fid_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        dpr_tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        # prepare dataset
        train_dataset = HFMTrainDataset(data_path=self.data_args.train_dataset,
                                        dpr_tokenizer=dpr_tokenizer,
                                        fid_tokenizer=fid_tokenizer,
                                        data_args=self.data_args,
                                        )
        val_dataset = HFMTrainDataset(data_path=self.data_args.val_dataset,
                                        dpr_tokenizer=dpr_tokenizer,
                                        fid_tokenizer=fid_tokenizer,
                                        data_args=self.data_args,
                                        )
        self.train_dataset = MTrainDataset(self.data_args, train_dataset.process(), dpr_tokenizer)
        self.val_dataset = MTrainDataset(self.data_args, val_dataset.process(), dpr_tokenizer)
        world_size = os.environ.get('WORLD_SIZE')
        node_rank = os.environ.get('NODE_RANK')
        local_rank = os.environ.get('LOCAL_RANK')
        print("Set dataset at : world_size {}  node_rank {} local_rank {}".format(world_size, node_rank, local_rank))

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
        local_rank = os.environ.get('LOCAL_RANK')
        if local_rank == 0:
            print("Batch size: {}".format(batch['dpr_question'].shape[0]))
        optimizers = self.optimizers()
        loss, loss_ans, loss_l0, alpha, origin_sim, sim, gates = self(batch)
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        self.manual_backward(loss)
        # alpha梯度上升
        self.alpha.grad *= -1
        # clip gradients
        self.clip_gradients(optimizers[0], gradient_clip_val=1, gradient_clip_algorithm="norm")
        self.clip_gradients(optimizers[1], gradient_clip_val=1, gradient_clip_algorithm="norm")
        # step
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
                "alpha value": float(self.alpha),
                "mean sim":float(torch.mean(origin_sim)),
                "max sim":float(torch.max(origin_sim)),
                "min sim":float(torch.min(origin_sim)),
                "mean logits":float(torch.mean(sim)),
                "mean gates":float(torch.mean(gates))
                }, logger=True, prog_bar=True)
        
        for n, p in self.named_parameters():
            if p.requires_grad == True and p.grad is not None:
                self.log_dict({str(n):float(torch.mean(p.grad))}, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, loss_ans, loss_l0, alpha, _, _, _ = self(batch)
        metrics = {
                "val_loss_ans": float(loss_ans), 
                "val_loss_step":float(loss),
                "val_loss_l0":float(loss_l0),
                }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)
        return metrics

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                params=[
                    {
                        "params": self.mdense.parameters(),
                        "lr": self.params.learning_rate,
                    },
                    {
                        "params": [self.bias],
                        "lr": self.params.learning_rate,
                    },
                ]
            ),
            torch.optim.Adam(
                params=[self.alpha],
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
    
    def on_train_epoch_end(self) -> None:
        output_dir = self.trainer.default_root_dir + '/mdense_at_epoch' + str(self.current_epoch)
        os.makedirs(output_dir, exist_ok=True)
        self.mdense.save(output_dir=output_dir)
        return super().on_train_epoch_end()

    def constrain_alpha(self):
            # 保证在0--200之间
            self.alpha.data = torch.where(
                self.alpha.data < 0,
                torch.full_like(self.alpha.data, 0),
                self.alpha.data,
            )
            self.alpha.data = torch.where(
                self.alpha.data > 200,
                torch.full_like(self.alpha.data, 200),
                self.alpha.data,
            )
