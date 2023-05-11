from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_constant_schedule,
    AutoConfig,
    AutoTokenizer,
    T5Tokenizer
)
from statistics import mean, median
import random
import os
from tevatron.mrag.data import HFMTrainDataset, MTrainCollator, MTrainDataset
from tevatron.mrag.lookahead import LookaheadRMSprop
from tevatron.mrag.data import MTrainCollator
from torch.utils.data import DataLoader
from tevatron.mrag.fid import FiDT5
from tevatron.mrag.MRetriever import MDenseModel
import logging
from tevatron.mrag.setter import fid_setter
from tevatron.mrag.my_distributions import RectifiedStreched, BinaryConcrete
from tevatron.fid_src.evaluation import ems

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
        self.train_collator = MTrainCollator(mode='train', in_batch_negative=False, model_args=self.model_args, data_args=self.data_args, )
        self.val_collator = MTrainCollator(mode='validate', in_batch_negative=False, model_args=self.model_args, data_args=self.data_args, )
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
        self.alpha = torch.nn.Parameter(torch.tensor(self.params.alpha), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(self.params.bias), requires_grad=True)
        # validate
        self.em = []
        self.top_3_em = []
        # debug
        self.loss_ans = []
    
    def forward(self, inputs):
        # inputs 应该是q,p,(labels, context_ids, context_mask)
        dpr_q, dpr_p = inputs['dpr_question'], inputs['dpr_passages']
        fid_a, fid_p_ids, fid_p_mask = inputs['fid_answer_ids'], inputs['fid_passage_ids'], inputs['fid_passage_mask']
        bsz = dpr_q['input_ids'].shape[0]
        n_context = int(dpr_p['input_ids'].shape[0]/bsz)
        
        logging.debug("n_context :{}".format(n_context))
        logging.debug("bsz :{}".format(bsz))
        logging.debug("dpr_p['input_ids'].shape: {}".format(dpr_p['input_ids'].shape))
        logging.debug("dpr_q['input_ids'].shape: {}".format(dpr_q['input_ids'].shape))
        question_rep = self.mdense.encode_query(dpr_q)
        passages_rep = self.mdense.encode_passage(dpr_p)
        # logging.debug("question_rep: {}".format(question_rep))
        # logging.debug("passages_rep: {}".format(passages_rep))
        # 释放显存
        del dpr_p, dpr_q
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_context, -1)
        )
        # 释放显存
        del question_rep, passages_rep
        # 控制初始score范围
        origin_sim = sim
        # logging.debug("origin sim:{}".format(origin_sim))
        # minmax normalization then constrain to (-10,10)
        # MIN, MAX = 99, 132
        MIN, MAX = torch.min(sim), torch.max(sim)
        sim = 20 * ( (sim - MIN)/(MAX - MIN + 0.00001) - 0.5 ) + self.bias
        # logging.debug("constrained sim: {}".format(str(sim)))
        # sim = self.f(sim)* self.max_activation + self.bias_out

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.1,
        )
        # bsz, b_passages
        gates = dist.rsample()
        expected_l0 = dist.expected_L0()
        logging.debug("fid_p_ids: {}".format(fid_p_ids))
        logging.debug("gates: {}".format(gates))
        logging.debug("expected_l0: {}".format(expected_l0))
        # scalar
        loss_l0 = expected_l0.mean(-1).mean(-1)

        loss_ans, logits = fid_setter(
            model=self.fid,
            passage_ids=fid_p_ids,
            passage_masks=fid_p_mask,
            target_ids=fid_a,
            n_context=self.n_context,
            gates=gates,
            placeholder=self.mdense.placeholder
        )
        # 计算loss差
        origin_loss_ans, origin_logits = self.fid(input_ids=fid_p_ids, attention_mask=fid_p_mask, labels=fid_a, return_dict=False)[:2]
        div_loss_ans = loss_ans - origin_loss_ans
        loss = self.alpha*( div_loss_ans - self.eps) + loss_l0

        logging.debug("loss_ans: {}".format(loss_ans))
        logging.debug("origin_loss_ans: {}".format(origin_loss_ans))
        logging.debug("div_loss_ans : {}".format(div_loss_ans))
        # loss = loss_ans
        return loss, div_loss_ans, loss_l0, self.alpha, origin_sim, sim, gates

    def forward_sim(self, inputs):
        # inputs 应该是q,p,(labels, context_ids, context_mask)
        dpr_q, dpr_p = inputs['dpr_question'], inputs['dpr_passages']
        logging.debug("dpr_q {}".format(dpr_q))
        logging.debug("dpr_q['input_ids'].shape {}".format(dpr_q['input_ids'].shape))
        #  bsz*n_context, dim
        logging.debug("dpr_p['input_ids'].shape {}".format(dpr_p['input_ids'].shape))
        
        bsz = dpr_q['input_ids'].shape[0]
        n_context = int(dpr_p['input_ids'].shape[0]/bsz)

        question_rep = self.mdense.encode_query(dpr_q)
        passages_rep = self.mdense.encode_passage(dpr_p)
        # 释放显存
        del dpr_p, dpr_q
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_context, -1)
        )
        return sim
    
    def setup(self, stage):
        # 先保存所有参数，便于复现
        self.save_args()
        # proxies = {'http': 'http://10.130.3.188:1087'}
        # print("Using proxy {}".format(proxies))
        logging.info("Loading Tokenizer ...")
        fid_tokenizer = T5Tokenizer.from_pretrained(self.params.t5_base_tokenizer_path)
        dpr_tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)
        # used for validate
        self.fid_tokenizer = fid_tokenizer
        logging.info("Load Tokenizer OK")
        # prepare dataset
        train_dataset = HFMTrainDataset(data_path=self.data_args.train_dataset,
                                        dpr_tokenizer=dpr_tokenizer,
                                        fid_tokenizer=fid_tokenizer,
                                        data_args=self.data_args,
                                        mode='train'
                                        )
        val_dataset = HFMTrainDataset(data_path=self.data_args.val_dataset,
                                        dpr_tokenizer=dpr_tokenizer,
                                        fid_tokenizer=fid_tokenizer,
                                        data_args=self.data_args,
                                        mode='validate'
                                        )
        logging.info("Processing datastes...")
        self.train_dataset = MTrainDataset(self.data_args, train_dataset.process(), dpr_tokenizer, mode='train')
        self.val_dataset = MTrainDataset(self.data_args, val_dataset.process(), dpr_tokenizer, mode='validate')
        world_size = os.environ.get('WORLD_SIZE')
        node_rank = os.environ.get('NODE_RANK')
        local_rank = os.environ.get('LOCAL_RANK')
        print("Set dataset at : world_size {}  node_rank {} local_rank {}".format(world_size, node_rank, local_rank))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.train_batch_size,
            collate_fn=self.train_collator,
            drop_last=True,
            num_workers=self.params.dataloader_num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.val_dataset:

            return DataLoader(
                self.val_dataset,
                batch_size=self.params.train_batch_size,
                collate_fn=self.val_collator,
                drop_last=True,
                num_workers=self.params.dataloader_num_workers,
            )

    def training_step(self, batch, batch_idx):
        # local_rank = os.environ.get('LOCAL_RANK')
        # if local_rank == 0:
            # print("Batch size: {}".format(batch['dpr_question'].shape[0]))
        optimizers = self.optimizers()
        if self.params.sample_num !=1:
            # sample and compute expectation
            for k in batch.keys():
                if k == 'fid_passage_ids' or k == 'fid_passage_mask':
                    batch[k] = batch[k].repeat(self.params.sample_num, 1, 1)
                elif k == 'dpr_question' or k == 'dpr_passages':
                    for k_d in batch[k].keys():
                        batch[k][k_d] = batch[k][k_d].repeat(self.params.sample_num, 1)
                else:
                    batch[k] = batch[k].repeat(self.params.sample_num, 1)

            loss, loss_ans, loss_l0, _, _, _, _ = self(batch)
        else:
            loss, loss_ans, loss_l0, _, _, _, _ = self(batch)
        
        # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()
        self.manual_backward(loss)
        # alpha梯度上升
        self.alpha.grad *= -1
        # clip gradients
        self.clip_gradients(optimizers[0], gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        self.clip_gradients(optimizers[1], gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        # step
        optimizers[0].step()
        optimizers[1].step()

        self.constrain_alpha()
        
        # multiple schedulers
        sch1= self.lr_schedulers()
        sch1.step()
        self.log_dict({
            "learning_rate_mdense": sch1.get_last_lr()[0],
        }, logger=True, prog_bar=True)
        
        
        
        self.log_dict({
                "loss_ans": float(loss_ans), 
                "loss_step":float(loss),
                "loss_l0":float(loss_l0),
                "alpha value": float(self.alpha),
                # "mean sim":float(torch.mean(origin_sim)),
                # "max sim":float(torch.max(origin_sim)),
                # "min sim":float(torch.min(origin_sim)),
                # "mean logits":float(torch.mean(sim)),
                # "mean gates":float(torch.mean(gates))
                }, logger=True, prog_bar=True)
        
        # for n, p in self.named_parameters():
        #     if p.requires_grad == True and p.grad is not None:
        #         self.log_dict({str(n):float(torch.mean(p.grad))}, logger=True)

    def validation_step_(self, batch, batch_idx):
        loss, loss_ans, loss_l0, alpha, _, _, _ = self(batch)
        metrics = {
                "val_loss_ans": float(loss_ans), 
                "val_loss_step":float(loss),
                "val_loss_l0":float(loss_l0),
                }
        self.log_dict(metrics, logger=True, prog_bar=True, sync_dist=True)
        return metrics
    
    def validation_step__(self, batch, batch_idx):

        batch_gold_ans = batch['raw_answers']
        # bsz, n_context, len    
        fid_a, fid_p_ids, fid_p_mask = batch['fid_answer_ids'], batch['fid_passage_ids'], batch['fid_passage_mask']
        # generate
        outputs = self.fid.generate(
                    input_ids=fid_p_ids[:,:10,:],
                    attention_mask=fid_p_mask[:,:10,:],
                    max_length=50,
                )
        
        for k, o in enumerate(outputs):
            ans = self.fid_tokenizer.decode(o, skip_special_tokens=True)
            gold_ans = batch_gold_ans[k]
            # 与fid相同的em指标，包含多答案和answer normilize，如小写，去除标点和空格等，详细见fid论文实验部分
            em_score = ems(ans, gold_ans)
            self.em.append(em_score)
            logging.debug("ans: {}".format(ans))
            logging.debug("gold_ans : {}".format(gold_ans))
            logging.debug("em_score : {}".format(em_score))
    
    def validation_step(self, batch, batch_idx):
        logging.debug("batch keys {}".format(batch.keys()))
        k=self.params.validate_k  #get top 10
        sim =  self.forward_sim(batch)

        sorted_sim, indices = torch.sort(sim, descending=True, dim=-1)
        indices = indices[:, :k]

        # bsz, n_context, len    
        scores = batch['scores']
        fid_a, fid_p_ids, fid_p_mask = batch['fid_answer_ids'], batch['fid_passage_ids'], batch['fid_passage_mask']
        batch_gold_ans = batch['raw_answers']
        new_fid_p_ids, new_fid_p_mask = [], []
        #  bsz,n_context,len
        for line in range(fid_a.shape[0]):
            new_fid_p_ids.append(torch.index_select(fid_p_ids[line], dim=0, index=indices[line]))
            new_fid_p_mask.append(torch.index_select(fid_p_mask[line], dim=0, index=indices[line]))
        new_fid_p_ids = torch.stack(new_fid_p_ids,dim=0)
        new_fid_p_mask = torch.stack(new_fid_p_mask, dim=0)
        # 释放显存
        del fid_p_mask, batch
        # generate
        outputs = self.fid.generate(
                    input_ids=new_fid_p_ids,
                    attention_mask=new_fid_p_mask,
                    max_length=50,
                )
        # 计算EM
        ans_list, gold_list, em_list = [], [], []
        for k, o in enumerate(outputs):
            ans = self.fid_tokenizer.decode(o, skip_special_tokens=True)
            gold_ans = batch_gold_ans[k]
            # 与fid相同的em指标，包含多答案和answer normilize，如小写，去除标点和空格等，详细见fid论文实验部分
            em_score = ems(ans, gold_ans)
            self.em.append(em_score)

            ans_list.append(ans)
            gold_list.append(gold_ans)
            em_list.append(em_score)
            
        # debug output to files
        if batch_idx ==2 or batch_idx ==10 or batch_idx==0:
            output_dir = self.trainer.default_root_dir+"/debug_output/"
            if not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)
            with open(output_dir+'batchid_{}_epoch_{}_step_{}_rand{}.txt'.format(batch_idx, self.current_epoch, self.global_step, random.randint(1, 10000)),'w') as f:
                f.write("----batch idx: {}----\n".format(batch_idx))
                f.write("scores: {}\n".format(scores))
                f.write("sim : {}\n".format(sim))
                f.write("indices[:,:k]: {}\n".format(indices))
                f.write("ans: {}\n".format(ans_list))
                f.write("gold_ans : {}\n".format(gold_list))
                f.write("em_score : {}\n".format(em_list))
                bsz = fid_p_ids.shape[0]
                n_context= fid_p_ids.shape[1]
                passages = self.fid_tokenizer.batch_decode(fid_p_ids.view(bsz*n_context,-1), skip_special_tokens=True)
                for i in range(bsz):
                    f.write("----------------问题{}：------------------\n".format(i))
                    for j in range(n_context):
                        f.write("\npassage {}\n".format(j))
                        f.write("passages: {}\n".format(passages[i*n_context+j]))
                        # logging.info(str(i*n_context+j))
                        f.write("scores : {}\n".format(scores[i][j]))
                        f.write("sim: {}\n".format(sim[i,j]))

    def validation_step_debug(self, batch, bacth_idx):
         # inputs 应该是q,p,(labels, context_ids, context_mask)
        dpr_q, dpr_p = batch['dpr_question'], batch['dpr_passages']
        fid_a, fid_p_ids, fid_p_mask = batch['fid_answer_ids'], batch['fid_passage_ids'], batch['fid_passage_mask']
        bsz = fid_p_ids.shape[0]
        n_context = fid_p_ids.shape[1]
        assert bsz == 1
        assert n_context==20

        question_rep = self.mdense.encode_query(dpr_q)
        passages_rep = self.mdense.encode_passage(dpr_p)
        # # 释放显存
        del dpr_p, dpr_q
        # bsz, b_passages
        sim = torch.einsum(
            'bd,bid->bi',
            question_rep,
            passages_rep.view(bsz, n_context, -1)
        )
        # 释放显存
        del question_rep, passages_rep
        # 控制初始score范围
        MIN, MAX = torch.min(sim), torch.max(sim)
        sim = 20 * ( (sim - MIN)/(MAX - MIN + 0.00001) - 0.5 ) + self.bias

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.1,
        )
        # bsz, b_passages
        gates = dist.rsample()
        gates = torch.full((bsz,n_context),1.0).cuda()
        gates = torch.full_like(gates,1.0)
        gates[:,0] = 0.0

        # expected_l0 = dist.expected_L0()
        # logging.info("n_context {}  bsz {}".format(n_context,bsz))
        logging.debug("gates: {}  {}".format(gates.shape,gates))

        loss_ans, logits = self.fid(input_ids=fid_p_ids, attention_mask=fid_p_mask, labels=fid_a, return_dict=False)[:2]
        
        logging.debug("loss_ans: {}".format(loss_ans))
        self.loss_ans.append(float(loss_ans))
        # loss = self.alpha*(loss_ans-self.eps) + loss_l0
        # logging.debug("loss_ans: {}".format(loss_ans))
        self.log_dict({"nomask_loss_ans":loss_ans}, on_step=True, prog_bar=True)
    
    def on_validation_epoch_debug(self):
        # debug
        loss_ans_mean = mean(self.loss_ans)
        logging.info("loss_ans mean:{}".format(loss_ans_mean))
        logging.info("loss_ans median:{}".format(median(self.loss_ans)))

    def on_validation_epoch_end(self) :
        em_mean = sum(self.em) / len(self.em)
        # 清空现有结果
        self.em = []
        logging.info("validate, em :{}".format(em_mean))
        self.log("em", em_mean, sync_dist=True)
        if self.global_step!=0:
            self.save_top_3_model(em_mean)
        


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
                "scheduler": get_constant_schedule_with_warmup(optimizers[0], self.params.warmup_step),
                "interval": "step",
            },
            # {
            #     "scheduler": get_constant_schedule_with_warmup(optimizers[1], 24 * 50),
            #     "interval": "step",
            # }
            
        ]
        return optimizers, schedulers
    
    # def on_train_epoch_end(self) -> None:
    #     output_dir = self.trainer.default_root_dir + '/mdense_model_at_epoch' + str(self.current_epoch) +'_step_'+str(self.global_step)+'rand'+str(random.randint(0,9999))
    #     os.makedirs(output_dir, exist_ok=True)
    #     self.mdense.save(output_dir=output_dir)
    #     # return super().on_train_epoch_end()

    def constrain_alpha(self):
            self.alpha.data = torch.where(
                self.alpha.data < 0,
                torch.full_like(self.alpha.data, 0.00001),
                self.alpha.data,
            )
            self.alpha.data = torch.where(
                self.alpha.data > self.params.alpha_up_constrain,
                torch.full_like(self.alpha.data, self.params.alpha_up_constrain),
                self.alpha.data,
            )

    def save_top_3_model(self, current_em):
        output_dir = self.trainer.default_root_dir + '/mdense_at'+'_em_'+str(current_em)+'_global_step_'+str(self.global_step)+'_epoch_'+str(self.current_epoch)
        logging.info("Current top_3_em: {}".format(self.top_3_em))
        if len(self.top_3_em)<3:
            self.top_3_em.append(current_em)
            logging.info("save model at {}。 current top_k_em {}".format(output_dir,self.top_3_em))
            self.mdense.save(output_dir= output_dir)
        else:
            # 升序排列 len(self.top_3_em)=3 
            assert len(self.top_3_em)==3
            self.top_3_em.sort(reverse=False)
            logging.debug("self.top_3_em : {}".format(self.top_3_em))
            if current_em > self.top_3_em[0]:
                self.top_3_em[0] = current_em
                logging.info("save model at {}。 current top_k_em {}".format(output_dir,self.top_3_em))
                self.mdense.save(output_dir= output_dir)
                self.top_3_em.sort(reverse=False)
    
    def save_args(self):
        output_dir = self.trainer.default_root_dir + "/args.log"
        with open(output_dir,'w') as f:
            f.write("hparams : {}\n".format(self.params))
            f.write("data_args : {}\n".format(self.data_args))
            f.write("model_args : {}\n".format(self.model_args))
        logging.info("Args saved in {}".format(output_dir))