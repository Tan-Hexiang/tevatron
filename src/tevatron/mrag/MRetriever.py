import torch
import torch.nn as nn
from torch import Tensor
import logging
import os
from ..modeling.dense import DenseModel
from tevatron.mrag.fid import FiDT5
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments
from tevatron.mrag.setter import fid_setter
from tevatron.mrag.distributions import RectifiedStreched, BinaryConcrete

logger = logging.getLogger(__name__)


class MDenseModel(DenseModel):
    def __init__(self, placeholder=True, **kwargs):
        super(MDenseModel, self).__init__(**kwargs)
        self.placeholder_flag = placeholder
        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, 1, 768 * 200)
                ), requires_grad=True
            )
        else:
            self.placeholder = torch.zeros((1, 1, 768 * 200)),
    
    def save(self, output_dir: str):
        super(MDenseModel, self).save(output_dir)

        placeholder_path = os.path.join(output_dir, 'placeholder.pt')
        torch.save(self.placeholder, placeholder_path)

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):   
        model = super().load(model_name_or_path, **hf_kwargs)
        if model.placeholder_flag:
            placeholder_path = os.path.join(model_name_or_path, 'placeholder.pt')
            model.placeholder = torch.load(placeholder_path)

class mrag(nn.Module):
    def __init__(self, fid:FiDT5, mdense:MDenseModel, n_context, eps, max_activation=10) -> None:
        super().__init__()
        self.n_context = n_context
        self.eps = eps
        self.fid = fid
        self.mdense = mdense
        # freeze fid
        for params in self.fid.parameters():
            params.requires_grad = False
        #  constrain output range
        self.alpha = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.max_activation = max_activation
        self.f = torch.nn.Tanh()

    # Trainer调用
    def save(self, output_dir: str):
        self.mdense.save(output_dir)
    
    def forward(self, inputs):
        logging.info("gpu memory:{}".format(torch.cuda.max_memory_allocated(0)))
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
        logging.info("dpr sim details: {}".format(sim))
        # 控制初始score范围
        origin_sim = sim
        # logging.info("sim: {}".format(str(sim)))
        logging.info("gpu memory:{}".format(torch.cuda.max_memory_allocated(0)))

        # minmax normalization then constrain to (-10,10)
        # MIN, MAX = 99, 132
        MIN, MAX = torch.min(sim), torch.max(sim)
        assert (MAX-MIN)!=0
        sim = 20 * ( (sim - MIN)/(MAX - MIN) - 0.5 ) + self.bias
        logging.info("sim details:{}".format(sim))
        # logging.info("constrained sim: {}".format(str(sim)))
        # sim = self.f(sim)* self.max_activation + self.bias_out

        dist = RectifiedStreched(
            BinaryConcrete(torch.full_like(sim, 0.2), sim), l=-0.2, r=1.0,
        )
        # bsz, b_passages
        gates = dist.rsample()
        logging.info("gates: {}".format(gates))
        expected_l0 = dist.log_expected_L0().exp()
        logging.info("l0: {}".format(expected_l0))
        # logging.info("gates : {}".format(str(gates)))
        # logging.info("l0: {}".format(expected_l0))
        # scalar
        loss_l0 = expected_l0.sum(-1).mean(-1)
        logging.info("loss_l0:{}".format(loss_l0))

        loss_ans, logits = fid_setter(
            model=self.fid,
            passage_ids=fid_p_ids,
            passage_masks=fid_p_mask,
            target_ids=fid_a,
            n_context=self.n_context,
            gates=gates,
            placeholder=self.mdense.placeholder
        )
        logging.info("loss_ans {}".format(loss_ans))
        logging.info("alpha {}".format(self.alpha))

        loss = self.alpha*(loss_ans-self.eps) + loss_l0

        logging.info("loss {}".format(loss))
        logging.info("fid logits:{}".format(logits))


        logging.info("gpu memory:{}".format(torch.cuda.max_memory_allocated(0)))
        # loss = loss_ans
        return loss, loss_ans, loss_l0, self.alpha, origin_sim, sim, gates

    
        

        
