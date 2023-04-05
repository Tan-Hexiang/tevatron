import torch
import torch.nn as nn
from torch import Tensor
import logging
import os
from ..modeling.dense import DenseModel
from tevatron.mrag.fid import FiDT5
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


class MDenseModel(DenseModel):
    def __init__(self, placeholder=True, **kwargs):
        super(MDenseModel, self).__init__(**kwargs)
        self.placeholder_flag = placeholder
        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, 1, 768 * 200)
                )
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
    def __init__(self, fid:FiDT5, mdense:MDenseModel) -> None:
        super().__init__()
        self.fid = fid
        self.mdense = mdense
        # freeze fid
        for params in self.fid.parameters():
            params.requires_grad = False
    # Trainer调用
    def save(self, output_dir: str):
        self.mdense.save(output_dir)

        
