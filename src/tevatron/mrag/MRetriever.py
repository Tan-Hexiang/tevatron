import torch
import torch.nn as nn
from torch import Tensor
import logging
from ..modeling.dense import DenseModel
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


class MDenseModel(DenseModel):
    def __init__(self, placeholder=True, **kwargs):
        if placeholder:
            self.placeholder = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(
                    torch.empty(1, 1, 768 * 200)
                )
            )
        else:
            self.placeholder = torch.zeros((1, 1, 768 * 200)),

        super(MDenseModel, self).__init__(**kwargs)
