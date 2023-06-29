# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:18
@File    : base_model.py
@Function: XX
@Other: XX
"""
import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel, AlbertModel, RoFormerModel, NezhaModel, \
    LongformerModel, MT5ForConditionalGeneration, ErnieModel


class BaseModel(nn.Module):
    """
    基础的预训练模型
    """

    def __init__(self, bert_dir, dropout_prob, model_name=None):
        """
        利用transformers库加载预训练torch模型
        :param bert_dir: 预训练模型的路径
        :param dropout_prob: 对预训练模型的输出进行dropout
        :param model_name: 预训练模型的名字
        """
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')
        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        if 'albert' in model_name:
            self.bert_module = AlbertModel.from_pretrained(bert_dir,
                                                           output_hidden_states=False,
                                                           hidden_dropout_prob=dropout_prob)
        elif 'robert' in model_name or 'bert' in model_name:
            # 想要接收和使用来自其他隐藏层的输出，而不仅仅是 last_hidden_state 就用True
            self.bert_module = BertModel.from_pretrained(bert_dir,
                                                         output_hidden_states=False,
                                                         hidden_dropout_prob=dropout_prob)
        elif 'roformer' in model_name:
            self.bert_module = RoFormerModel.from_pretrained(bert_dir,
                                                             output_hidden_states=False,
                                                             hidden_dropout_prob=dropout_prob)
        elif 'nezha' in model_name:
            self.bert_module = NezhaModel.from_pretrained(bert_dir,
                                                          output_hidden_states=False,
                                                          hidden_dropout_prob=dropout_prob)
        elif 'longformer' in model_name:
            LongformerModel.base_model_prefix = 'bert'
            self.bert_module = LongformerModel.from_pretrained(bert_dir,
                                                               output_hidden_states=False,
                                                               hidden_dropout_prob=dropout_prob)
        elif 't5' in model_name:
            self.bert_module = MT5ForConditionalGeneration.from_pretrained(bert_dir,
                                                                           output_hidden_states=False)
        elif 'uie' in model_name:
            # self.bert_module = UIE(ErnieModel)
            # model = torch.load(os.path.join(bert_dir, 'pytorch_model.pt'), map_location=torch.device('cpu'))
            # self.bert_module.load_state_dict(model, strict=True)
            self.bert_module = torch.load(os.path.join(bert_dir, 'pytorch_model.bin'))  # 模型结构见：UIE
            self.bert_module.config = None
        else:
            self.bert_module = AutoModel.from_pretrained(bert_dir,
                                                         output_hidden_states=False,
                                                         hidden_dropout_prob=dropout_prob)
        # elif 'nezha' in model_name:
        #     self.bert_module = AutoModel.from_pretrained(bert_dir,
        #                                                  output_hidden_states=False,
        #                                                  hidden_dropout_prob=dropout_prob)

        self.base_config = self.bert_module.config

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
