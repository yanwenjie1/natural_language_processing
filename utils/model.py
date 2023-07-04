# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/6/29
@Time    : 9:49
@File    : model.py
@Function: XX
@Other: XX
"""

import torch
import torch.nn as nn


class UIE(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0

        Reference:
            https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/model.py
        """
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            input_ids: torch.tensor,
            token_type_ids: torch.tensor,
            attention_mask=None,
            pos_ids=None,
    ) -> tuple:
        """
        forward 函数，返回开始/结束概率向量。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            tuple:  start_prob -> (batch, seq_len)
                    end_prob -> (batch, seq_len)
        """
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )
        # return sequence_output
        sequence_output = sequence_output['last_hidden_state']
        start_logits = self.linear_start(sequence_output)  # (batch, seq_len, 1)
        start_logits = torch.squeeze(start_logits, -1)  # (batch, seq_len)
        start_prob = self.sigmoid(start_logits)  # (batch, seq_len)
        end_logits = self.linear_end(sequence_output)  # (batch, seq_len, 1)
        end_logits = torch.squeeze(end_logits, -1)  # (batch, seq_len)
        end_prob = self.sigmoid(end_logits)  # (batch, seq_len)
        return start_prob, end_prob
