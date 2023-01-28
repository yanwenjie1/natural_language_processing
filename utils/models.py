# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/5
@Time    : 13:46
@File    : models.py
@Function: XX
@Other: XX
"""
import torch
import torch.nn as nn
import numpy as np
import logging
import pynvml
import os
import config
from tqdm import tqdm
from torchcrf import CRF
from utils.adversarial_training import PGD
from utils.base_model import BaseModel
from utils.functions import load_model_and_parallel, build_optimizer_and_scheduler, get_sinusoid_encoding_table


class BaseFeature:
    def __init__(self, token_ids, attention_masks, token_type_ids, labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        # labels
        self.labels = labels


class Classification(BaseModel):
    def __init__(self, args):
        super(Classification, self).__init__(bert_dir=args.bert_dir,
                                             dropout_prob=args.dropout_prob,
                                             model_name=args.model_name)
        self.args = args
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.base_config.hidden_size, args.num_tags)
        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.base_config.initializer_range)

    def forward(self, token_ids, attention_masks, token_type_ids):
        bert_outputs = self.bert_module(input_ids=token_ids,  # vocab 对应的id
                                        attention_mask=attention_masks,  # pad mask 情况
                                        token_type_ids=token_type_ids  # CLS *** SEP *** SEP 区分第一个和第二个句子
                                        )

        # 在这一个地方详细的描述一次：
        # 输出是namedtuple或字典对象  可以通过属性或序号访问模型输出结果
        # 输入的维度是：input_ids:[batch_size, tokens] (tokens=max_len)
        # outputs一共四个属性、last_hidden_state, pooler_output, hidden_states, attentions
        # 一般不取hidden_states太多了 但是某种用途下 取后n层做平均 可能会有更好的效果 无论是分类还是序列任务

        # outputs[0]是last_hidden_state, 是基于token表示的， 对于实体命名、问答非常有用、实际包括四个维度[layers, batches, tokens, features]
        # 不获取全部的12层输出的条件下 是 [batches, tokens, features]
        # outputs[1]是pooler_output 整个输入的合并表达 形状为[batches, features]
        # 是由 hidden_states获取了cls标签后进行了dense 和 Tanh后的输出 dense层是768*768的全连接, 就是调整一下输出
        # 所以bert的model并不是简单的组合返回, 一般来说，需要使用bert做句子级别的任务，可以使用pooled_output结果做baseline， 进一步的微调可以使用last_hidden_state的结果
        # 分类任务的时候 再乘 [features, num_tags]的线性层 实现 one_hot的输出

        # 常规
        # seq_out = bert_outputs[1]  # [batchsize, features] 有空的时候这里要看看   bert_outputs['pooler_output']
        # # seq_out1 = bert_outputs[1]  # bert_outputs['pooler_output']
        # seq_out = self.dropout(seq_out)
        # seq_out = self.classifier(seq_out)  # [batchsize, num_tags]

        # 平均池化
        seq_out = bert_outputs[0].mean(1)
        seq_out = self.dropout(seq_out)
        seq_out = self.classifier(seq_out)  # [batchsize, num_tags]

        return seq_out


class SequenceLabeling(BaseModel):
    def __init__(self, args):
        super(SequenceLabeling, self).__init__(bert_dir=args.bert_dir,
                                               dropout_prob=args.dropout_prob,
                                               model_name=args.model_name)
        self.args = args
        gpu_ids = self.args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])  # 指示当前设备
        self.args.lstm_hidden = self.base_config.hidden_size // 2

        if self.args.use_lstm:  # 如果使用lstm层 这边默认保证lstm输入输出维度一致 仅作上下文语义调整
            # 这里num_layers是同一个time_step的结构堆叠 Lstm堆叠层数与time step无关
            self.lstm = nn.LSTM(self.base_config.hidden_size,
                                self.args.lstm_hidden,
                                self.args.num_layers,
                                bidirectional=True,
                                batch_first=True,
                                dropout=self.args.dropout)
            self.linear = nn.Linear(self.args.lstm_hidden * 2, self.args.num_tags)  # lstm之后的线性层
            init_blocks = [self.linear]
        else:
            self.mid_linear = nn.Sequential(
                nn.Linear(self.base_config.hidden_size, self.base_config.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.args.dropout))
            self.linear = nn.Linear(self.base_config.hidden_size, self.args.num_tags)  # lstm之后的线性层
            init_blocks = [self.mid_linear, self.linear]

        self.criterion = nn.CrossEntropyLoss()
        self._init_weights(init_blocks, initializer_range=self.base_config.initializer_range)

        self.crf = CRF(self.args.num_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device, requires_grad=True)
        c0 = torch.randn(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device, requires_grad=True)
        return h0, c0

    def init_hidden_zero(self, batch_size):
        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        h0 = torch.zeros(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device, requires_grad=True)
        c0 = torch.zeros(2 * self.args.num_layers, batch_size, self.args.lstm_hidden, device=self.device, requires_grad=True)
        return h0, c0

    def forward(self, token_ids, attention_masks, token_type_ids, model='Train'):
        bert_outputs = self.bert_module(input_ids=token_ids,  # vocab 对应的id
                                        attention_mask=attention_masks,  # pad mask 情况
                                        token_type_ids=token_type_ids  # CLS *** SEP *** SEP 区分第一个和第二个句子
                                        )
        print(token_ids.size())
        for i in tqdm(range(10000)):
            _ = self.bert_module(input_ids=token_ids,  # vocab 对应的id
                                 attention_mask=attention_masks,  # pad mask 情况
                                 token_type_ids=token_type_ids  # CLS *** SEP *** SEP 区分第一个和第二个句子
                                 )
        # 常规
        seq_out = bert_outputs[0]  # bert_outputs['last_hidden_state']
        batch_size = seq_out.size(0)

        if self.args.use_lstm:
            if model == 'Train':
                hidden = self.init_hidden(batch_size)
            else:
                hidden = self.init_hidden_zero(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
            # contiguous() 函数 返回一个连续内存空间的deepcopy的张量 作用有两点：
            # 1、当一个Tensor经过 tensor.transpose()、tensor.permute()等这类维度变换函数后，内存并不是连续的，
            # 而tensor.view()维度变形函数的要求是需要Tensor的内存连续，所以在运行tensor.view()之前，先使用 tensor.contiguous()，防止报错
            # 2、维度变换函数是进行的浅拷贝操作，即view操作会连带原来的变量一同变形，这是不合法的，也会报错
            seq_out = seq_out.contiguous().view(-1, self.args.lstm_hidden * 2)
            seq_out = self.linear(seq_out)
            seq_out = seq_out.contiguous().view(batch_size, -1, self.args.num_tags)
        else:
            seq_out = self.mid_linear(seq_out)
            seq_out = self.linear(seq_out)  # [batchsize, max_len, 256]

        return seq_out


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(self, hidden_size, heads, head_size, rope=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads  # 可以类似理解为attention的head? n个头就意味着有n个矩阵
        self.head_size = head_size
        self.RoPE = rope
        self.tril_mask = tril_mask

        self.dense = nn.Linear(hidden_size, heads * head_size * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [batch_size, max_len, embeddings]
            mask: [batch_size, max_len], padding部分为0
        '''
        sequence_output = self.dense(inputs)  # sequence_output: [batch_size, max_len, heads*head_size*2]
        # chunk的方法做的是对张量进行分块，返回一个张量列表。
        # [batchsize, 150, 8, 64*2]
        # list of [batch_size, max_len, head_size*2] len(list) = heads
        sequence_output = torch.chunk(sequence_output, self.heads, dim=-1)
        # stack:沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        sequence_output = torch.stack(sequence_output, dim=-2)  # [batch_size, max_len, heads, head_size*2]
        # qw:[batch_size, max_len, heads, head_size], kw:[batch_size, max_len, heads, head_size]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)  # [btz, heads, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(nn.Module):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """
    def __init__(self, hidden_size, heads, head_size, rope=True, max_len=512, use_bias=True, tril_mask=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = rope
        self.tril_mask = tril_mask

        self.p_dense = nn.Linear(hidden_size, head_size * 2, bias=use_bias)
        self.q_dense = nn.Linear(head_size * 2, heads * 2, bias=use_bias)
        if self.RoPE:
            self.position_embedding = RoPEPositionEncoding(max_len, head_size)

    def forward(self, inputs, mask=None):
        ''' inputs: [batch_size, max_len, embeddings]
            mask: [batch_size, max_len], padding部分为0
        '''
        sequence_output = self.p_dense(inputs)  # sequence_output: [batch_size, max_len, head_size*2]
        # qw,kw: [batch_size, max_len, head_size]
        qw, kw = sequence_output[..., :self.head_size], sequence_output[..., self.head_size:]

        # ROPE编码
        if self.RoPE:
            qw = self.position_embedding(qw)
            kw = self.position_embedding(kw)

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size**0.5  # [btz, seq_len, seq_len]
        bias_input = self.q_dense(sequence_output)  # [..., heads*2]
        bias = torch.stack(torch.chunk(bias_input, self.heads, dim=-1), dim=-2).transpose(1,2)  # [btz, head_size, seq_len,2]
        logits = logits.unsqueeze(1) + bias[..., :1] + bias[..., 1:].transpose(2, 3)  # [btz, head_size, seq_len, seq_len]

        # 排除padding
        if mask is not None:
            attention_mask1 = 1 - mask.unsqueeze(1).unsqueeze(3)  # [btz, 1, seq_len, 1]
            attention_mask2 = 1 - mask.unsqueeze(1).unsqueeze(2)  # [btz, 1, 1, seq_len]
            logits = logits.masked_fill(attention_mask1.bool(), value=-float('inf'))
            logits = logits.masked_fill(attention_mask2.bool(), value=-float('inf'))

        # 排除下三角
        if self.tril_mask:
            logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12

        return logits


class GlobalPointerNer(BaseModel):
    def __init__(self, args):
        super(GlobalPointerNer, self).__init__(bert_dir=args.bert_dir,
                                               dropout_prob=args.dropout_prob,
                                               model_name=args.model_name)
        self.args = args
        if args.use_efficient_globalpointer:
            self.global_pointer = EfficientGlobalPointer(hidden_size=self.base_config.hidden_size,
                                                         heads=self.args.num_tags,
                                                         head_size=self.args.head_size)
        else:
            self.global_pointer = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                                heads=self.args.num_tags,
                                                head_size=self.args.head_size)

    def forward(self, token_ids, attention_masks, token_type_ids):
        output = self.bert_module(token_ids, attention_masks, token_type_ids)
        sequence_output = output[0]  # [batch_size, seq_len, hidden_size]
        logits = self.global_pointer(sequence_output, attention_masks.gt(0).long())
        return logits
        # if labels is None:
        #     # scale返回
        #     return logits
        #
        # loss = self.criterion(logits, labels)
        # return loss, logits


class GlobalPointerRe(BaseModel):
    def __init__(self, args):
        super(GlobalPointerRe, self).__init__(bert_dir=args.bert_dir,
                                              dropout_prob=args.dropout_prob,
                                              model_name=args.model_name)
        self.args = args
        if args.use_efficient_globalpointer:
            self.entity_output = EfficientGlobalPointer(hidden_size=self.base_config.hidden_size,
                                                        heads=2,
                                                        head_size=self.args.head_size)
            self.head_output = EfficientGlobalPointer(hidden_size=self.base_config.hidden_size,
                                                      heads=self.args.num_tags,
                                                      head_size=self.args.head_size,
                                                      rope=False,
                                                      tril_mask=False)
            self.tail_output = EfficientGlobalPointer(hidden_size=self.base_config.hidden_size,
                                                      heads=args.num_tags,
                                                      head_size=self.args.head_size,
                                                      rope=False,
                                                      tril_mask=False)
        else:
            self.entity_output = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                               heads=2,
                                               head_size=self.args.head_size)
            self.head_output = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                             heads=self.args.num_tags,
                                             head_size=self.args.head_size,
                                             rope=False,
                                             tril_mask=False)
            self.tail_output = GlobalPointer(hidden_size=self.base_config.hidden_size,
                                             heads=args.num_tags,
                                             head_size=self.args.head_size,
                                             rope=False,
                                             tril_mask=False)

    def forward(self, token_ids, attention_masks, token_type_ids, head_labels=None, tail_labels=None, entity_labels=None):
        output = self.bert_module(token_ids, attention_masks, token_type_ids)  # [btz, seq_len, hdsz]
        sequence_output = output[0]  # [batch_size, seq_len, hidden_size]
        mask = attention_masks

        entity_output = self.entity_output(sequence_output, mask)  # [btz, heads, seq_len, seq_len]
        head_output = self.head_output(sequence_output, mask)  # [btz, heads, seq_len, seq_len]
        tail_output = self.tail_output(sequence_output, mask)  # [btz, heads, seq_len, seq_len]
        return entity_output, head_output, tail_output
        # if head_labels is None:
        #   return entity_output, head_output, tail_output
        # loss = self.criterion([entity_output, head_output, tail_output], [entity_labels, head_labels, tail_labels])
        # return loss


class ViterbiDecoder(object):
    """苏剑林的Viterbi解码算法基类
    """
    def __init__(self, trans, starts=None, ends=None):
        self.trans = trans
        self.num_labels = len(trans)
        self.non_starts = []
        self.non_ends = []
        if starts is not None:
            for i in range(self.num_labels):
                if i not in starts:
                    self.non_starts.append(i)
        if ends is not None:
            for i in range(self.num_labels):
                if i not in ends:
                    self.non_ends.append(i)

    def decode(self, nodes):
        """nodes.shape=[seq_len, num_labels]
        """
        # 预处理
        nodes[0, self.non_starts] -= np.inf
        nodes[-1, self.non_ends] -= np.inf

        # 动态规划
        labels = torch.from_numpy(np.arange(self.num_labels).reshape((1, -1))).to('cuda:0')
        scores = nodes[0].reshape((-1, 1))
        paths = labels
        for l in range(1, len(nodes)):
            M = scores + self.trans + nodes[l].reshape((1, -1))
            idxs = M.argmax(0)
            aaa, bbb = M.max(0)
            scores = aaa.reshape((-1, 1))
            paths = torch.cat((paths[:, idxs], labels), 0)

        # 最优路径
        return paths[:, scores[:, 0].argmax()]


class RoPEPositionEncoding(nn.Module):
    """旋转式位置编码: https://kexue.fm/archives/8265
    """
    def __init__(self, max_position, embedding_size):
        super(RoPEPositionEncoding, self).__init__()
        position_embeddings = get_sinusoid_encoding_table(max_position, embedding_size)  # [seq_len, hdsz]
        cos_position = position_embeddings[:, 1::2].repeat_interleave(2, dim=-1)
        sin_position = position_embeddings[:, ::2].repeat_interleave(2, dim=-1)
        # register_buffer是为了最外层model.to(device)，不用内部指定device
        self.register_buffer('cos_position', cos_position)
        self.register_buffer('sin_position', sin_position)

    def forward(self, qw, seq_dim=-2):
        # [batch_size, max_len, heads, head_size] / [batch_size, max_len, head_size]
        # 默认最后两个维度为[seq_len, hdsz]? 不太对吧
        seq_len = qw.shape[seq_dim]
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1).reshape_as(qw)
        return qw * self.cos_position[:seq_len] + qw2 * self.sin_position[:seq_len]

