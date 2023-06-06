# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:32
@File    : config.py
@Function: XX
@Other: XX
"""
import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()  # linux命令行格式下 解析配置信息
        return parser

    @staticmethod
    def initialize(parser):  # 初始化配置信息
        # args for path
        # chinese-bert-wwm-ext
        # chinese-albert-base-cluecorpussmall chinese-albert-tiny
        # chinese-roberta-wwm-ext
        # chinese-roberta-base-wwm-cluecorpussmall
        # chinese-electra-180g-base-discriminator chinese-electra-180g-small-discriminator
        # chinese-nezha-base
        parser.add_argument('--bert_dir', default='../model/chinese-albert-tiny/',
                            help='pre train model dir for uer')
        parser.add_argument('--data_dir', default='./data/ske/',
                            help='data dir for uer')

        # other args
        parser.add_argument('--task_type', default=None, type=str,
                            help='type of task, classification sequence or relationship')
        parser.add_argument('--task_type_detail', default=None, type=str,
                            help='type of task_details, sentence_pair sequence_labeling')
        parser.add_argument('--seed', type=int, default=1024,
                            help='random seed')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')
        parser.add_argument('--max_seq_len', default=256, type=int)
        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')

        # train args
        parser.add_argument('--train_epochs', default=100, type=int,
                            help='Max training epoch')
        parser.add_argument('--dropout_prob', default=0.3, type=float,
                            help='the drop out probability of pre train model ')
        parser.add_argument('--lr', default=2e-5, type=float,
                            help='bert学习率')
        parser.add_argument('--other_lr', default=2e-4, type=float,
                            help='bi-lstm和多层感知机学习率')
        parser.add_argument('--crf_lr', default=2e-3, type=float,
                            help='条件随机场学习率')
        parser.add_argument('--max_grad_norm', default=0.5, type=float,
                            help='max grad clip')
        parser.add_argument('--warmup_proportion', default=0.1, type=float)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--use_lstm', type=bool, default=True,
                            help='use lstm or not')
        parser.add_argument('--num_layers', default=2, type=int,
                            help='lstm层数大小 形成堆叠lstm')
        parser.add_argument('--dropout', default=0.3, type=float,
                            help='对多层lstm的输出dropout')
        parser.add_argument('--use_crf', type=bool, default=True,
                            help='use crf or not')
        parser.add_argument('--use_advert_train', type=bool, default=True,
                            help='use advert or not --PGD')

        # GlobalPointer
        parser.add_argument('--use_gp', type=bool, default=True,
                            help='是否使用GlobalPointer')
        parser.add_argument('--RoPE', type=bool, default=True,
                            help='是否使用旋转位置编码')
        parser.add_argument('--head_size', type=int, default=64,
                            help='还不太确定是干什么的')
        parser.add_argument('--use_efficient_globalpointer', type=bool, default=True,
                            help='是否使用更高效的globalpointer')
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()

