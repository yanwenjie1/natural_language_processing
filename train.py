# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : train.py
@Function: XX
@Other: XX
"""
import datetime
import os
import shutil
import logging
import torch
from utils.functions import set_seed, set_logger, save_json, gp_collate_fn, gp_collate_fn_re
from utils.train_models import TrainClassification, TrainSequenceLabeling, TrainGlobalPointerNer, TrainGlobalPointerRe
import config
import json
from torch.utils.data import Dataset, SequentialSampler, DataLoader
import pickle


args = config.Args().get_parser()
logger = logging.getLogger(__name__)


class NerDataset(Dataset):
    def __init__(self, features):
        # self.callback_info = callback_info
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        self.labels = [torch.tensor(example.labels).long() for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index], 'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index], 'labels': self.labels[index]}

        return data


class GPDataset(Dataset):
    def __init__(self, features, args):
        # self.callback_info = callback_info
        self.nums = len(features)
        self.args = args
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]
        # self.labels = gp_entity_to_label(features, args)
        self.labels = [example.labels for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index], 'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index], 'labels': self.labels[index], 'args': self.args}

        return data


if __name__ == '__main__':
    args.data_name = os.path.basename(os.path.abspath(args.data_dir))
    args.model_name = os.path.basename(os.path.abspath(args.bert_dir))
    args.save_path = os.path.join('./checkpoints',
                                  args.data_name + '-' + args.model_name
                                  + '-' + str(datetime.date.today()))
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    set_logger(os.path.join(args.save_path, 'log.txt'))

    if args.data_name == "cxqc":
        # set_seed(args.seed)
        args.task_type = 'classification'
        args.task_type_detail = 'sentence_pair'
        args.batch_size = 64
        # args.crf_lr = 2e-5
        # args.num_layers = 2
        args.use_lstm = False
        args.use_crf = False
        args.train_epochs = 50

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        logger.info(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

    if args.data_name == "fxjg":
        # set_seed(args.seed)
        args.task_type = 'sequence'
        args.task_type_detail = 'sequence_labeling'
        args.batch_size = 32
        # args.crf_lr = 2e-5
        args.num_layers = 2
        args.train_epochs = 10

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        logger.info(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

    if args.data_name == "ssws":
        # set_seed(args.seed)
        args.task_type = 'sequence'
        args.task_type_detail = 'sequence_labeling'
        args.batch_size = 16
        # args.crf_lr = 2e-5
        args.num_layers = 1
        args.train_epochs = 5
        args.use_crf = True
        args.use_lstm = False

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        logger.info(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = NerDataset(train_features)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = NerDataset(dev_features)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                num_workers=0)

    if args.data_name == "yjyc":
        # set_seed(args.seed)
        args.task_type = 'sequence'
        args.task_type_detail = 'sequence_labeling'
        args.batch_size = 1
        # args.crf_lr = 2e-5
        args.train_epochs = 20
        args.use_gp = True

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        logger.info(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = GPDataset(train_features, args)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=gp_collate_fn,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = GPDataset(dev_features, args)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                collate_fn=gp_collate_fn,
                                num_workers=0)

    if args.data_name == "ske":
        # set_seed(args.seed)
        args.task_type = 'relationship'
        args.batch_size = 64
        # args.crf_lr = 2e-5
        args.train_epochs = 10
        args.use_gp = True
        args.use_efficient_globalpointer = True
        args.use_advert_train = False
        args.max_seq_len = 150

        with open(os.path.join(args.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        label2id = {}
        id2label = {}
        for k, v in enumerate(label_list):
            label2id[v] = k
            id2label[k] = v
        args.num_tags = len(label_list)

        logger.info(args)
        save_json(args.save_path, vars(args), 'args')

        with open(os.path.join(args.data_dir, 'train_data.pkl'), 'rb') as f:
            train_features = pickle.load(f)
        train_dataset = GPDataset(train_features, args)
        train_sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=gp_collate_fn_re,
                                  shuffle=False,
                                  num_workers=0)
        with open(os.path.join(args.data_dir, 'dev_data.pkl'), 'rb') as f:
            dev_features = pickle.load(f)
        dev_dataset = GPDataset(dev_features, args)
        dev_sampler = SequentialSampler(dev_dataset)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                collate_fn=gp_collate_fn_re,
                                shuffle=False,
                                num_workers=0)

    if args.task_type == 'classification':
        bertForClassify = TrainClassification(args, train_loader, dev_loader, label_list, logger)
        bertForClassify.train()

    if args.task_type == 'sequence':
        if args.use_gp:
            GpForSequence = TrainGlobalPointerNer(args, train_loader, dev_loader, label_list, logger)
            GpForSequence.train()
        else:
            bertForSequence = TrainSequenceLabeling(args, train_loader, dev_loader, label_list, logger)
            bertForSequence.train()

    if args.task_type == 'relationship':
        GpForSequence = TrainGlobalPointerRe(args, train_loader, dev_loader, label_list, logger)
        GpForSequence.train()
