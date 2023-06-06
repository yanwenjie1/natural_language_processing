# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/6
@Time    : 10:11
@File    : functions.py
@Function: XX
@Other: XX
"""
import copy
import os
import torch
import random
import numpy as np
import logging
import json
import math

import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torchcontrib.optim import SWA
from torch.nn.utils.rnn import pad_sequence


def set_seed(seed=2022):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # 固定随机种子 CPU
    torch.cuda.manual_seed(seed)  # 固定随机种子 当前GPU
    torch.cuda.manual_seed_all(seed)  # 固定随机种子 所有GPU
    np.random.seed(seed)  # 保证后续使用random时 产生固定的随机数
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 固定网络结构
    torch.backends.cudnn.benchmark = False  # GPU和网络结构固定时 可以为True 自动寻找更优
    torch.backends.cudnn.enabled = False


def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    """
    加载模型 & 放置到 GPU 中（单卡 / 多卡）
    :param model: 实例化的模型对象
    :param gpu_ids: GPU的ID
    :param ckpt_path: 模型加载路径
    :param strict: 是否严格加载
    :return:
    """
    gpu_ids = gpu_ids.split(',')
    # set to device to the first cuda
    device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
    # 用strict=False进行加载模型，则 能塞则塞 不能塞则丢。
    # strict=True时一旦有key不匹配则出错，如果设置strict=False，则直接忽略不匹配的key，对于匹配的key则进行正常的赋值。
    if ckpt_path is not None:
        print('Load ckpt from {}'.format(ckpt_path))
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)

    if len(gpu_ids) > 1:
        print('Use multi gpus in: {}'.format(gpu_ids))
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        print('Use single gpu in: {}'.format(gpu_ids))
        model = model.to(device)

    return model, device


def build_optimizer_and_scheduler(args, model, t_total):
    """
    不同的模块使用不同的学习率
    :param args:
    :param model:
    :param t_total:
    :return:
    """
    # hasattr 判断对象是否包含对应的属性
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_lr},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )
    # opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)  #  后面尝试swa的效果

    return optimizer, scheduler


def save_model(args, model, global_step, log):
    """
    保存验证集效果最好的那个模型
    :param log:
    :param args:
    :param model:
    :param global_step:
    :return:
    """
    # take care of model distributed / parallel training  小心分布式训练or并行训练
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    # log.info('Saving model checkpoint to {}'.format(output_dir))
    # torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_{}.pt'.format(global_step)))
    torch.save(model_to_save.state_dict(), os.path.join(args.save_path, 'model_best.pt'))


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_json(data_dir, data, desc):
    """保存数据为json"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_entity_bieos(labels):
    """
    输入标签的List 输出每一个实体
    :param labels:
    :return:list of (label, start, end)
    """
    entities = []
    entity = []
    for i in range(len(labels)):
        if labels[i] == "O":
            entity = []
        elif labels[i][0] == "B":
            entity = [labels[i][2:], i]
        elif labels[i][0] == "I" and len(entity) == 2 and labels[i][2:] == entity[0]:
            continue
        elif labels[i][0] == "E" and len(entity) == 2 and labels[i][2:] == entity[0]:
            entity.append(i + 1)
            entities.append(entity[:])
            entity = []
        elif labels[i][0] == "S" and labels[i] != "SEP":
            entity = [labels[i][2:], i, i + 1]
            entities.append(entity[:])
            entity = []
        else:
            entity = []
    return [tuple(i) for i in entities]


def get_entity_gp(tensors, id2label):
    """

    :param tensors:
    :param id2label: dict of key:id, value:label
    :return: list of Tuple: (label, start, end)
    """
    assert tensors.size(0) == len(id2label)

    entities = []
    for key, value in id2label.items():
        logits = tensors[key, :, :]  # 可以考虑用mask矩阵
        for start, end in zip(*np.where(logits.cpu().numpy() > 0)):
            entities.append([value, start, end])
    return [tuple(i) for i in entities]


def get_entity_gp_re(tensors, attention_masks, id2label):
    """

    :param tensors: list of tensor [entities, heads, tails]
    :param attention_masks:
    :param id2label: dict of key:id, value:label
    :return: list of Tuple: (subject_head, subject_tail, pre_ids, object_head, object_tail)
    """
    # tensors: len is 3
    # tnesors[0].shape: [2, max_len, max_len] 2: subject and object
    # tnesors[1].shape: [num_tags, max_len, max_len]
    # tnesors[2].shape: [num_tags, max_len, max_len]
    assert tensors[1].size(0) == len(id2label)
    assert tensors[2].size(0) == len(id2label)
    tokens_len = sum(attention_masks)
    subject_ids = []
    object_ids = []
    results = []

    subject_entity = tensors[0][0, :tokens_len, :tokens_len].squeeze()
    object_entity = tensors[0][1, :tokens_len, :tokens_len].squeeze()
    head_output = tensors[1][:, :tokens_len, :tokens_len]
    tail_output = tensors[2][:, :tokens_len, :tokens_len]

    subject_entity = np.where(subject_entity.cpu().numpy() > 0)
    object_entity = np.where(object_entity.cpu().numpy() > 0)
    for m, n in zip(*subject_entity):
        subject_ids.append((m, n))
    for m, n in zip(*object_entity):
        object_ids.append((m, n))

    for sh, st in subject_ids:
        for oh, ot in object_ids:
            re1 = np.where(head_output[:, sh, oh].cpu().numpy() > 0)[0]
            re2 = np.where(tail_output[:, st, ot].cpu().numpy() > 0)[0]
            res = set(re1) & set(re2)
            for r in res:
                results.append((sh, st, r, oh, ot))

    return results


def get_entity_gp_re_confidence(tensors, attention_masks, id2label):
    """

    :param tensors: list of tensor [entities, heads, tails]
    :param attention_masks:
    :param id2label: dict of key:id, value:label
    :return: list of Tuple: (subject_head, subject_tail, pre_ids, object_head, object_tail)
    """
    # tensors: len is 3
    # tnesors[0].shape: [2, max_len, max_len] 2: subject and object
    # tnesors[1].shape: [num_tags, max_len, max_len]
    # tnesors[2].shape: [num_tags, max_len, max_len]
    assert tensors[1].size(0) == len(id2label)
    assert tensors[2].size(0) == len(id2label)
    tokens_len = sum(attention_masks)
    subject_ids = []
    object_ids = []
    results = []

    subject_entity = tensors[0][0, :tokens_len, :tokens_len].squeeze()
    object_entity = tensors[0][1, :tokens_len, :tokens_len].squeeze()
    head_output = tensors[1][:, :tokens_len, :tokens_len]
    tail_output = tensors[2][:, :tokens_len, :tokens_len]

    subject_entity_1 = np.where(subject_entity.cpu().numpy() > 0)
    object_entity_1 = np.where(object_entity.cpu().numpy() > 0)
    for m, n in zip(*subject_entity_1):
        subject_ids.append((m, n, torch.sigmoid(subject_entity[m, n] / 1)))
    for m, n in zip(*object_entity_1):
        object_ids.append((m, n, torch.sigmoid(object_entity[m, n] / 1)))

    for sh, st, s_confi in subject_ids:
        for oh, ot, o_confi in object_ids:
            re1 = np.where(head_output[:, sh, oh].cpu().numpy() > 0)[0]
            re2 = np.where(tail_output[:, st, ot].cpu().numpy() > 0)[0]
            res = set(re1) & set(re2)

            for r in res:
                r_confi = torch.sigmoid((head_output[r, sh, oh] + tail_output[r, sh, oh]) / 2)
                confi = (s_confi + o_confi + r_confi) / 3
                confi = float(confi.detach().cpu().numpy())
                results.append((sh, st, r, oh, ot, confi))

    return results


def gp_collate_fn(batch):
    """

    :param batch: list, len(batch) == batch_size
    :return:
    """
    token_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for item in batch:
        token_ids.append(item['token_ids'].numpy())
        attention_masks.append(item['attention_masks'].numpy())
        token_type_ids.append(item['token_type_ids'].numpy())
        labels.append(item['labels'])
        # labels.append(gp_entity_to_label(item['labels'], item['args']))

    token_ids = np.array(token_ids, dtype=float)
    token_ids = torch.tensor(token_ids, dtype=torch.long, device='cuda:0')
    attention_masks = np.array(attention_masks, dtype=float)
    attention_masks = torch.tensor(attention_masks, dtype=torch.uint8, device='cuda:0')
    token_type_ids = np.array(token_type_ids, dtype=float)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device='cuda:0')
    labels = gp_entity_to_label(labels, batch[0]['args'])
    # labels = torch.tensor(np.array(labels), dtype=torch.long, device='cuda:0')
    return {'token_ids': token_ids,
            'attention_masks': attention_masks,
            'token_type_ids': token_type_ids,
            'labels': labels}


def gp_collate_fn_re(batch):
    """

    :param batch: list, len(batch) == batch_size
    :return:
    """
    token_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for item in batch:
        token_ids.append(item['token_ids'].numpy())
        attention_masks.append(item['attention_masks'].numpy())
        token_type_ids.append(item['token_type_ids'].numpy())
        labels.append(item['labels'])
        # labels.append(gp_entity_to_label(item['labels'], item['args']))

    token_ids = torch.tensor(np.array(token_ids, dtype=float), dtype=torch.long, device='cuda:0')
    attention_masks = torch.tensor(np.array(attention_masks, dtype=float), dtype=torch.uint8, device='cuda:0')
    token_type_ids = torch.tensor(np.array(token_type_ids, dtype=float), dtype=torch.long, device='cuda:0')
    batch_entity_labels, batch_head_labels, batch_tail_labels = gp_entity_to_label_re(labels, batch[0]['args'])
    # batch_entity_labels_, batch_head_labels_, batch_tail_labels_ = gp_entity_to_label_re(labels, batch[0]['args'])
    # labels = torch.tensor(np.array(labels), dtype=torch.long, device='cuda:0')
    return {'token_ids': token_ids,
            'attention_masks': attention_masks,
            'token_type_ids': token_type_ids,
            'labels': [batch_entity_labels, batch_head_labels, batch_tail_labels],
            'callback': labels}


def gp_entity_to_label(entities, args):
    """
    注意CLS的问题
    :param entities:
    :param args:
    :return:
    """
    # self.labels = [example.labels for example in features]

    # labels = np.zeros((args.num_tags, args.max_seq_len, args.max_seq_len), dtype=int)
    # for label, start, end in entities:
    #     labels[label, start, end] = 1
    # return labels
    labels = torch.zeros((len(entities), args.num_tags, args.max_seq_len, args.max_seq_len),
                         device='cuda:0',
                         dtype=torch.long)
    for i, entity in enumerate(entities):
        for label, start, end in entity:
            labels[i, label, start + 1, end + 1] = 1
    return labels


def gp_entity_to_label_re(batches, args):
    """
    注意CLS的问题
    另外要注意关系抽取下，用的是稀疏版多标签交叉熵，所以每次都只传输正类所对应的的下标
    :param entities: list of list of 5元组
    :param args:
    :return: [batch_size, 2, max_entity_len, 2] [batch_size, num_tags, max_entity_len, 2] [batch_size, num_tags, max_entity_len, 2]
    """
    # 事实上 主客实体个数, 头指针个数, 尾指针个数 都是不一致的
    batch_entity_labels_ = []
    batch_head_labels_ = []
    batch_tail_labels_ = []
    for batch in batches:
        entity_labels = [set() for _ in range(2)]  # [主体， 客体]
        head_labels = [set() for _ in range(args.num_tags)]  # 每个关系中主体和客体的头
        tail_labels = [set() for _ in range(args.num_tags)]  # 每个关系中主体和客体的尾
        for sh, st, p, oh, ot in batch:
            entity_labels[0].add((sh, st))
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))
            tail_labels[p].add((st, ot))
        batch_entity_labels_.append(entity_labels)
        batch_head_labels_.append(head_labels)
        batch_tail_labels_.append(tail_labels)

    # 下面计数
    max_entity_len = max([max([len(j) for j in i]) for i in batch_entity_labels_])
    max_head_len = max([max([len(j) for j in i]) for i in batch_head_labels_])
    max_tail_len = max([max([len(j) for j in i]) for i in batch_tail_labels_])

    max_entity_len = max(max_entity_len, 1)
    max_head_len = max(max_head_len, 1)
    max_tail_len = max(max_tail_len, 1)

    # assert max_head_len == max_tail_len, 'max_head_len:' + str(max_head_len) + '  max_tail_len:' + str(max_tail_len)

    batch_entity_labels = torch.zeros((len(batches), 2, max_entity_len, 2), dtype=torch.float,
                                      device='cuda:0')
    batch_head_labels = torch.zeros((len(batches), args.num_tags, max_head_len, 2), dtype=torch.float,
                                    device='cuda:0')
    batch_tail_labels = torch.zeros((len(batches), args.num_tags, max_tail_len, 2), dtype=torch.float,
                                    device='cuda:0')

    for i in range(len(batches)):
        for p in range(2):  # p in (0, 1)
            entity_label = batch_entity_labels_[i][p]
            for j, (head, tail) in enumerate(entity_label):
                batch_entity_labels[i, p, j, 0] = head
                batch_entity_labels[i, p, j, 1] = tail

        for p in range(args.num_tags):
            head_label = batch_head_labels_[i][p]
            tail_label = batch_tail_labels_[i][p]
            for j, (s_head, o_head) in enumerate(head_label):
                batch_head_labels[i, p, j, 0] = s_head
                batch_head_labels[i, p, j, 1] = o_head

            for j, (s_tali, o_tali) in enumerate(tail_label):
                batch_tail_labels[i, p, j, 0] = s_tali
                batch_tail_labels[i, p, j, 1] = o_tali

    return batch_entity_labels, batch_head_labels, batch_tail_labels


def gp_entity_to_label_re_old(batches, args):
    """
    注意CLS的问题
    另外要注意关系抽取下，用的是稀疏版多标签交叉熵，所以每次都只传输正类所对应的的下标
    :param entities: list of list of 5元组
    :param args:
    :return: [batch_size, 2, max_entity_len, 2] [batch_size, num_tags, max_entity_len, 2] [batch_size, num_tags, max_entity_len, 2]
    """
    batch_entity_labels = []
    batch_head_labels = []
    batch_tail_labels = []

    for batch in batches:
        entity_labels = [set() for _ in range(2)]  # [主体， 客体]
        head_labels = [set() for _ in range(args.num_tags)]  # 每个关系中主体和客体的头
        tail_labels = [set() for _ in range(args.num_tags)]  # 每个关系中主体和客体的尾
        for sh, st, p, oh, ot in batch:
            entity_labels[0].add((sh, st))  # (sh, st) 元组
            entity_labels[1].add((oh, ot))
            head_labels[p].add((sh, oh))
            tail_labels[p].add((st, ot))
        for label in entity_labels + head_labels + tail_labels:
            if not label:  # 每个集合至少要有一个标签
                label.add((0, 0))  # 如果没有则用0填充

        entity_labels = sequence_padding([list(l) for l in entity_labels])  # [subject/object=2, 实体个数, 实体起终点]
        head_labels = sequence_padding(
            [list(l) for l in head_labels])  # [关系个数, 该关系下subject/object配对数, subject/object起点]
        tail_labels = sequence_padding(
            [list(l) for l in tail_labels])  # [关系个数, 该关系下subject/object配对数, subject/object终点]
        batch_head_labels.append(head_labels)
        batch_tail_labels.append(tail_labels)
        batch_entity_labels.append(entity_labels)
        # assert head_labels.shape == tail_labels.shape

    batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2), dtype=torch.float,
                                     device='cuda:0')
    batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2), dtype=torch.float,
                                     device='cuda:0')
    batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2), dtype=torch.float,
                                       device='cuda:0')
    return batch_entity_labels, batch_head_labels, batch_tail_labels


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度
    """
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    elif isinstance(inputs[0], torch.Tensor):
        assert mode == 'post', '"mode" argument must be "post" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''Returns: [seq_len, d_hid]
    '''
    position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2).float() * (-math.log(10000.0) / d_hid))
    embeddings_table = torch.zeros(n_position, d_hid)
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    return embeddings_table


