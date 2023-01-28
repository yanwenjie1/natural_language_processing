# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/16
@Time    : 8:48
@File    : preprocess.py
@Function: XX
@Other: XX
"""
import copy
import json
import os
import os
import json
from tqdm import tqdm
import pickle
from transformers import BertTokenizer
import config
import random
from utils.models import BaseFeature
from utils.tokenizers import sentence_encode
import config


def search(pattern, sequence):
    """
    从sequence中寻找子串pattern 如果找到，返回第一个下标；否则返回-1。
    :param pattern:
    :param sequence:
    :return:
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def load_data(filename):
    results = []
    label_list = []
    with open(filename, encoding='utf-8') as file:
        contents = file.readlines()
        for content in contents:
            content = json.loads(content)
            spo_list = content['spo_list']
            for spo in spo_list:
                predicate = spo['predicate']
                if predicate not in label_list:
                    label_list.append(predicate)
    label_list.sort()

    print('labels_all：', label_list)
    print('labels_all_len：', len(label_list))
    with open(os.path.join(args.data_dir, 'labels.json'), 'w', encoding='utf-8') as file:
        file.write(json.dumps(label_list, ensure_ascii=False))

    tag2id = {}
    id2tag = {}
    for k, v in enumerate(label_list):
        tag2id[v] = k
        id2tag[k] = v

    for content in contents:
        content = json.loads(content)

        text = content['text']
        tokens = [i for i in text]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        spo_list = content['spo_list']
        labels = []  # [subject, predicate, object]

        if len(tokens) > args.max_seq_len:
            continue

        for spo in spo_list:
            subject = spo['subject']
            object = spo['object']
            predicate = spo['predicate']
            s = [i for i in subject]
            o = [i for i in object]
            p = tag2id[predicate]

            s_idx = search(s, tokens)  # 主体的头
            o_idx = search(o, tokens)  # 客体的头
            # 这里为了适应别人的语料，如果我们自己的语料有索引，就不用寻找了
            if s_idx != -1 and o_idx != -1:
                labels.append((s_idx, s_idx + len(s), p, o_idx, o_idx + len(o)))

        results.append((tokens, labels))  # 只取了客体 关系 主体
    return results


def convert_examples_to_features(examples, tokenizer: BertTokenizer):
    features = []
    for (text, labels) in tqdm(examples):
        text = text[1:-1]
        assert 0 < len(text) <= args.max_seq_len, 'len(text):' + str(len(text))
        token_ids, attention_masks, token_type_ids = sentence_encode(text, args.max_seq_len, tokenizer)
        feature = BaseFeature(
            token_ids=token_ids.squeeze(),
            attention_masks=attention_masks.squeeze(),
            token_type_ids=token_type_ids.squeeze(),
            labels=labels,  # labels: list of 5元组
        )
        features.append(feature)

    return features


if __name__ == '__main__':
    args = config.Args().get_parser()
    #  调整配置
    args.task_type = 'relationship'
    args.use_gp = True
    args.data_dir = os.getcwd()
    args.max_seq_len = 150

    my_tokenizer = BertTokenizer(os.path.join('../../' + args.bert_dir, 'vocab.txt'))
    train_data = load_data(os.path.join(args.data_dir, 'train_data.json'))
    dev_data = load_data(os.path.join(args.data_dir, 'dev_data.json'))

    random.shuffle(train_data)  # 打乱数据集

    train_data = convert_examples_to_features(train_data, my_tokenizer)
    dev_data = convert_examples_to_features(dev_data, my_tokenizer)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('train_data')), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(args.data_dir, '{}.pkl'.format('dev_data')), 'wb') as f:
        pickle.dump(dev_data, f)
