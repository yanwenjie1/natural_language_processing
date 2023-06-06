# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/5/9
@Time    : 13:12
@File    : server_confindence.py
@Function: XX
@Other: XX
"""

import copy
import json
import os
import torch
import time
import socket
import numpy as np
import tqdm
from flask import Flask, request
from torchcrf import CRF
from gevent import pywsgi
from transformers import BertTokenizer
from utils.functions import load_model_and_parallel, get_entity_bieos, get_entity_gp, get_entity_gp_re, \
    get_entity_gp_re_confidence
from utils.models import Classification, SequenceLabeling, GlobalPointerNer, GlobalPointerRe
from utils.tokenizers import sentence_pair_encode_list, sentence_encode
from typing import List, Optional


def torch_env():
    """
    测试torch环境是否正确
    :return:
    """
    import torch.backends.cudnn

    print('torch版本:', torch.__version__)  # 查看torch版本
    print('cuda版本:', torch.version.cuda)  # 查看cuda版本
    print('cuda是否可用:', torch.cuda.is_available())  # 查看cuda是否可用
    print('可行的GPU数目:', torch.cuda.device_count())  # 查看可行的GPU数目 1 表示只有一个卡
    print('cudnn版本:', torch.backends.cudnn.version())  # 查看cudnn版本
    print('输出当前设备:', torch.cuda.current_device())  # 输出当前设备（我只有一个GPU为0）
    print('0卡名称:', torch.cuda.get_device_name(0))  # 获取0卡信息
    print('0卡地址:', torch.cuda.device(0))  # <torch.cuda.device object at 0x7fdfb60aa588>
    x = torch.rand(3, 2)
    print(x)  # 输出一个3 x 2 的tenor(张量)


def get_ip_config():
    """
    ip获取
    :return:
    """
    myIp = [item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None) if ':' not in item[4][0]][0]
    return myIp


def encode(texts):
    """

    :param texts: list of str
    :return:
    """
    assert type(texts) == type([1, 2])
    # 一个比较简单的实现是 按照list直接喂给 tokenizer
    if args.task_type_detail == 'sentence_pair':
        text1 = []
        text2 = []
        for text in texts:
            a, b = text.split('@@')
            text1.append(a)
            text2.append(b)
        token_ids, attention_masks, token_type_ids = sentence_pair_encode_list(text1, text2,
                                                                               args.max_seq_len, tokenizer, 'Predict')
    else:
        # 本部分的实现是 每个都独立生成 然后拼一起 因为按字划分tokens和batch_encode_plus不可兼得
        token_ids = None
        attention_masks = None
        token_type_ids = None
        max_len = max([len(i) for i in texts]) + 2
        max_len = min(max_len, 512)
        for text in texts:
            token_id, attention_mask, token_type_id = sentence_encode(list(text), max_len + 2, tokenizer)
            if token_ids is None:
                token_ids = token_id
                attention_masks = attention_mask
                token_type_ids = token_type_id
            else:
                token_ids = torch.cat((token_ids, token_id), dim=0)
                attention_masks = torch.cat((attention_masks, attention_mask), dim=0)
                token_type_ids = torch.cat((token_type_ids, token_type_id), dim=0)

    return token_ids, attention_masks, token_type_ids


def decode(token_ids, attention_masks, token_type_ids):
    """

    :param token_ids:
    :param attention_masks:
    :param token_type_ids:
    :return:
    """
    results = []

    if args.task_type == 'classification':
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        output = logits.detach().cpu()
        output = torch.softmax(output, dim=-1)
        max_value = np.amax(output.numpy(), axis=-1)
        max_indices = np.argmax(output.numpy(), axis=-1)
        for y_value, y_pre in zip(max_value, max_indices):
            label = id2label[y_pre]
            results.append((label, float(y_value)))

    if args.task_type == 'sequence' and args.use_gp is True:
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        for y_pre in logits.detach():
            entities = get_entity_gp(y_pre, id2label)
            for i in range(len(entities)):
                confidence = torch.sigmoid(
                    y_pre[label2id[entities[i][0]], entities[i][1], entities[i][2]] / 2).detach().cpu().numpy()
                entities[i] = (entities[i][0], entities[i][1] - 1, entities[i][2] - 1, float(confidence))
            results.append(copy.deepcopy(entities))

    if args.task_type == 'sequence' and args.use_gp is False:
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), 'dev')
        if args.use_crf:
            output1 = logits.detach().cpu()
            output = model.crf.decode(output1, mask=attention_masks)
        else:
            output1 = logits.detach().cpu()
            output1 = torch.softmax(output1, dim=-1)
            output = torch.argmax(output1, axis=-1)
        index = 0
        for y_pre in output:
            entities = get_entity_bieos([id2label[i] for i in y_pre][1:-1])
            if args.use_crf:
                for i in range(len(entities)):
                    start = entities[i][1]
                    end = entities[i][2] + 2
                    tags = torch.tensor(y_pre[start: end]).unsqueeze(0).long()
                    confidence = myCrf.confidence(output1[index, start:end, :].unsqueeze(0),
                                                  tags).detach().cpu().numpy()
                    entities[i] = entities[i] + (float(confidence[0]),)
            else:
                for i in range(len(entities)):
                    start = entities[i][1]
                    end = entities[i][2] + 2
                    tags = output[index, start:end]
                    confidence = torch.mean(output1[index, start:end, tags]).detach().cpu().numpy()
                    entities[i] = entities[i] + (float(confidence[0]),)
            results.append(copy.deepcopy(entities))
            index = index + 1

    if args.task_type == 'relationship':
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        output = [i.detach().cpu() for i in logits]
        for index in range(output[0].shape[0]):  # 循环 batch
            entities = get_entity_gp_re_confidence([output[0][index],
                                                    output[1][index],
                                                    output[2][index]],
                                                   attention_masks[index].detach(),
                                                   id2label)
            results.append(copy.deepcopy(entities))
        return results

    return results


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class MyCRF(CRF):
    def __init__(self, crf: CRF):
        super().__init__(args.num_tags, batch_first=True)
        self.transitions = crf.transitions
        self.start_transitions = crf.start_transitions
        self.end_transitions = crf.end_transitions
        # torch.nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        # torch.nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def confidence(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None):
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        return torch.exp(numerator - denominator)


torch_env()
model_name = './checkpoints/cxqc-chinese-albert-tiny-2023-06-06'
args_path = os.path.join(model_name, 'args.json')
model_path = os.path.join(model_name, 'model_best.pt')
labels_path = os.path.join(model_name, 'labels.json')

port = 12003
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(labels_path, 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
label2id = {v: k for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))

if args.task_type == 'classification':
    model, device = load_model_and_parallel(Classification(args), args.gpu_ids, model_path)
elif args.task_type == 'sequence':
    if args.use_gp == True:
        model, device = load_model_and_parallel(GlobalPointerNer(args), args.gpu_ids, model_path)
    else:
        model, device = load_model_and_parallel(SequenceLabeling(args), args.gpu_ids, model_path)
        model.crf.cpu()
        myCrf = MyCRF(model.crf)
elif args.task_type == 'relationship':
    model, device = load_model_and_parallel(GlobalPointerRe(args), args.gpu_ids, model_path)
    pass
model.eval()
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
        # msgs = request.get_json("content")
        msgs = msgs.decode('utf-8')
        # print(msg)
        msgs = json.loads(msgs)
        assert type(msgs) == type([1, 2])
        results = []
        count = 10  # 控制小batch推理
        for index in range(len(msgs) // count + 1):
            msg = msgs[index * count: index * count + count]
            if len(msg) == 0:
                continue
            token_ids, attention_masks, token_type_ids = encode(msg)
            partOfResults = decode(token_ids, attention_masks, token_type_ids)
            if args.task_type == 'sequence':
                for i, result in enumerate(partOfResults):
                    results.append(
                        [[msg[i][item[1]:item[2]], item[0], int(item[1]), int(item[2]), item[3]] for item in result])
            elif args.task_type == 'classification':
                results.extend(partOfResults)
            elif args.task_type == 'relationship':
                for i, result in enumerate(partOfResults):  # result:[(s_h, s_t, p, o_h, o_t, confi)]
                    results.append([[(msg[i][item[0] - 1: item[1] - 1], int(item[0] - 1), int(item[1] - 1)),
                                     id2label[item[2]],
                                     (msg[i][item[3] - 1: item[4] - 1], int(item[3] - 1), int(item[4] - 1)),
                                     item[5]] for item in
                                    result])
        res = json.dumps(results, ensure_ascii=False)
        # torch.cuda.empty_cache()
        return res
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=port, threaded=False, debug=False)
    server = pywsgi.WSGIServer(('0.0.0.0', port), app)
    print("Starting server in python...")
    print('Service Address : http://' + get_ip_config() + ':' + str(port))
    server.serve_forever()
    print("done!")
    # app.run(host=hostname, port=port, debug=debug)  注释以前的代码
    # manager.run()  # 非开发者模式
