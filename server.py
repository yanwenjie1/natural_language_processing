# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/4
@Time    : 16:31
@File    : server.py
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
from gevent import pywsgi
from transformers import BertTokenizer
from utils.functions import load_model_and_parallel, get_entity_bieos
from utils.models import Classification, SequenceLabeling
from utils.tokenizers import sentence_pair_encode_list, sentence_encode


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
    logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), 'dev')
    if args.use_crf:  # args.use_crf
        output = logits
        output = model.crf.decode(output.cpu(), mask=attention_masks)
    else:
        output = logits.detach().cpu().numpy()
        output = np.argmax(output, axis=-1)
    # 批量输入的时候 就不能只取1了
    results = []
    for y_pre in output:
        if args.task_type == 'classification':
            results.append(id2label[y_pre])
        if args.task_type == 'sequence':
            entities = get_entity_bieos([id2label[i] for i in y_pre][1:-1])
            results.append(copy.deepcopy(entities))
    return results


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = 'ssws-chinese-albert-tiny-2023-01-12'  # 24这个可以！ 27也可以 44亦可  64 ok
data_path = './data/ssws'
args_path = './checkpoints/{}/args.json'.format(model_name)
model_path = './checkpoints/{}/model_best.pt'.format(model_name)
port = 12000
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(os.path.join(data_path, 'labels.json'), 'r', encoding='utf-8') as f:
    label_list = json.load(f)
id2label = {k: v for k, v in enumerate(label_list)}
args = Dict2Class(**tmp_args)
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))

if args.task_type == 'classification':
    model, device = load_model_and_parallel(Classification(args), args.gpu_ids, model_path)
elif args.task_type == 'sequence':
    model, device = load_model_and_parallel(SequenceLabeling(args), args.gpu_ids, model_path)
    model.crf.cpu()
model.eval()
app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def prediction():
    # noinspection PyBroadException
    try:
        msgs = request.get_data()
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
                    results.append([[msg[i][item[1]:item[2]], item[0], item[1], item[2]] for item in result])
            elif args.task_type == 'classification':
                results.extend(partOfResults)
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
