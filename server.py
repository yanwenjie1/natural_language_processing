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
from torchcrf import CRF
from gevent import pywsgi
from transformers import BertTokenizer, AutoTokenizer
from utils.functions import load_model_and_parallel, get_entity_bieos, get_entity_gp_re, get_entity_gp, get_span
from utils.models import Classification, SequenceLabeling, GlobalPointerRe, GlobalPointerNer, MT54Generation, UIE4Ner
from utils.tokenizers import sentence_pair_encode_list, sentence_encode, sentence_pair_encode_plus
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
            token_id, attention_mask, token_type_id = sentence_encode(list(text), max_len, tokenizer)
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
    # 批量输入的时候 就不能只取1了
    results = []

    if args.task_type == 'classification':
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        output = logits.detach().cpu().numpy()
        output = np.argmax(output, axis=-1)
        for y_pre in output:
            results.append(id2label[y_pre])

    if args.task_type == 'sequence' and args.use_gp is True:
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        for y_pre in logits:
            entities = get_entity_gp(y_pre, id2label)
            results.append(copy.deepcopy(entities))

    if args.task_type == 'sequence' and args.use_gp is False:
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device), 'dev')
        if args.use_crf:
            output = logits
            output = model.crf.decode(output.cpu(), mask=attention_masks)
        else:
            output = logits.detach().cpu().numpy()
            output = np.argmax(output, axis=-1)
        for y_pre in output:
            entities = get_entity_bieos([id2label[i] for i in y_pre][1:-1])
            results.append(copy.deepcopy(entities))

    if args.task_type == 'relationship':
        logits = model(token_ids.to(device), attention_masks.to(device), token_type_ids.to(device))
        output = [i.detach().cpu() for i in logits]
        for index in range(output[0].shape[0]):
            entities = get_entity_gp_re([output[0][index],
                                         output[1][index],
                                         output[2][index]],
                                        attention_masks[index].detach(),
                                        id2label)
            results.append(copy.deepcopy(entities))
        return results

    return results


def uie_encode_ner(prompts: list, content: str) -> dict:
    result = []
    for prompt in prompts:
        max_len = args.max_seq_len - 3 - len(prompt)
        if len(prompt) > max(args.max_seq_len * 0.25, 128):
            continue
        i = 0
        while i < len(content):
            token_ids, attention_masks, token_type_ids = sentence_pair_encode_plus(prompt,
                                                                                   content[i: i + max_len],
                                                                                   args.max_seq_len, tokenizer)
            result.append({'token_ids': token_ids,
                           'attention_masks': attention_masks,
                           'token_type_ids': token_type_ids,
                           'prompt': prompt,
                           'content': content[i: i + max_len],
                           'start_id': i})
            i += max_len
    return result


def get_uie_result(content: str, parallel=16) -> dict:
    """
    model tokenizer label_list
    :param content:
    :param parallel:
    :return:
    """
    # 先是ner的encode
    label_ner = [i for i in label_list if type(i) == type('str')]
    label_relation = [i for i in label_list if type(i) == type({})][0]

    ner_encode_result = uie_encode_ner(label_ner, content)
    # 然后是ner的decode
    result_all = {k: [] for k in label_ner}
    start_ids_all = [[] for _ in ner_encode_result]
    end_ids_all = [[] for _ in ner_encode_result]
    index = 0
    while index < len(ner_encode_result):
        ner_encode_result_part = ner_encode_result[index: index + parallel]
        token_ids = [i['token_ids'].squeeze().numpy() for i in ner_encode_result_part]
        attention_masks = [i['attention_masks'].squeeze().numpy() for i in ner_encode_result_part]
        token_type_ids = [i['token_type_ids'].squeeze().numpy() for i in ner_encode_result_part]
        token_ids = torch.tensor(np.array(token_ids), dtype=torch.long).to(device)
        attention_masks = torch.tensor(np.array(attention_masks), dtype=torch.uint8).to(device)
        token_type_ids = torch.tensor(np.array(token_type_ids), dtype=torch.long).to(device)
        output_sp, output_ep = model(token_ids, attention_masks, token_type_ids)
        for _index, location in torch.nonzero(output_sp > 0.5):
            start_ids_all[_index.item() + index].append(location.item())
        for _index, location in torch.nonzero(output_ep > 0.5):
            end_ids_all[_index.item() + index].append(location.item())
        index += parallel
    # 下面是解码
    for i, (start_ids, end_ids) in enumerate(zip(start_ids_all, end_ids_all)):
        label_set = get_span(list(start_ids), list(end_ids))
        for (start, end) in label_set:
            prompt = ner_encode_result[i]['prompt']
            new_ner_result = {
                    'text': ner_encode_result[i]['content'][start - len(prompt) - 2: end - len(prompt) - 2],
                    'start': ner_encode_result[i]['start_id'] + start - len(prompt) - 2,
                    'end': ner_encode_result[i]['start_id'] + end - len(prompt) - 2
                }
            assert new_ner_result['text'] == content[new_ner_result['start']: new_ner_result['end']]
            result_all[prompt].append(new_ner_result)

    # 然后是relation的decode
    for subject_label in label_relation:
        for subject_result in result_all[subject_label]:
            subject = subject_result['text']
            object_labels = label_relation[subject_label]
            relation_encode_result = uie_encode_ner([subject + "的" + i for i in object_labels], content)
            relation_start_ids_all = [[] for _ in relation_encode_result]
            relation_end_ids_all = [[] for _ in relation_encode_result]
            index = 0
            while index < len(relation_encode_result):
                relation_encode_result_part = relation_encode_result[index: index + parallel]
                token_ids = [i['token_ids'].squeeze().numpy() for i in relation_encode_result_part]
                attention_masks = [i['attention_masks'].squeeze().numpy() for i in relation_encode_result_part]
                token_type_ids = [i['token_type_ids'].squeeze().numpy() for i in relation_encode_result_part]
                token_ids = torch.tensor(np.array(token_ids), dtype=torch.long).to(device)
                attention_masks = torch.tensor(np.array(attention_masks), dtype=torch.uint8).to(device)
                token_type_ids = torch.tensor(np.array(token_type_ids), dtype=torch.long).to(device)
                output_sp, output_ep = model(token_ids, attention_masks, token_type_ids)
                for _index, location in torch.nonzero(output_sp > 0.5):
                    relation_start_ids_all[_index.item() + index].append(location.item())
                for _index, location in torch.nonzero(output_ep > 0.5):
                    relation_end_ids_all[_index.item() + index].append(location.item())
                index += parallel

            for i, (start_ids, end_ids) in enumerate(zip(relation_start_ids_all, relation_end_ids_all)):
                label_set = get_span(list(start_ids), list(end_ids))
                for (start, end) in label_set:
                    prompt = relation_encode_result[i]['prompt']
                    prompt_true = prompt.replace(subject + "的", '')
                    if 'relations' not in subject_result.keys():
                        subject_result['relations'] = {}
                    if prompt_true not in subject_result['relations']:
                        subject_result['relations'][prompt_true] = []
                    new_ner_result = {
                            'text': relation_encode_result[i]['content'][start - len(prompt) - 2: end - len(prompt) - 2],
                            'start': relation_encode_result[i]['start_id'] + start - len(prompt) - 2,
                            'end': relation_encode_result[i]['start_id'] + end - len(prompt) - 2}
                    assert new_ner_result['text'] == content[new_ner_result['start']: new_ner_result['end']]
                    subject_result['relations'][prompt_true].append(new_ner_result)

    return result_all


class Dict2Class:
    def __init__(self, **entries):
        self.__dict__.update(entries)


torch_env()
model_name = './checkpoints/CemeteryFundUIE-chinese-uie-base-2023-06-29'
args_path = os.path.join(model_name, 'args.json')
model_path = os.path.join(model_name, 'model_best.pt')
labels_path = os.path.join(model_name, 'labels.json')

port = 12005
with open(args_path, "r", encoding="utf-8") as fp:
    tmp_args = json.load(fp)
with open(labels_path, 'r', encoding='utf-8') as f:
    label_list = json.load(f)

args = Dict2Class(**tmp_args)
if args.task_type_detail != 'uie':
    id2label = {k: v for k, v in enumerate(label_list)}
tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))

if args.task_type == 'classification':
    model, device = load_model_and_parallel(Classification(args), args.gpu_ids, model_path)
elif args.task_type == 'sequence':
    if args.task_type_detail != 'uie':
        if args.use_gp == True:
            model, device = load_model_and_parallel(GlobalPointerNer(args), args.gpu_ids, model_path)
        else:
            model, device = load_model_and_parallel(SequenceLabeling(args), args.gpu_ids, model_path)
            model.crf.cpu()
    else:
        model, device = load_model_and_parallel(UIE4Ner(args), args.gpu_ids, model_path)
elif args.task_type == 'relationship':
    model, device = load_model_and_parallel(GlobalPointerRe(args), args.gpu_ids, model_path)
elif args.task_type == 'generation':
    model, device = load_model_and_parallel(MT54Generation(args), args.gpu_ids, model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir)
    pass
model.eval()
for name, param in model.named_parameters():
    param.requires_grad = False
app = Flask(__name__)
torch.set_float32_matmul_precision('high')


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
        count = 5  # 控制小batch推理
        with torch.no_grad():
            if args.task_type == 'generation':
                for input in msgs:
                    input_token = tokenizer(input + '[SEP]',
                                            add_special_tokens=False,
                                            return_tensors="pt")
                    outputs = model.generate(
                        max_length=args.output_max_len,
                        eos_token_id=tokenizer.sep_token_id,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        input_ids=input_token.input_ids.to(device),
                        attention_mask=input_token.attention_mask.to(device))
                    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    one_result = [item.replace(' ', '') for item in outputs][0]
                    results.append(one_result)
            elif args.task_type_detail == 'uie':
                for input in msgs:
                    results.append(get_uie_result(input))
                    pass
            else:
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
                    elif args.task_type == 'relationship':
                        for i, result in enumerate(partOfResults):  # result:[(s_h, s_t, p, o_h, o_t)]
                            results.append([[(msg[i][item[0] - 1: item[1] - 1], int(item[0] - 1), int(item[1] - 1)),
                                             id2label[item[2]],
                                             (msg[i][item[3] - 1: item[4] - 1], int(item[3] - 1), int(item[4] - 1))] for
                                            item in result])
                            pass
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
