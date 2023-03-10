# -*- coding: utf-8 -*-
"""
@author: YanWJ
@Date    : 2023/1/11
@Time    : 8:09
@File    : tokenizers.py
@Function: XX
@Other: XX
"""
from transformers import BertTokenizer


def sentence_pair_encode(text1, text2, max_seq_len, tokenizer: BertTokenizer, model='Train'):
    """

    :param text1:
    :param text2:
    :param max_seq_len:
    :param tokenizer:
    :param model:
    :return:
    """
    text1 = text1[:max_seq_len // 2 - 2]
    text2 = text2[:max_seq_len // 2 - 2]
    if model == 'Predict':
        max_seq_len = len(text1) + len(text2) + 3
    word_ids = tokenizer.encode_plus(text1, text2,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation='longest_first',
                                     return_token_type_ids=True,
                                     return_attention_mask=True)
    token_ids = word_ids['input_ids']
    attention_masks = word_ids['attention_mask']
    token_type_ids = word_ids['token_type_ids']
    return token_ids, attention_masks, token_type_ids
    pass


def sentence_pair_encode_list(text1, text2, max_seq_len, tokenizer: BertTokenizer, model='Train'):
    """

    :param text1:
    :param text2:
    :param max_seq_len:
    :param tokenizer:
    :param model:
    :return:
    """
    text = []
    assert len(text1) == len(text2)
    for i in range(len(text1)):
        text.append((text1[i][:max_seq_len // 2 - 2], text2[i][:max_seq_len // 2 - 2]))
    if model == 'Predict':
        word_ids = tokenizer.batch_encode_plus(text,
                                               padding=True,
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
    else:
        word_ids = tokenizer.batch_encode_plus(text,
                                               max_length=max_seq_len,
                                               padding="max_length",
                                               truncation='longest_first',
                                               return_token_type_ids=True,
                                               return_attention_mask=True,
                                               return_tensors='pt')
    token_ids = word_ids['input_ids']
    attention_masks = word_ids['attention_mask']
    token_type_ids = word_ids['token_type_ids']
    return token_ids, attention_masks, token_type_ids


def sentence_encode(text, max_seq_len, tokenizer: BertTokenizer):
    """

    :param text:
    :param max_seq_len:
    :param tokenizer:
    :param model:
    :param text2:
    :return:
    """
    assert type(text) == type([1, 2])
    word_ids = tokenizer.encode_plus(text,
                                     max_length=max_seq_len,
                                     padding="max_length",
                                     truncation='longest_first',
                                     return_token_type_ids=True,
                                     return_attention_mask=True,
                                     return_tensors='pt')
    token_ids = word_ids['input_ids']
    attention_masks = word_ids['attention_mask'].byte()
    token_type_ids = word_ids['token_type_ids']
    for j, token in enumerate(text):
        if token == ' ':
            token_ids[0, j + 1] = 50
    return token_ids, attention_masks, token_type_ids

