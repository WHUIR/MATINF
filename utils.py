#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections
from typing import List

from rouge import Rouge
import torch
import torch.nn
import math
import numpy as np
import pandas as pd
import pickle
from dataset import Dataset

CLASS3_NAME_TO_INDEX = {
    '0-1岁': 0,
    '1-2岁': 1,
    '2-3岁': 2
}

CLASS18_NAME_TO_INDEX = {
    '动作发育': 0,
    '幼儿园': 1,
    '产褥期保健': 2,
    '婴幼常见病': 3,
    '家庭教育': 4,
    '未准父母': 5,
    '婴幼保健': 6,
    '婴幼期喂养': 7,
    '疫苗接种': 8,
    '腹泻': 9,
    '宝宝上火': 10,
    '婴幼心理': 11,
    '皮肤护理': 12,
    '流产和不孕': 13,
    '婴幼早教': 14,
    '儿童过敏': 15,
    '孕期保健': 16,
    '婴幼营养': 17
}


def pad_sents(sents, pad_token):
    """pad list of sentences according to the longest sent.
    """
    sents_padded = []
    max_len = max([len(sent) for sent in sents])
    for s in sents:
        if len(s) < max_len:
            s_len = len(s)
            sents_padded.append(s + (max_len - s_len) * [pad_token])
        else:
            sents_padded.append(s)
    return sents_padded


def build_embeddings(file_path, vocab):
    with open(file_path, encoding='UTF-8') as f:
        line = f.readline().strip().split(' ')
        size, dim = vocab.size(), int(line[1])
        weight_matrix = torch.randn((size, dim), dtype=torch.float)

        for line in f:
            line = line.rstrip().split(' ')
            if line[0] in vocab.word2id.keys():
                weight = list(map(float, line[-dim:]))
                weight = torch.tensor(weight, dtype=torch.float)
                weight_matrix[vocab.word2id[line[0]]] = torch.unsqueeze(weight, dim=0)

        return torch.nn.Embedding.from_pretrained(weight_matrix)


def read_data(file_path):
    """Read dataset file.
    """
    dataset_cls3 = Dataset()
    dataset_cls18 = Dataset()
    max_len = 256
    num_summ_qa = num_cls3 = num_cls18 = 0

    data_table = pd.read_csv(file_path, sep=',', encoding='UTF-8')

    for i in range(0, len(data_table)):
        question = str(data_table.iat[i, 1]).strip()
        description = str(data_table.iat[i, 2]).strip()
        answer = str(data_table.iat[i, 3]).strip()
        category = str(data_table.iat[i, 4]).strip()

        if len(description) > max_len or len(answer) > max_len:
            print('Too long: ', str(data_table.iat[i, 0]))
            continue

        num_summ_qa += 1
        if category in CLASS3_NAME_TO_INDEX:
            num_cls3 += 1
            dataset_cls3.add_data(question, description, answer, category)
        elif category in CLASS18_NAME_TO_INDEX:
            num_cls18 += 1
            dataset_cls18.add_data(question, description, answer, category)
        else:
            print('Unexpected category! id:{}'.format(data_table.iat[i, 0]))
            continue

    print('samples num for sum and qa:', num_summ_qa)
    print('samples num for cls3:', num_cls3)
    print('samples num for cls18:', num_cls18)

    return dataset_cls3, dataset_cls18


def cal_rouge(hyps:List[str],refs:List[str],avg:bool=False,ignore_empty:bool=False):
    """
    :param hyps: List of hyps, each hyp is a 'str' consists of a sequence of tokens separated by spaces.
    :param refs: List of refs, each ref is a 'str' consists of a sequence of tokens separated by spaces.
    :param avg: If scoring multiple sentences, 'avg' should be 'True'.
    :param ignore_empty: Filter out hyps of 0 length.
    :return:
        scores: a single dict with average values (avg=True) or a list of n dicts (avg=False)
        a dict:
        {"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-l": { ... }}
    """
    rouge = Rouge()
    scores = rouge.get_scores(hyps,refs,avg,ignore_empty)
    return scores


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data :
        (list of (src_sents, tgt_sents)) list of tuples containing source and target sentences.
    OR
        (list of src_sents) list of source sentences.
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    @return
        src_sents,tgt_sents: both list[list[str]] with length of batch_size.
    OR
        examples: (list[list[str]]) with length of batch_size.
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    if isinstance(data[0], tuple):
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]

            examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
            src_sents = [e[0] for e in examples]
            tgt_sents = [e[1] for e in examples]

            yield src_sents, tgt_sents
    elif isinstance(data[0], list):
        for i in range(batch_num):
            indices = index_array[i * batch_size: (i + 1) * batch_size]
            examples = [data[idx] for idx in indices]
            examples = sorted(examples, key=lambda e: len(e), reverse=True)
            yield examples


if __name__ == '__main__':
    print('read and split dataset...')
    for mode in ['train', 'dev', 'test']:
        dataset_cls3, dataset_cls18 = read_data('./data/{}.csv'.format(mode))
        with open('./data/{}_{}.pkl'.format(mode, 'cls3'), 'wb') as f:
            pickle.dump(dataset_cls3, f)
        with open('./data/{}_{}.pkl'.format(mode, 'cls18'), 'wb') as f:
            pickle.dump(dataset_cls18, f)
    exit(0)
