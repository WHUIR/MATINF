#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.utils.data as data

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


class Dataset(data.Dataset):
    def __init__(self):
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def add_data(self, question, description, answer, category):
        q = []
        d = []
        a = []

        for w in question:
            q.append(w)
        for w in description:
            d.append(w)
        for w in answer:
            a.append(w)
        if category in CLASS3_NAME_TO_INDEX:
            c = CLASS3_NAME_TO_INDEX[category]
        else:
            c = CLASS18_NAME_TO_INDEX[category]

        self.data.append({
            'question': q,  # list of tokens
            'description': d,  # list of tokens
            'answer': a,  # list of tokens
            'category': c  # int
        })
