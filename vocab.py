#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from utils import build_embeddings,pad_sents
import pandas as pd
import pickle


class Vocab:
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<start>'] = 0
            self.word2id['<end>'] = 1
            self.word2id['<pad>'] = 2
            self.word2id['<unk>'] = 3
        self.special_num = 4            # the number of the above special tokens.

        self.id2word = {v: k for k, v in self.word2id.items()}

    def add(self, word):
        if word not in self.word2id.keys():
            self.word2id[word] = len(self.word2id)
            self.id2word[self.word2id[word]] = word

    def size(self):
        return len(self.word2id)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.word2id, f)

    def word2indix(self, word:str):
        if word in self.word2id.keys():
            return self.word2id[word]
        else:
            return self.word2id['<unk>']

    def word2indices(self,sents):
        """Convert list of words or list of sentence of words into list or list of list indices.
        """
        if type(sents[0]) == list:
            return [[self.word2indix(w) for w in s] for s in sents]
        else:
            return [self.word2indix(w) for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents, device)->torch.Tensor:
        """Convert list of sentence into tensor with necessary padding for shorter sentence.
        """
        sents_padded = pad_sents(sents,'<pad>')
        word_ids = self.word2indices(sents_padded)
        sents_torch = torch.tensor(word_ids, dtype=torch.long,device=device)
        return sents_torch.t()

    @staticmethod
    def build(file:str):
        vocab = Vocab()

        data = pd.read_csv(file, sep=',', encoding='UTF-8')
        for i in range(0, len(data)):
            sent = str(data.iat[i,1]) + str(data.iat[i,2]) + str(data.iat[i,3])
            for word in sent:
                vocab.add(word)
        print('vocab size:', vocab.size())
        return vocab

    @staticmethod
    def load(file_path):
        with open(file_path) as f:
            word2id = json.load(f)
            return Vocab(word2id)


if __name__ == '__main__':
    print('Building vocab...')
    vocab = Vocab.build('./data/train.csv')
    vocab.save('./data/vocab.json')

    # print('Loading vocab...')
    # vocab = Vocab.load('./data/vocab.json')
    print('Building embeddings...')
    embeddings = build_embeddings('./data/ChineseEmbedding.txt', vocab)
    with open('./data/embeddings.pkl','wb') as f:
        pickle.dump(embeddings,f)