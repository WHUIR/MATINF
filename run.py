#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import os
import math

import torch
import torch.nn as nn
import torch.utils.data as data

from typing import Dict, List
from rouge.rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pickle
import re
from collections import namedtuple
from itertools import cycle
from tqdm import tqdm

from vocab import Vocab
from seq2seq_model import Seq2seq


MODEs = ('summ', 'qa', 'cls3', 'cls18')
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

seed = 2019

BATCH_SIZE_SUMM_QA = 80
BATCH_SIZE_CLS3 = 16
BATCH_SIZE_CLS18 = 64

SUMM_WEIGHT = 1.0
QA_WEIGHT = 1.0
CLS_WEIGHT = 1.0

log_every = 100
valid_niter = 500
max_patience = 10
max_epoch = 10

VALID_BATCH = 64
VALID_NUM = -1    # '-1' if use the whole dev set to validate.
TEST_DATA_FILE = os.sep.join(['DATA', 'test.csv'])
model_save_path = 'checkpoints/'
model_load_path = ''
USE_CUDA = True
GPU_PARALLEL = False
is_training = True
OUTPUT_FILE = os.sep.join(['output', 'test_output.txt'])
base_learning_rate = 0.001

DATASET_TRAIN_CLS3 = './data/train_cls3.pkl'
DATASET_TRAIN_CLS18 = './data/train_cls18.pkl'
DATASET_DEV_CLS3 = './data/dev_cls3.pkl'
DATASET_DEV_CLS18 = './data/dev_cls18.pkl'
DATASET_TEST_CLS3 = './data/test_cls3.pkl'
DATASET_TEST_CLS18 = './data/test_cls18.pkl'

vocab_file = './data/vocab.json'
embeddings_file = './data/embeddings.pkl'

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

device = torch.device("cuda:0" if USE_CUDA else "cpu")


def train(args):
    print('Loading dataset...')
    with open(DATASET_TRAIN_CLS3, 'rb') as f:
        dataset_tr_cls3 = pickle.load(f)
    with open(DATASET_TRAIN_CLS18, 'rb') as f:
        dataset_tr_cls18 = pickle.load(f)
    dataset_tr_summ_qa = data.ConcatDataset([dataset_tr_cls3, dataset_tr_cls18])

    dataloader_tr_summ_qa = cycle(data.DataLoader(dataset=dataset_tr_summ_qa,
                                                  batch_size=BATCH_SIZE_SUMM_QA,
                                                  shuffle=True,
                                                  collate_fn=lambda x: x))
    dataloader_tr_cls3 = cycle(data.DataLoader(dataset=dataset_tr_cls3,
                                               batch_size=BATCH_SIZE_CLS3,
                                               shuffle=True,
                                               collate_fn=lambda x: x))
    dataloader_tr_cls18 = cycle(data.DataLoader(dataset=dataset_tr_cls18,
                                                batch_size=BATCH_SIZE_CLS18,
                                                shuffle=True,
                                                collate_fn=lambda x: x))

    print('Loading vocab...')
    vocab = Vocab.load(vocab_file)

    print('Loading embeddings...')
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    print('-----OK-----')

    if not os.path.exists(model_save_path):
        print('create dir: {}'.format(model_save_path))
        os.mkdir(model_save_path)

    if model_load_path:
        print('Loading model...')
        model = Seq2seq.load(model_load_path)
    else:
        model = Seq2seq(hidden_size=200, vocab=vocab, embddings=embeddings)

    if USE_CUDA:
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
    if GPU_PARALLEL:    # there may exists something wrong... please set it to 'False'.
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=base_learning_rate)
    CELoss = nn.CrossEntropyLoss()

    epoch = train_iter = 0
    report_iter_num = 0
    report_loss_summ = report_loss_qa = report_loss_cls3 = 0
    report_loss_cls18 = 0
    cum_examples_summ_qa = cum_examples_cls3 = cum_examples_cls18 = 0
    best_results = [0, 0, 0, 0]  # best results for [summ, qa, cls3, cls18]

    # patience = 0
    iter_num_summ_qa = math.ceil(len(dataset_tr_summ_qa) / BATCH_SIZE_SUMM_QA)
    iter_num_cls3 = math.ceil(len(dataset_tr_cls3) / BATCH_SIZE_CLS3)
    iter_num_cls18 = math.ceil(len(dataset_tr_cls18) / BATCH_SIZE_CLS18)
    iter_num_of_one_epoch = min(iter_num_summ_qa, iter_num_cls3, iter_num_cls18)

    begin_time = time.time()

    while True:
        epoch += 1

        for i in range(iter_num_of_one_epoch):
            train_iter += 1
            report_iter_num += 1

            # --------------------------------------------------------------------
            # summ: D -> Q
            optimizer.zero_grad()

            mini_batch = next(iter(dataloader_tr_summ_qa))
            question = [data['question'] for data in mini_batch]
            description = [data['description'] for data in mini_batch]

            for i in range(len(question)):
                question[i].insert(0, '<start>')
                question[i].insert(len(question[i]), '<end>')

            try:
                example_losses_summ = -model(description, question, mode='summ')
                batch_loss_summ = example_losses_summ.sum()  # total batch loss.
                loss_summ = batch_loss_summ / len(mini_batch) * SUMM_WEIGHT  # final(avg.) batch loss
                loss_summ.backward()

                report_loss_summ += batch_loss_summ.item()
                cum_examples_summ_qa += len(mini_batch)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()

            # --------------------------------------------------------------------
            # QA: Q -> A
            optimizer.zero_grad()

            question = [data['question'] for data in mini_batch]
            answer = [data['answer'] for data in mini_batch]

            for i in range(len(answer)):
                answer[i].insert(0, '<start>')
                answer[i].insert(len(answer[i]), '<end>')

            try:
                example_losses_qa = -model(question, answer, mode='qa')
                batch_loss_qa = example_losses_qa.sum()  # total batch loss.
                loss_qa = batch_loss_qa / len(mini_batch) * QA_WEIGHT  # final(ave) batch loss
                loss_qa.backward()

                report_loss_qa += batch_loss_qa.item()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()

            # --------------------------------------------------------------------
            # cls3: D, Q -> C
            optimizer.zero_grad()

            mini_batch = next(iter(dataloader_tr_cls3))
            question = [data['question'] for data in mini_batch]
            description = [data['description'] for data in mini_batch]
            category = torch.tensor([data['category'] for data in mini_batch]).to(device)

            y_pred = model(source=description, source2=question, target=None, mode='cls3')
            loss_cls3 = CELoss(y_pred, category) * CLS_WEIGHT

            try:
                loss_cls3.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()

            report_loss_cls3 += loss_cls3.item()
            cum_examples_cls3 += len(mini_batch)

            # --------------------------------------------------------------------
            # cls18: D,Q -> C
            optimizer.zero_grad()

            mini_batch = next(iter(dataloader_tr_cls18))
            question = [data['question'] for data in mini_batch]
            description = [data['description'] for data in mini_batch]
            category = torch.tensor([data['category'] for data in mini_batch]).to(device)

            y_pred = model(source=description, source2=question, target=None, mode='cls18')
            loss_cls18 = CELoss(y_pred, category) * CLS_WEIGHT

            try:
                loss_cls18.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()

            report_loss_cls18 += loss_cls18.item()
            cum_examples_cls18 += len(mini_batch)

            if train_iter % log_every == 0:
                print('-' * 50)
                print('epoch:', epoch)
                print('iters:', train_iter)
                kwargs_summ = {
                    'report_loss': report_loss_summ,
                    'report_iter_num': report_iter_num,
                    'cum_examples': cum_examples_summ_qa,
                    'num_of_train_set': len(dataset_tr_summ_qa),
                    'begin_time': begin_time
                }
                report(mode='summ', **kwargs_summ)
                kwargs_qa = {
                    'report_loss': report_loss_qa,
                    'report_iter_num': report_iter_num,
                    'cum_examples': cum_examples_summ_qa,
                    'num_of_train_set': len(dataset_tr_summ_qa),
                    'begin_time': begin_time
                }
                report(mode='qa', **kwargs_qa)
                kwargs_cls3 = {
                    'report_loss': report_loss_cls3,
                    'report_iter_num': report_iter_num,
                    'cum_examples': cum_examples_cls3,
                    'num_of_train_set': len(dataset_tr_cls3),
                    'begin_time': begin_time
                }
                report(mode='cls3', **kwargs_cls3)
                kwargs_cls18 = {
                    'report_loss': report_loss_cls18,
                    'report_iter_num': report_iter_num,
                    'cum_examples': cum_examples_cls18,
                    'num_of_train_set': len(dataset_tr_cls18),
                    'begin_time': begin_time
                }
                report(mode='cls18', **kwargs_cls18)
                print('-' * 50)
                report_loss_summ = report_loss_qa = report_loss_cls3 = report_loss_cls18 = 0
                report_iter_num = 0

            if train_iter % valid_niter == 0:
                print('begin validation ...', file=sys.stderr)

                which_better = []
                save_model = False
                results = valid(model)
                for i in range(len(results)):
                    if i < 2 and isinstance(results[i], dict) and results[i]['rouge-l']['f'] > best_results[i]:
                        save_model = True
                        best_results[i] = results[i]['rouge-l']['f']
                        which_better.append(i)
                    elif i >= 2 and results[i] > best_results[i]:
                        save_model = True
                        best_results[i] = results[i]
                        which_better.append(i)

                if save_model:
                    print('Task {} get better scores!'.format(which_better))
                    Seq2seq.save(model, model_save_path + 'model_iter_{}.pt'.format(train_iter))

        cum_examples_summ_qa = cum_examples_cls3 = cum_examples_cls18 = 0
        # END one epoch.

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=sys.stderr)
            # exit(0)
            break


# dev
def valid(model, mode='all'):
    model.eval()
    with open(DATASET_DEV_CLS3, 'rb') as f:
        dataset_cls3 = pickle.load(f)
    with open(DATASET_DEV_CLS18, 'rb') as f:
        dataset_cls18 = pickle.load(f)
    dataset_summ_qa = data.ConcatDataset([dataset_cls3, dataset_cls18])

    cls3_loader = torch.utils.data.DataLoader(dataset=dataset_cls3,
                                              batch_size=VALID_BATCH,
                                              shuffle=False,
                                              collate_fn=lambda x: x)
    cls3_iterator = iter(cls3_loader)
    cls18_loader = torch.utils.data.DataLoader(dataset=dataset_cls18,
                                               batch_size=VALID_BATCH,
                                               shuffle=False,
                                               collate_fn=lambda x: x)
    cls18_iterator = iter(cls18_loader)

    rouge_summ = rouge_qa = None
    acc_cls3 = acc_cls18 = 0
    # --------------------------------------------------------------------
    if mode in ['all', 'summ','qa']:
        data_val_sum_qa = []
        if VALID_NUM > 0:
            for i in range(VALID_NUM):
                data_val_sum_qa.append(dataset_summ_qa[i])
        else:
            for i in range(len(dataset_summ_qa)):
                data_val_sum_qa.append(dataset_summ_qa[i])

    if mode in ['all','summ']:
        refs = [' '.join(data['question']) for data in data_val_sum_qa]
        x = [data['description'] for data in data_val_sum_qa]
        hyps = beam_search('summ', model, x)
        hyps = [' '.join(list(sent)) for sent in hyps]
        rouge = Rouge()
        try:
            rouge_summ = rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
            print_rouge(rouge_summ)
        except RuntimeError:
            print('Failed to compute Rouge!')

    if mode in ['all', 'qa']:
        refs = [' '.join(data['answer']) for data in data_val_sum_qa]
        x = [data['question'] for data in data_val_sum_qa]
        hyps = beam_search('qa', model, x)
        hyps = [' '.join(list(sent)) for sent in hyps]
        rouge = Rouge()
        try:
            rouge_qa = rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
            print_rouge(rouge_qa)
        except RuntimeError:
            print('Failed to compute Rouge!')

    # cls3 & cls18
    def iter_through_cls_dev(iterator, mode):
        val_correct = 0
        val_num = 0
        for i in range(math.ceil(VALID_NUM / VALID_BATCH)):
            mini_batch = next(iterator)
            question = [data['question'] for data in mini_batch]
            description = [data['description'] for data in mini_batch]
            y_gt = torch.tensor([data['category'] for data in mini_batch]).to(device)
            y_pred = model(source=description, source2=question, target=None, mode=mode)
            y_pred_labels = torch.argmax(y_pred, dim=1)
            val_correct += (y_gt == y_pred_labels).sum().item()
            val_num += len(mini_batch)

        return val_correct / val_num

    if mode in ['all', 'cls3']:
        acc_cls3 = iter_through_cls_dev(cls3_iterator, 'cls3')
        print('Acc_cls3:', acc_cls3)
    if mode in ['all', 'cls18']:
        acc_cls18 = iter_through_cls_dev(cls18_iterator, 'cls18')
        print('Acc_cls18:', acc_cls18)

    if is_training:
        model.train()

    return rouge_summ, rouge_qa, acc_cls3, acc_cls18


def report(mode: str, **kwargs):
    if mode not in MODEs:
        print('Failed to report! Invalid mode {}.'.format(mode))
        return

    print('mode %s: avg. loss %.2f, progress %.2f, '
          'time elapsed %.2f sec' % (mode,
                                     kwargs['report_loss'] / kwargs[
                                         'report_iter_num'],
                                     float(kwargs['cum_examples']) / kwargs['num_of_train_set'] * 100,
                                     time.time() - kwargs['begin_time']))


def print_rouge(rouge: Rouge):
    # print('p: ', [str(rouge['rouge-1']['p']), str(rouge['rouge-2']['p']), str(rouge['rouge-l']['p'])])
    # print('r: ', [str(rouge['rouge-1']['r']), str(rouge['rouge-2']['r']), str(rouge['rouge-l']['r'])])
    print('f: ', [str(rouge['rouge-1']['f']), str(rouge['rouge-2']['f']), str(rouge['rouge-l']['f'])])


# Test
def evaluate_summ_qa(model, dataset, mode, batch_size=64):
    assert mode in ('summ', 'qa'), 'Invalid mode!'

    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=lambda x: x)

    rouge1_f_sum = rouge2_f_sum = rougeL_f_sum = bleu_sum = 0
    examples_rouge = examples_bleu = 0

    rouge = Rouge()
    count = 0
    if mode == 'summ':
        for mini_batch in tqdm(data_loader):
            count += 1
            refs = [' '.join(data['question']) for data in mini_batch]
            x = [data['description'] for data in mini_batch]
            hyps_raw = beam_search('summ', model, x)
            hyps = [' '.join(list(sent)) for sent in hyps_raw]
            try:
                rouge_score = rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
                rouge1_f_sum += rouge_score['rouge-1']['f'] * len(mini_batch)
                rouge2_f_sum += rouge_score['rouge-2']['f'] * len(mini_batch)
                rougeL_f_sum += rouge_score['rouge-l']['f'] * len(mini_batch)
                examples_rouge += len(mini_batch)
            except ValueError as e:
                print(str(e) + ' | continuing...')
                continue

    elif mode == 'qa':
        for mini_batch in tqdm(data_loader):
            count += 1
            refs = [' '.join(data['answer']) for data in mini_batch]
            x = [data['question'] for data in mini_batch]
            hyps_raw = beam_search('qa', model, x)
            hyps = [' '.join(list(sent)) for sent in hyps_raw]
            try:
                rouge_score = rouge.get_scores(hyps, refs, avg=True, ignore_empty=True)
                rouge1_f_sum += rouge_score['rouge-1']['f'] * len(mini_batch)
                rouge2_f_sum += rouge_score['rouge-2']['f'] * len(mini_batch)
                rougeL_f_sum += rouge_score['rouge-l']['f'] * len(mini_batch)
                examples_rouge += len(mini_batch)
            except ValueError as e:
                print(str(e) + ' | continuing...')
                continue

            # calculate BLEU score
            refs = [data['answer'] for data in mini_batch]
            hyps = [list(sent) for sent in hyps_raw]
            smoothie = SmoothingFunction().method4
            for i in range(len(hyps)):
                try:
                    bleu = sentence_bleu([refs[i]], hyps[i], smoothing_function=smoothie)
                    bleu_sum += bleu
                    examples_bleu += 1
                except ZeroDivisionError as e:
                    print(str(e) + ' | continuing...')
                    continue

    rouge_1_f = rouge1_f_sum / examples_rouge
    rouge_2_f = rouge2_f_sum / examples_rouge
    rouge_L_f = rougeL_f_sum / examples_rouge
    if mode == 'qa':
        bleu_score = bleu_sum / examples_bleu

    # with open('output/test_{}.txt'.format(mode), 'w', encoding='utf-8') as f:
    #     f.write('rouge-1 f: ' + str(rouge_1_f) + '\n')
    #     f.write('rouge-2 f: ' + str(rouge_2_f) + '\n')
    #     f.write('rouge-L f: ' + str(rouge_L_f) + '\n')
    #     f.write('\n')
    #
    #     for i in range((len(candidates)):
    #         f.write('input: ' + inputs[i] + '\n')
    #         f.write('hyp: ' + ''.join(candidates[i]) + '\n')
    #         f.write('ref: ' + targets[i] + '\n\n')

    if is_training:
        model.train()
    print('rouge-1 f: ' + str(rouge_1_f))
    print('rouge-2 f: ' + str(rouge_2_f))
    print('rouge-L f: ' + str(rouge_L_f))
    if mode == 'qa':
        print('bleu: ', bleu_score)


def evaluate_cls(model, dataset, mode, batch_size=16):
    assert mode in ('cls3', 'cls18'), 'Invalid mode!'
    model.eval()

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              collate_fn=lambda x: x)

    val_correct = 0
    val_num = 0

    for mini_batch in tqdm(data_loader):
        question = [data['question'] for data in mini_batch]
        description = [data['description'] for data in mini_batch]
        y_gt = torch.tensor([data['category'] for data in mini_batch]).to(device)  # (batch,1)
        y_pred = model(source=description, source2=question, target=None, mode=mode)  # (batch,3)
        y_pred_labels = torch.argmax(y_pred, dim=1)
        val_correct += (y_gt == y_pred_labels).sum()
        val_num += len(mini_batch)

    accuracy = val_correct.item() / val_num
    # with open('output/test_{}.txt'.format(mode), 'w', encoding='utf-8') as f:
    #     f.write('accuracy: ' + str(accuracy))

    if is_training:
        model.train()

    print('mode:' + mode + ' | acc: ' + str(accuracy))


def test(mode, model_path, args):
    """ Performs decoding on a test set, and save the best-scoring decoding results.

    """
    assert mode in MODEs, 'Invalid mode!'
    print('mode:', mode)
    print("load test data...")
    if mode == 'cls3':
        with open(DATASET_TEST_CLS3, 'rb') as f:
            dataset_test = pickle.load(f)
    elif mode == 'cls18':
        with open(DATASET_TEST_CLS18, 'rb') as f:
            dataset_test = pickle.load(f)
    else:
        with open(DATASET_TEST_CLS3, 'rb') as f:
            dataset_cls3 = pickle.load(f)
        with open(DATASET_TEST_CLS3, 'rb') as f:
            dataset_cls18 = pickle.load(f)
        dataset_test = data.ConcatDataset([dataset_cls3, dataset_cls18])

    print("load model from {}".format(model_path))
    model = Seq2seq.load(model_path)

    if USE_CUDA:
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
    if GPU_PARALLEL:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    if mode in ('summ', 'qa'):
        evaluate_summ_qa(model, dataset_test, mode, batch_size=128)
    else:
        evaluate_cls(model, dataset_test, mode, batch_size=512)


def beam_search(mode: str, model: Seq2seq, test_data_src: List[List[str]], beam_size: int = 5,
                max_decoding_time_step: int = 100):
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[str]): List of Hypothesis for every source sentence.
    """
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in test_data_src:
            example_hyps = model.beam_search(mode, src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if is_training:
        model.train()

    # with open('output/check_{}.txt'.format(mode), 'w', encoding='UTF-8') as f:
    #     for i in range(50):
    #         f.write('Source: ' + ''.join(test_data_src[i]))
    #         f.write('\n')
    #         f.write('Output: ' + hypotheses[i])
    #         f.write('\n-------------------------\n')
    hypotheses = [re.sub(r'<start>|<end>', '', sent) for sent in hypotheses]
    return hypotheses


def single_or_finetune(**kwargs):
    parameters = kwargs
    assert parameters['mode'] in MODEs
    print('Loading dataset...')
    if parameters['mode'] == 'cls3':
        with open(DATASET_TRAIN_CLS3, 'rb') as f:
            dataset = pickle.load(f)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=parameters['batch_size'],
                                                 shuffle=True,
                                                 collate_fn=lambda x: x)
    elif parameters['mode'] == 'cls18':
        with open(DATASET_TRAIN_CLS18, 'rb') as f:
            dataset = pickle.load(f)
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=parameters['batch_size'],
                                                 shuffle=False,
                                                 collate_fn=lambda x: x)
    else:
        with open(DATASET_TRAIN_CLS3, 'rb') as f:
            dataset_cls3 = pickle.load(f)
        with open(DATASET_TRAIN_CLS18, 'rb') as f:
            dataset_cls18 = pickle.load(f)
        dataset = data.ConcatDataset([dataset_cls3, dataset_cls18])
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=parameters['batch_size'],
                                                 shuffle=True,
                                                 collate_fn=lambda x: x)

    print('Loading vocab...')
    vocab = Vocab.load(vocab_file)

    print('Loading embeddings...')
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)
    print('-----OK-----')

    if not os.path.exists(parameters['model_save_path']):
        print('create dir: {}'.format(parameters['model_save_path']))
        os.mkdir(parameters['model_save_path'])

    if parameters['task'] == 'finetune' and parameters['model_load_path']:
        print('Loading model from {}...'.format(parameters['model_load_path']))
        model = Seq2seq.load(parameters['model_load_path'])
    elif parameters['task'] == 'single':
        model = Seq2seq(hidden_size=200, vocab=vocab, embddings=embeddings,
                        enc_num_layers=1, dec_num_layers=1)
    else:
        raise RuntimeError('Parameters error!')

    if USE_CUDA:
        print('use device: %s' % device, file=sys.stderr)
        model = model.to(device)
    if GPU_PARALLEL:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = model.to('cuda:0')
        model = torch.nn.DataParallel(model, device_ids=[0, 1])

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])
    CELoss = nn.CrossEntropyLoss()

    epoch = train_iter = 0
    report_iter_num = 0
    report_loss = cum_examples = 0
    best_results = 0

    print('Performing {} task, mode: {}.'.format(parameters['task'], parameters['mode']))
    begin_time = time.time()
    while True:
        epoch += 1
        for mini_batch in dataloader:
            train_iter += 1
            report_iter_num += 1

            # --------------------------------------------------------------------
            # summ
            if parameters['mode'] == 'summ':

                optimizer.zero_grad()

                question = [data['question'] for data in mini_batch]
                description = [data['description'] for data in mini_batch]

                for i in range(len(question)):
                    question[i].insert(0, '<start>')
                    question[i].insert(len(question[i]), '<end>')

                try:
                    example_losses_summ = -model(description, question, mode='summ')
                    batch_loss_summ = example_losses_summ.sum()
                    loss_summ = batch_loss_summ / len(mini_batch) * SUMM_WEIGHT
                    loss_summ.backward()

                    report_loss += batch_loss_summ.item()
                    cum_examples += len(mini_batch)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 25)

                optimizer.step()

            # --------------------------------------------------------------------
            # QA
            if parameters['mode'] == 'qa':
                optimizer.zero_grad()

                question = [data['question'] for data in mini_batch]
                answer = [data['answer'] for data in mini_batch]

                for i in range(len(answer)):
                    answer[i].insert(0, '<start>')
                    answer[i].insert(len(answer[i]), '<end>')

                try:
                    example_losses_qa = -model(question, answer, mode='qa')
                    batch_loss_qa = example_losses_qa.sum()
                    loss_qa = batch_loss_qa / len(mini_batch) * QA_WEIGHT
                    loss_qa.backward()

                    report_loss += batch_loss_qa.item()
                    cum_examples += len(mini_batch)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 25)

                optimizer.step()

            # --------------------------------------------------------------------
            # cls3
            if parameters['mode'] == 'cls3':
                optimizer.zero_grad()

                question = [data['question'] for data in mini_batch]
                description = [data['description'] for data in mini_batch]
                category = torch.tensor([data['category'] for data in mini_batch]).to(device)

                y_pred = model(source=description, source2=question, target=None, mode='cls3')
                loss_cls3 = CELoss(y_pred, category) * CLS_WEIGHT

                try:
                    loss_cls3.backward()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

                optimizer.step()

                report_loss += loss_cls3.item()
                cum_examples += len(mini_batch)

            # --------------------------------------------------------------------
            # cls18
            if parameters['mode'] == 'cls18':
                optimizer.zero_grad()

                question = [data['question'] for data in mini_batch]
                description = [data['description'] for data in mini_batch]
                category = torch.tensor([data['category'] for data in mini_batch]).to(device)

                y_pred = model(source=description, source2=question, target=None, mode='cls18')
                loss_cls18 = CELoss(y_pred, category) * CLS_WEIGHT

                try:
                    loss_cls18.backward()
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e

                # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

                optimizer.step()

                report_loss += loss_cls18.item()
                cum_examples += len(mini_batch)

            if train_iter % log_every == 0:
                print('-' * 50)
                print('epoch:', epoch)
                print('iters:', train_iter)
                kwargs_summ = {
                    'report_loss': report_loss,
                    'report_iter_num': report_iter_num,
                    'cum_examples': cum_examples,
                    'num_of_train_set': len(dataset),
                    'begin_time': begin_time
                }
                report(mode='summ', **kwargs_summ)

                print('-' * 50)
                report_loss = report_iter_num = 0

            if train_iter % valid_niter == 0:
                print('begin validation ...', file=sys.stderr)
                save_model = False
                results = valid(model, mode=parameters['mode'])
                i = MODEs.index(parameters['mode'])

                if i < 2 and isinstance(results[i], dict) and results[i]['rouge-l']['f'] > best_results:
                    save_model = True
                    best_results = results[i]['rouge-l']['f']
                elif i >= 2 and results[i] > best_results:
                    save_model = True
                    best_results = results[i]

                if save_model:
                    print('get better score!')
                    Seq2seq.save(model, parameters['model_save_path'] + 'model_{}_{}.pt'.format(parameters['mode'],
                                                                                                train_iter))

        cum_examples = 0
        # END one epoch.

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)


def main():
    # set the random number generators
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    stage = 'train'  # 'train'||'test'||'single'||'finetune'
    global is_training

    if stage == 'train':
        is_training = True
        train()
    elif stage == 'test':
        is_training = False
        test(mode='qa', model_path='./checkpoints_seed_2019/model_13000.pt')
    elif stage in ['finetune', 'single']:
        is_training = True
        parameters = {
            'lr': 1e-3,
            'task': stage,
            'mode': 'cls18',
            'model_load_path': './checkpoints_seed_2019/model_13000.pt',
            'model_save_path': './{}_seed_{}/'.format(stage, seed),
            'batch_size': 128,
        }
        single_or_finetune(**parameters)
    else:
        raise RuntimeError('invalid run mode')
    exit(0)


if __name__ == '__main__':
    main()
