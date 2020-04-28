#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import sys
from typing import List, Tuple
from collections import namedtuple

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])
MODE = ('summ', 'qa', 'cls3', 'cls18')


class Seq2seq(nn.Module):
    def __init__(self, hidden_size, vocab, embddings, enc_num_layers=1, dec_num_layers=1):
        super(Seq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.embeddings = embddings
        self.embeddings.weight.requires_grad = True
        self.embed_size = self.embeddings.weight.shape[1]
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers

        # 1 represents 'Summarization' task.
        self.encoder1 = nn.LSTM(self.embed_size, self.hidden_size, self.enc_num_layers)
        self.decoder1 = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.att_projection1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.h_projection1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.c_projection1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.combined_output_projection1 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.dropout1 = nn.Dropout(0.2)
        self.target_vocab_projection1 = nn.Linear(self.hidden_size, self.vocab.size())

        # 2 represents 'QA' task.
        self.encoder2 = self.decoder1
        self.decoder2 = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.h_projection2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.c_projection2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.att_projection2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.combined_output_projection2 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.dropout2 = nn.Dropout(0.2)
        self.target_vocab_projection2 = nn.Linear(self.hidden_size, self.vocab.size())

        self.cls_dropout = nn.Dropout(0.2)
        self.fc_share = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # where 3 represents 3-classification; 18 represents 18-classification.
        self.fc3 = nn.Linear(self.hidden_size, 3)
        self.fc18 = nn.Linear(self.hidden_size, 18)

    def forward(self, source, target, mode, source2=None):
        """
        :param
            source: (list[list[str]])
            target: ([list[list[str]]])
        :return
            scores (b,): Array of log-likelihoods of target sentences. (for summ & QA tasks)
        OR  y_pred (b, 3|18) (for classification tasks)
        """
        assert mode in MODE, 'unrecognized mode!'
        if mode == 'summ':
            source_lengths = [len(s) for s in source]
            source_padded = self.vocab.to_input_tensor(source, device=self.device)
            target_padded = self.vocab.to_input_tensor(target, device=self.device)

            enc_hiddens, dec_init_state, _ = self.encode_summ(source_padded, source_lengths)
            enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
            combined_outputs = self.decode_summ(dec_init_state, target_padded, enc_hiddens, enc_masks)

            P = F.log_softmax(self.target_vocab_projection1(combined_outputs),
                              dim=-1)
            target_masks = (target_padded != self.vocab.word2id['<pad>']).float()

            # Compute log probability of generating true target words
            target_gold_words_log_prob = torch.gather(P, index=(target_padded[1:]).unsqueeze(-1), dim=-1).squeeze(
                -1) * target_masks[1:]
            scores = target_gold_words_log_prob.sum(dim=0)
            return scores
        elif mode == 'qa':
            source_lengths = [len(s) for s in source]
            source_padded = self.vocab.to_input_tensor(source, device=self.device)
            target_padded = self.vocab.to_input_tensor(target, device=self.device)

            enc_hiddens, dec_init_state, _ = self.encode_qa(source_padded, source_lengths)
            enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
            combined_outputs = self.decode_qa(dec_init_state, target_padded, enc_hiddens, enc_masks)

            P = F.log_softmax(self.target_vocab_projection2(combined_outputs),
                              dim=-1)  # (max_len_sents, batch_size, vocab_size)

            # Zero out, probabilities for which we have nothing in the target text
            target_masks = (target_padded != self.vocab.word2id['<pad>']).float()

            # Compute log probability of generating true target words
            target_gold_words_log_prob = torch.gather(P, index=(target_padded[1:]).unsqueeze(-1), dim=-1).squeeze(
                -1) * target_masks[1:]
            scores = target_gold_words_log_prob.sum(dim=0)
            return scores
        elif mode.startswith('cls'):
            source_lengths1 = [len(s) for s in source]
            source_padded1 = self.vocab.to_input_tensor(source, device=self.device)

            source_lengths2 = [len(s) for s in source2]
            source_padded2 = self.vocab.to_input_tensor(source2, device=self.device)

            _, _, last_hidden1 = self.encode_summ(source_padded1, source_lengths1)
            _, _, last_hidden2 = self.encode_qa(source_padded2, source_lengths2)
            x1 = torch.squeeze(last_hidden1, dim=0)
            x = torch.cat((x1, last_hidden2), dim=1)
            x = self.fc_share(x)
            x = self.cls_dropout(x)
            if mode == 'cls3':
                y_pred = self.fc3(x)
            else:
                y_pred = self.fc18(x)

            return y_pred

    def encode_summ(self, source_padded, source_lengths):
        """Apply encoder to source_padded to obtain the hidden states
        """
        X = self.embeddings(source_padded)
        self.encoder1.flatten_parameters()
        enc_hiddens, (last_hidden, last_cell) = self.encoder1(
            pack_padded_sequence(X, source_lengths, enforce_sorted=False))
        enc_hiddens = pad_packed_sequence(enc_hiddens, batch_first=True)[0]

        init_decoder_hidden = self.h_projection1(torch.squeeze(last_hidden, dim=0))
        init_decoder_cell = self.c_projection1(torch.squeeze(last_cell, dim=0))
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state, last_hidden

    def encode_qa(self, source_padded, source_lengths):
        """Apply encoder to source_padded to obtain the final state of encoder.
        """
        X = self.embeddings(source_padded)
        max_length = X.size()[0]
        batch_size = X.size()[1]
        embed_size = X.size()[2]
        enc_hiddens = torch.zeros(X.size()).to(self.device)
        last_hidden = None
        last_cell = None
        h_t = torch.zeros(batch_size, embed_size).to(self.device)
        c_t = torch.zeros(batch_size, embed_size).to(self.device)
        i = 0
        for x_t in torch.split(X, 1, dim=0):
            x_t = torch.squeeze(x_t, dim=0)
            (h_t, c_t) = self.encoder2(x_t, (h_t, c_t))
            enc_hiddens[i] = h_t
            i = i + 1
            if i == max_length:
                last_hidden = h_t
                last_cell = c_t
        init_decoder_hidden = self.h_projection2(last_hidden)
        init_decoder_cell = self.c_projection2(last_cell)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens.permute(1, 0, 2), dec_init_state, last_hidden

    def decode_summ(self, dec_init_state: (torch.Tensor, torch.Tensor), target_padded: torch.Tensor,
                    enc_hiddens: torch.Tensor, enc_masks: torch.Tensor):
        # Chop of the <END> token for max length sentences.
        batch_size = target_padded.size()[1]
        target_padded = target_padded[:-1]

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        dec_state = dec_init_state

        combined_outputs = []

        enc_hiddens_proj = self.att_projection1(enc_hiddens)
        Y = self.embeddings(target_padded)

        for Y_t in torch.split(Y, 1, dim=0):
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.add(Y_t, o_prev)
            dec_state, o_t, e_t = self.step_summ(Ybar_t,
                                                 dec_state,
                                                 enc_hiddens,
                                                 enc_hiddens_proj,
                                                 enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def decode_qa(self, dec_init_state: (torch.Tensor, torch.Tensor), target_padded: torch.Tensor,
                  enc_hiddens: torch.Tensor, enc_masks: torch.Tensor):

        # Chop of the <END> token for max length sentences.
        batch_size = target_padded.size()[1]
        target_padded = target_padded[:-1]

        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)
        dec_state = dec_init_state

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        enc_hiddens_proj = self.att_projection2(enc_hiddens)
        Y = self.embeddings(target_padded)

        for Y_t in torch.split(Y, 1, dim=0):
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.add(Y_t, o_prev)
            dec_state, o_t, e_t = self.step_qa(Ybar_t,
                                               dec_state,
                                               enc_hiddens,
                                               enc_hiddens_proj,
                                               enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def step_summ(self, Ybar_t: torch.Tensor,
                  dec_state: Tuple[torch.Tensor, torch.Tensor],
                  enc_hiddens: torch.Tensor,
                  enc_hiddens_proj: torch.Tensor,
                  enc_masks: torch.Tensor):
        """One forward step of the decoder.
        :param Y_t: (batch_size, embed_size) The first tokens of each of the mini-batch of sents.
        :param dec_state: ...
        :returns dec_state: the current state of decoder.
        :returns output: the current hidden state of decoder.
        """
        dec_state = self.decoder1(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(
            torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2)), dim=2)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1)
        a_t = torch.squeeze(
            torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens),
            1)
        U_t = torch.cat((a_t, dec_hidden), dim=1)
        V_t = self.combined_output_projection1(U_t)
        O_t = self.dropout1(torch.tanh(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    def step_qa(self, Ybar_t: torch.Tensor,
                dec_state: Tuple[torch.Tensor, torch.Tensor],
                enc_hiddens: torch.Tensor,
                enc_hiddens_proj: torch.Tensor,
                enc_masks: torch.Tensor):
        """One forward step of the decoder.
        :param Y_t: (batch_size, embed_size) The first tokens of each of the mini-batch of sents.
        :param dec_state: ...
        :returns dec_state: the current state of decoder.
        :returns output:  the current hidden state of decoder.
        """

        dec_state = self.decoder2(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        e_t = torch.squeeze(
            torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden, 2)),
            dim=2)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = F.softmax(e_t, dim=1)
        a_t = torch.squeeze(
            torch.bmm(torch.unsqueeze(alpha_t, 1), enc_hiddens),
            1)
        U_t = torch.cat((a_t, dec_hidden), dim=1)
        V_t = self.combined_output_projection2(U_t)
        O_t = self.dropout2(torch.tanh(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    def predict(self, src_sent: List[List[str]]):
        """Predict the output sentence according to the src_sent.
        """

        src_sent_tensor = self.vocab.to_input_tensor(src_sent, self.device)
        dec_init_state = self.encode_summ(src_sent_tensor, [len(s) for s in src_sent])

        dec_state = dec_init_state
        batch_size = dec_state[0].size()[0]

        hypotheses = [''] * batch_size

        flags = [False] * batch_size
        y_t = [['<start>']] * batch_size
        y_t = self.vocab.to_input_tensor(y_t, device=self.device)
        y_t = self.embeddings(y_t)

        stop = False
        MAX_SENT = 50
        count = 0
        while not stop:
            count += 1
            stop = True
            y_t = torch.squeeze(y_t, dim=0)
            dec_state, output = self.step_summ(y_t, dec_state)
            top1_idxs = torch.argmax(F.log_softmax(self.target_vocab_projection1(output), dim=-1), -1)
            top1_idxs = top1_idxs.tolist()  # Convert tensor to list with length of batch_size.
            current_words = [self.vocab.id2word[id] for id in top1_idxs]
            for i in range(len(current_words)):
                if current_words[i] == '<end>':
                    flags[i] = True
                if not flags[i]:
                    hypotheses[i] = hypotheses[i] + current_words[i]
            for f in flags:
                if not f:
                    stop = False
            if count >= MAX_SENT:
                break
            y_t = [[hyp[-1]] for hyp in hypotheses]
            y_t = self.vocab.to_input_tensor(y_t, device=self.device)
            y_t = self.embeddings(y_t)

        return hypotheses

    def beam_search(self, mode: str, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70):
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        assert mode in ['summ', 'qa']
        src_sents_var = self.vocab.to_input_tensor([src_sent], self.device)
        if mode == 'summ':
            src_encodings, dec_init_vec, _ = self.encode_summ(src_sents_var, [len(src_sent)])
            src_encodings_att_linear = self.att_projection1(src_encodings)
        else:
            src_encodings, dec_init_vec, _ = self.encode_qa(src_sents_var, [len(src_sent)])
            src_encodings_att_linear = self.att_projection2(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        hypotheses = [['<start>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.word2id[hyp[-1]] for hyp in hypotheses], dtype=torch.long,
                                 device=self.device)
            y_t_embed = self.embeddings(y_tm1)
            x = torch.add(y_t_embed, att_tm1)

            if mode == 'summ':
                (h_t, cell_t), att_t, _ = self.step_summ(x, h_tm1,
                                                         exp_src_encodings, exp_src_encodings_att_linear,
                                                         enc_masks=None)
                log_p_t = F.log_softmax(self.target_vocab_projection1(att_t), dim=-1)
            else:
                (h_t, cell_t), att_t, _ = self.step_qa(x, h_tm1,
                                                       exp_src_encodings, exp_src_encodings_att_linear,
                                                       enc_masks=None)
                log_p_t = F.log_softmax(self.target_vocab_projection2(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / self.vocab.size()
            hyp_word_ids = top_cand_hyp_pos % self.vocab.size()

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '<end>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        hypothesis = Hypothesis(value=None, score=-float('inf'))
        for hypo in completed_hypotheses:
            if hypo.score > hypothesis.score:
                hypothesis = hypo

        return ''.join(hypothesis.value)

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float, device=self.device)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        """
        model = torch.load(model_path, map_location=torch.device('cuda:0'))

        return model

    @staticmethod
    def save(model, path: str):
        """ Save the model to a file.
        """
        print('save the model to [%s]' % path, file=sys.stderr)
        torch.save(model, path)
