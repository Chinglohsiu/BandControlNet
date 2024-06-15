import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import os

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

class PositionalEncoding(nn.Module):
    def __init__(self, D, dropout=0.1, max_len=2**14):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(1e4) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (bz, seq_len, D)
        # pe: (1, max_len, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, vocab_size, D, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, D, padding_idx)
        self.D = D

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.D)


def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs

def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word

def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def sampling(logit, t=1.1, p=0.9):
    logit = logit.squeeze().detach().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word

def topk_sampling(logit, k_p=0.1):
    # refer to x-transformers.autoregressive_wrapper
    num_tokens = logit.shape[-1]
    k = int(num_tokens * k_p)

    val, ind = torch.topk(logit, k)
    probs = torch.full_like(logit, float('-inf'))
    return probs.scatter(1, ind, val)

def topp_sampling(logit, t=1.1, p=0.9):
    probs = torch.softmax(logit / t, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cum_probs > p
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_probs[sorted_indices_to_remove] = float('-inf')
    return sorted_probs.scatter(1, sorted_indices, sorted_probs)

def sampling_v2(logit, k_p=None, t=None, p=None, mode='topp'):
    # topk: t&p
    # topp: k_p
    if mode == 'topp':
        assert k_p == None
        probs = topp_sampling(logit, t, p)

    if mode == 'topk':
        assert p == None
        assert t == None
        probs = topk_sampling(logit, k_p)

    probs = torch.softmax(probs, dim=-1)
    return torch.multinomial(probs, 1)


def get_lr_multiplier(step, warmup_steps, decay_end_steps, decay_end_multiplier):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def recover_position_track(merged_events_per_track):
    # 对multiple_sequence的单个轨道而言
    pos_index = [i for i, e in enumerate(merged_events_per_track) if 'Position_' in e] + [len(merged_events_per_track)]
    recover_events_per_track = merged_events_per_track[:pos_index[0]]

    for pos_st, pos_et in zip(pos_index[:-1], pos_index[1:]):
        pos_events = merged_events_per_track[pos_st:pos_et]
        pos_new_events = []
        for event in pos_events:
            if 'Position_' in event:
                current_position = event
            if 'Pitch_' not in event:
                pos_new_events.append(event)
            else:
                if pos_new_events[-1] != current_position:
                    pos_new_events.append(current_position)
                    pos_new_events.append(event)
                else:
                    pos_new_events.append(event)

        recover_events_per_track += pos_new_events
    return recover_events_per_track

def count_note_nums(midi):
    note_nums = []
    for ins in midi.instruments:
        note_nums.append(len(ins.notes))
    return sum(note_nums)