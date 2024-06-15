
import torch
import torch.nn.functional as F

from constants import DEFAULT_POS_PER_QUARTER, MASK_TOKEN, Ins_LIST

import copy

def get_bars(events):
    bars = [i for i, event in enumerate(events) if 'Bar_' in event[0]]
    # print(bars)
    # print(len(bars))

    bar_embed_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
    bar_embed_ids = torch.cumsum(bar_embed_ids, dim=0)

    bar_ids = torch.zeros(64, dtype=torch.long)
    bar_ids[:len(bars)] = torch.tensor(bars, dtype=torch.long)
    bar_len = torch.tensor(len(bars), dtype=torch.long)
    return bar_len, bar_ids, bar_embed_ids

def get_phrase_bar_len(phrase_info, bar_len_updated):
    # 获取每个乐句内的乐句数

    # bars_updated需为序列截断后的小节数量
    phrase_bar_len = [0] * 16

    counter = 0
    for i, p_info in enumerate(phrase_info):
        # each_bars_len = int(p_info['phrase'].split('-')[-1])
        each_bars_len = (p_info.end - p_info.start) // (DEFAULT_POS_PER_QUARTER * 4)
        counter += each_bars_len
        if counter > bar_len_updated:
            res = each_bars_len - (counter - bar_len_updated)
            if res > 0:
                phrase_bar_len[i] = res
        else:
            phrase_bar_len[i] = each_bars_len

    return phrase_bar_len

def get_phrases(events, phrase_bar_len):
    bars = [i for i, event in enumerate(events) if 'Bar_' in event[0]]

    phrases = []
    start = 0
    for i, _ in enumerate(phrase_bar_len):
        if phrase_bar_len[i] != 0:
            phrases.append(bars[start])
            start += phrase_bar_len[i]
    phrase_embed_ids = torch.bincount(torch.tensor(phrases, dtype=torch.int), minlength=len(events))
    phrase_embed_ids = torch.cumsum(phrase_embed_ids, dim=0)

    # 已改，只存储phrase所在首个小节的index
    phrase_ids = torch.zeros(16, dtype=torch.long)
    phrase_ids[:len(phrases)] = torch.tensor(phrases, dtype=torch.long)

    return phrase_ids, phrase_embed_ids

def get_ins_name(remi_track, track_status):
    ins_name_list = []
    for i, track_events in enumerate(remi_track):
        if track_status[i]:
            ins_name_list.append(track_events[0])
        else:
            ins_name_list.append(MASK_TOKEN)
    return ins_name_list

def delete_phrase_tokens(events):
    events_post = []
    for event in events:
        if 'Phrase_' in event or 'BCD_' in event:
            continue
        else:
            events_post.append(event)
    return events_post

def delete_phrase_tokens2(events):
    events_post = []
    for event in events:
        if 'Phrase_' in event:
            events_post.append('<mask>')
        elif 'BCD_' in event:
            continue
        else:
            events_post.append(event)
    return events_post

def padding_prior(prior, max_bars):
    # chords_seq: (bar_nums, 4)
    # feat_seq: (bar_nums, 6) / (bar_nums, 4, TRACK_NUMS)
    # vq_codes: (bar_nums, 16) / (bar_Nums, 16, TRACK_NUMS)

    # padding to max_bar_nums
    if len(prior.shape) == 2:
        prior = F.pad(prior, (0, 0, 0, max_bars-prior.shape[0]), 'constant', 0)
    elif len(prior.shape) == 3:
        prior = F.pad(prior, (0, 0, 0, 0, 0, max_bars - prior.shape[0]), 'constant', 0)
    else:
        raise ValueError('error prior shape.')
    assert prior.shape[0] == max_bars

    return prior

def reduce_music_info(music_info_ori):
    music_info = copy.deepcopy(music_info_ori)
    track_status = torch.tensor(music_info['TRACK_STATUS'], dtype=torch.bool)

    track_status[0] = True

    notes_info = music_info['NOTES_INFO']

    ins_notes_num = torch.ones(6, dtype=torch.long) * -4096
    for ins_name in notes_info.keys():
        ins_idx = Ins_LIST.index(ins_name)
        if not(ins_idx in [0, 5] or track_status[ins_idx] == False):
            ins_notes_num[ins_idx] = -len(notes_info[ins_name])

    k = torch.sum(track_status) - 4
    if k > 0:
        skip_track_index = torch.topk(ins_notes_num, k)[-1]
        track_status[skip_track_index] = False
    assert torch.sum(track_status) == 4

    track_status = track_status.tolist()
    for ins_name in notes_info.keys():
        ins_idx = Ins_LIST.index(ins_name)
        if not track_status[ins_idx]:
            notes_info[ins_name] = []

    music_info['NOTES_INFO'] = notes_info
    music_info['TRACK_STATUS'] = track_status

    return music_info






