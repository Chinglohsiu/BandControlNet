
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from representation_multiple_v2 import REMI_Plus_Raw
from vocab_v2 import RemiVocab
from constants import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, Ins_LIST
    )

from utils.dataset_utils import reduce_music_info

import pickle
import math
from tqdm import tqdm
import os

SERVER_DATA_DIR = '/home/data/music_gen_cl/Bandformer_new'
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VQVAE_Dataset_Raw(data.Dataset):
    def __init__(self, file_names,

                 max_seq_len=1024,
                 use_augmentation=False,
                 chord_enable=False,
                 velocity_enable=False,     
                 fix_phrase_bar_num=None,
                 server_mode=False,
                 ):

        self.file_names = file_names
        self.max_seq_len = max_seq_len - 1   

        self.use_augmentation = use_augmentation
        self.server_mode = server_mode

        self.chord_enable = chord_enable
        self.velocity_enable = velocity_enable
        self.fix_phrase_bar_num = fix_phrase_bar_num

        self.vocab = RemiVocab(velocity_enable=self.velocity_enable)

        self.all_files_bpe_ids = []
        self.all_files_contexts = []

        self.load_data(self.file_names)

    def __len__(self):
        return len(self.all_files_contexts)


    def __getitem__(self, item):
        file_id, start, end = self.all_files_contexts[item]
        remi_raw_slice = self.all_files_bpe_ids[file_id][start:end]

        remi_raw_slice = self.mask_and_delete_tokens(remi_raw_slice)[:self.max_seq_len]
        remi_raw_slice = [BOS_TOKEN]+remi_raw_slice+[EOS_TOKEN]

        src = torch.tensor(self.vocab.encode(remi_raw_slice), dtype=torch.long)

        x = {
            'seq': src,
            'file_name': self.file_names[file_id] + '_' + f'from-{start}-to-{end}',
        }

        return x

    def load_data(self, file_names):
        for file_name in tqdm(file_names):
            if not self.server_mode:
                music_info = pickle.load(open(file_name, 'rb'))
            else:
                each_file_name = SERVER_DATA_DIR + file_name.split('Dataset')[-1]
                music_info = pickle.load(open(each_file_name, 'rb'))

            music_info = reduce_music_info(music_info)
            remi_raw = REMI_Plus_Raw(music_info=music_info,
                                     chord_enable=self.chord_enable,
                                     velocity_enable=self.velocity_enable,
                                     fix_phrase_bar_num=self.fix_phrase_bar_num).get_final_sequence()

            bars, _ = self.get_bars_old(remi_raw)
            contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(remi_raw))]

            # (file_id, start, end)
            for start, end in contexts:
                if len(remi_raw[start:end]) > 3:
                    self.all_files_contexts.append((len(self.all_files_bpe_ids), start, end))

            self.all_files_bpe_ids.append(remi_raw)

    def get_bars_old(self, events, include_ids=True):
        bars = [i for i, event in enumerate(events) if 'Bar_' in event]
        if include_ids:
            bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
            bar_ids = torch.cumsum(bar_ids, dim=0)

            return bars, bar_ids
        else:
            return bars

    def mask_and_delete_tokens(self, events):
        # mask 'Bar_XXX', delete 'Phrase_XXX', 'BCD_XXX'
        events_post = []
        for event in events:
            if 'Bar_' in event:
                events_post.append(MASK_TOKEN)
            elif 'Phrase_' in event or 'BCD_' in event:
                continue
            else:
                events_post.append(event)
        return events_post

    @classmethod
    def collate(cls, data):
        batch = {}
        seq = [sample['seq'] for sample in data]
        seq = pad_sequence(seq, batch_first=True, padding_value=0)

        tmp = seq[:, :-1]
        target = seq[:, 1:].clone().detach()
        seq = tmp

        batch['input_seq'] = seq
        batch['target_seq'] = target

        batch['file_name'] = [sample['file_name'] for sample in data]

        batch['mask'] = (target != 0).long()

        return batch



class VQVAE_Dataset_Track(data.Dataset):
    def __init__(self, file_names,
                 max_seq_len=1024,
                 use_augmentation=False,
                 chord_enable=False,
                 velocity_enable=False,   
                 fix_phrase_bar_num=None,
                 server_mode=False,
                 ):

        self.file_names = file_names
        self.max_seq_len = max_seq_len - 1

        self.use_augmentation = use_augmentation
        self.server_mode = server_mode

        self.chord_enable = chord_enable
        self.velocity_enable = velocity_enable
        self.fix_phrase_bar_num = fix_phrase_bar_num

        self.vocab = RemiVocab(velocity_enable=self.velocity_enable)

        self.all_files_bpe_ids = []
        self.all_files_contexts = []

        self.load_data(self.file_names)

    def __len__(self):
        return len(self.all_files_contexts)

    def __getitem__(self, item):
        file_track_id, start, end, track_name, file_name = self.all_files_contexts[item]

        remi_track_slice = self.all_files_bpe_ids[file_track_id][start:end]

        remi_track_slice = self.mask_and_delete_tokens(remi_track_slice)[:self.max_seq_len]
        remi_track_slice = [track_name] + remi_track_slice + [EOS_TOKEN]

        src = torch.tensor(self.vocab.encode(remi_track_slice), dtype=torch.long)

        x = {
            'seq': src,
            'file_name': file_name + '_' + f'from-{start}-to-{end}' + '_' + track_name,
        }

        return x

    def load_data(self, file_names):
        for file_name in tqdm(file_names):
            if not self.server_mode:
                music_info = pickle.load(open(file_name, 'rb'))
            else:
                each_file_name = SERVER_DATA_DIR + file_name.split('Dataset')[-1]
                music_info = pickle.load(open(each_file_name, 'rb'))

            remi_track = REMI_Plus_Raw(music_info=music_info,
                                       chord_enable=self.chord_enable,
                                       velocity_enable=self.velocity_enable,
                                       fix_phrase_bar_num=self.fix_phrase_bar_num).get_final_sequence(tracks_mode=True)

            track_status = music_info['TRACK_STATUS']

            for idx, each_track in enumerate(remi_track):
                if track_status[idx]:
                    bars, _ = self.get_bars_old(each_track)
                    contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(each_track))]

                    for start, end in contexts:
                        if len(each_track[start:end]) > 3:
                            self.all_files_contexts.append((
                                len(self.all_files_bpe_ids),  
                                start,
                                end,
                                each_track[0],  # track_name
                                file_name,
                            ))
                    self.all_files_bpe_ids.append(each_track)

    def get_bars_old(self, events, include_ids=True):
        bars = [i for i, event in enumerate(events) if 'Bar_' in event]
        if include_ids:
            bar_ids = torch.bincount(torch.tensor(bars, dtype=torch.int), minlength=len(events))
            bar_ids = torch.cumsum(bar_ids, dim=0)

            return bars, bar_ids
        else:
            return bars

    def mask_and_delete_tokens(self, events):
        # mask 'Bar_XXX', delete 'Phrase_XXX', 'BCD_XXX'
        events_post = []
        for event in events:
            if 'Bar_' in event:
                events_post.append(MASK_TOKEN)
            elif 'Phrase_' in event or 'BCD_' in event:
                continue
            else:
                events_post.append(event)
        return events_post

    @classmethod
    def collate(cls, data):
        batch = {}
        seq = [sample['seq'] for sample in data]
        seq = pad_sequence(seq, batch_first=True, padding_value=0)

        tmp = seq[:, :-1]
        target = seq[:, 1:].clone().detach()
        seq = tmp

        batch['input_seq'] = seq
        batch['target_seq'] = target

        batch['file_name'] = [sample['file_name'] for sample in data]

        batch['mask'] = (target != 0).long()

        return batch
