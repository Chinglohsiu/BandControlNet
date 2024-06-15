
# Pop_Single_Dataset
# Pop_Multi_Dataset

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence

from representation_multiple_v2 import REMI_Plus_Raw
from BPE_tokenizer_v2 import Musicbpe
from vocab_v2 import RemiVocab
from vocab_v2 import DescriptionVocab_Bar, DescriptionVocab_DPT, DescriptionVocab_DND, \
    DescriptionVocab_ND, DescriptionVocab_MP, DescriptionVocab_MD, DescriptionVocab_MV, DescriptionVocab_Chord, \
    MMTVocab_instrument

from constants import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN, Ins_LIST
)

from utils.dataset_utils import get_bars, delete_phrase_tokens2, padding_prior, reduce_music_info

import pickle
import os
import random

SERVER_DATA_DIR = '/home/data/music_gen_cl/Bandformer_new'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RAW_LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH',
                                  os.path.join(os.getenv('SCRATCH', os.getenv('TMPDIR', './tempmv')), 'raw_latent_gpu'))
TRACK_LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH',
                                    os.path.join(os.getenv('SCRATCH', os.getenv('TMPDIR', './tempmv')), 'track_latent_gpu'))

TRACK_NUMS = 4

class CustomSampler(data.Sampler):
    def __init__(self, dataset, num_rates=2):
        self.dataset = dataset
        self.num_rates = num_rates
        self.num_samples = len(self.dataset) // self.num_rates

    def __iter__(self):
        n = len(self.dataset)
        return iter(torch.randperm(n).tolist()[:self.num_samples])

    def __len__(self):
        return self.num_samples

class Pop_Single_Dataset(data.Dataset):
    def __init__(self, file_names,
                 bpe_model_path='../Dataset/BPE_model_64',
                 max_seq_len=3584,  
                 max_bars=64,
                 server_mode=False,

                 use_augmentation=False,
                 chord_enable=False,
                 velocity_enable=True,
                 fix_phrase_bar_num=None,
                 ):

        self.file_names = file_names
        self.max_seq_len = max_seq_len - 1  # 外加<bos>长度，无<eos>
        self.max_bars = max_bars
        self.server_mode = server_mode

        self.use_augmentation = use_augmentation
        self.chord_enable = chord_enable
        self.velocity_enable = velocity_enable
        self.fix_phrase_bar_num = fix_phrase_bar_num

        self.bpe = Musicbpe(vocab_type='remi_raw', bpe_vocab_size=10000)
        self.bpe.load_model(bpe_model_path)

        self.drums_pt_vocab = DescriptionVocab_DPT()
        self.drums_nd_vocab = DescriptionVocab_DND()

        self.nd_vocab = DescriptionVocab_ND()
        self.mp_vocab = DescriptionVocab_MP()
        self.md_vocab = DescriptionVocab_MD()
        self.mv_vocab = DescriptionVocab_MV()

        self.chords_feat_vocab = DescriptionVocab_Chord()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        if not self.server_mode:
            music_info = pickle.load(open(self.file_names[item], 'rb'))
        else:
            each_file_name = SERVER_DATA_DIR + self.file_names[item].split('Dataset')[-1]
            music_info = pickle.load(open(each_file_name, 'rb'))

        music_info = reduce_music_info(music_info)
        remi = REMI_Plus_Raw(music_info=music_info,
                             chord_enable=self.chord_enable,
                             velocity_enable=self.velocity_enable,
                             fix_phrase_bar_num=self.fix_phrase_bar_num
                             )
        remi_raw = remi.get_final_sequence()
        remi_raw = delete_phrase_tokens2(remi_raw)

        # ==================================================================
        # event_ids
        bpe_encoded_event_ids = self.bpe.apply_bpe(remi_raw)[:self.max_seq_len]
        # 加 '<bos>'
        bpe_encoded_event_ids = self.bpe.apply_bpe([BOS_TOKEN]) + bpe_encoded_event_ids
        seq = torch.tensor(bpe_encoded_event_ids, dtype=torch.long)

        # 置换tokens list
        bpe_encoded_bytes = [self.bpe.bpe_model.id_to_token(id_) for id_ in bpe_encoded_event_ids]
        bpe_encoded_tokens = [self.bpe.bpe_bytes_to_tokens_dict[byte_] for byte_ in bpe_encoded_bytes]

        # ====== get bar-related (only bar info) => length, index, embeded_ids======
        bar_len = torch.tensor(len(remi.groups), dtype=torch.long)
        _, bar_ids, bar_embed_ids = get_bars(bpe_encoded_tokens)

        # =========== extract feature seq, chord seq, vq_codes seq (single_seq mode) ==========
        remi4desc = REMI_Plus_Raw(music_info=music_info,
                                  chord_enable=True,
                                  velocity_enable=self.velocity_enable,
                                  fix_phrase_bar_num=self.fix_phrase_bar_num,
                                  )


        features_seq_encoded, chords_seq_encoded = self.get_descriptions_raw(remi4desc)
        vq_codes = self.get_vq_codes_raw(os.path.basename(self.file_names[item]))  

        x = {
            'seq': seq,  
            'bar_len': bar_len,  
            'bar_ids': bar_ids,  
            'bar_embed_ids': bar_embed_ids,  

            'chords_seq': padding_prior(chords_seq_encoded, self.max_bars),  
            'feat_seq': padding_prior(features_seq_encoded, self.max_bars),  
            'vq_codes': padding_prior(vq_codes, self.max_bars),  

            'file_name': self.file_names[item],
            'track_status': music_info['TRACK_STATUS'],
        }

        return x

    def get_descriptions_raw(self, remi):
        # remi_raw中提取features_seq, chords_seq
        assert remi.velocity_enable == True
        assert remi.chord_enable == True

        bar_groups = remi.get_new_groups(cover_level_type='bar')
        descriptions = remi.get_description_raw(bar_groups)

        features_seq = descriptions[0][1:]
        chords_seq = descriptions[1][1:]

        # features_seq每小节的特征为
        # ['Bar_XXX', 'Drums_Pitch_Type_XXX', 'Drums_Note_Density_XXX',
        # 'Note_Density_XXX', 'Mean_Pitch_XXX', 'Mean_Duration_XXX', 'Mean_Velocity_XXX']
        assert len(features_seq) == len(bar_groups) * 7
        assert len(chords_seq) == len(bar_groups) * 5

        # 按小节划分features_seq, chords_seq
        bars_feat = [i * 7 for i in range(len(bar_groups) + 1)]
        bars_chord = [i * 5 for i in range(len(bar_groups) + 1)]

        context_feat = list(zip(bars_feat[:-1], bars_feat[1:]))
        context_chord = list(zip(bars_chord[:-1], bars_chord[1:]))

        # 已忽略掉'Bar_XXX', features_seq_grouped 6维, chords_seq_grouped 4维
        features_seq_grouped = [features_seq[start + 1:end] for start, end in context_feat]
        chords_seq_grouped = [chords_seq[start + 1:end] for start, end in context_chord]

        # features_seq_encoded: (bar_nums, 6)
        feat_vocab_list = [
            self.drums_pt_vocab, self.drums_nd_vocab,
            self.nd_vocab, self.mp_vocab, self.md_vocab, self.mv_vocab,
        ]
        features_seq_encoded = [self.encoding_attr(features_seq_grouped, feat_vocab_list[attr_id], attr_id)
                                for attr_id in range(6)
                                ]
        features_seq_encoded = torch.stack(features_seq_encoded, dim=1)

        # chords_seq_encoded: (bar_nums, 4)
        chords_vocab_list = [
            self.chords_feat_vocab, self.chords_feat_vocab, self.chords_feat_vocab, self.chords_feat_vocab
        ]
        chords_seq_encoded = [self.encoding_attr(chords_seq_grouped, chords_vocab_list[attr_id], attr_id)
                              for attr_id in range(4)
                              ]
        chords_seq_encoded = torch.stack(chords_seq_encoded, dim=1)

        # features_seq_encoded: (bar_nums, 6)
        # chords_seq_encoded:  (bar_nums, 4)
        return features_seq_encoded, chords_seq_encoded

    def encoding_attr(self, events, vocab, dim_id):
        events_attr = [event[dim_id] for event in events]
        encoded_attr = torch.tensor(vocab.encode(events_attr), dtype=torch.long)
        return encoded_attr

    def get_vq_codes_raw(self, name):
        latent_cache_path = os.path.join(RAW_LATENT_CACHE_PATH, str(self.max_bars))
        cache_file = os.path.join(latent_cache_path, name)
        vq_codes = pickle.load(open(cache_file, 'rb'))

        return vq_codes

    @classmethod
    def collate(cls, data):
        batch = {}
        seq = [sample['seq'] for sample in data]
        seq = pad_sequence(seq, batch_first=True, padding_value=0)

        tmp = seq[:, :-1]
        target = seq[:, 1:].clone().detach()
        seq = tmp

        seq_len = seq.size(1)

        # input & target seq, (bs, seq_len)
        batch['input_seq'] = seq
        batch['target_seq'] = target
        batch['mask'] = (target != 0).long()

        # bar_len, (bs, )
        batch['bar_len'] = torch.stack([sample['bar_len'] for sample in data], dim=0)
        # bar_ids, (bs, 64)
        batch['bar_ids'] = torch.stack([sample['bar_ids'] for sample in data], dim=0)
        # bar_embed_ids, (bs, seq_len)
        bar_embed_ids = [sample['bar_embed_ids'] for sample in data]
        bar_embed_ids = pad_sequence(bar_embed_ids, batch_first=True, padding_value=0)
        batch['bar_embed_ids'] = bar_embed_ids[:, :seq_len]

        # chord, (bs, max_bar_nums, 4)
        chords_seq = [sample['chords_seq'] for sample in data]
        chords_seq = pad_sequence(chords_seq, batch_first=True, padding_value=0)
        batch['chords_seq'] = chords_seq

        # feat, (bs, max_bar_nums, 6)
        feat_seq = [sample['feat_seq'] for sample in data]
        feat_seq = pad_sequence(feat_seq, batch_first=True, padding_value=0)
        batch['feat_seq'] = feat_seq

        # vq_codes, (bs, max_bar_nums, 8)
        vq_codes = [sample['vq_codes'] for sample in data]
        vq_codes = pad_sequence(vq_codes, batch_first=True, padding_value=0)
        batch['vq_codes'] = vq_codes

        # file_name
        batch['file_name'] = [sample['file_name'] for sample in data]

        return batch


class Pop_Multi_Dataset(data.Dataset):
    def __init__(self, file_names,
                 bpe_model_path='../Dataset/BPE_model_64',
                 max_seq_len=2048,  
                 max_bars=64,
                 server_mode=False,

                 use_augmentation=False,
                 chord_enable=False,
                 velocity_enable=True,
                 fix_phrase_bar_num=None,
                 ):
        self.file_names = file_names
        self.max_seq_len = max_seq_len - 1  # 外加<bos>长度，无<eos>
        self.max_bars = max_bars
        self.server_mode = server_mode

        self.use_augmentation = use_augmentation
        self.chord_enable = chord_enable
        self.velocity_enable = velocity_enable
        self.fix_phrase_bar_num = fix_phrase_bar_num

        self.bpe = Musicbpe(vocab_type='remi_track', bpe_vocab_size=10000)
        self.bpe.load_model(bpe_model_path)

        self.drums_pt_vocab = DescriptionVocab_DPT()
        self.drums_nd_vocab = DescriptionVocab_DND()

        self.nd_vocab = DescriptionVocab_ND()
        self.mp_vocab = DescriptionVocab_MP()
        self.md_vocab = DescriptionVocab_MD()
        self.mv_vocab = DescriptionVocab_MV()

        self.chords_feat_vocab = DescriptionVocab_Chord()

        self.ins_vocab = MMTVocab_instrument()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, item):
        if not self.server_mode:
            music_info = pickle.load(open(self.file_names[item], 'rb'))
        else:
            each_file_name = SERVER_DATA_DIR + self.file_names[item].split('Dataset')[-1]
            music_info = pickle.load(open(each_file_name, 'rb'))

        remi = REMI_Plus_Raw(music_info=music_info,
                             chord_enable=self.chord_enable,
                             velocity_enable=self.velocity_enable,
                             fix_phrase_bar_num=self.fix_phrase_bar_num)
        remi_track = remi.get_final_sequence(tracks_mode=True)
        remi_track = [delete_phrase_tokens2(event_track) for event_track in remi_track]

        # 对超过4个轨道的，删除较短的轨道，修改track_status，保持轨道数量始终为4
        track_status = torch.tensor(music_info['TRACK_STATUS'], dtype=torch.bool)
        track_status[0] = True

        remi_track_length = []
        for track_i, track in enumerate(remi_track):
            if track_i in [0, 5] or track_status[track_i] == False:
                remi_track_length.append(-4096)
            else:
                remi_track_length.append(-len(track))

        remi_track_length = torch.tensor(remi_track_length, dtype=torch.long)

        k = torch.sum(track_status) - TRACK_NUMS
        if k > 0:
            skip_track_index = torch.topk(remi_track_length, k)[-1]
            track_status[skip_track_index] = False

        # ========== get track_name ==========
        assert torch.sum(track_status) == TRACK_NUMS
        track_name = self.get_4track_name(remi_track, track_status)
        assert len(track_name) == TRACK_NUMS
        track_name = torch.tensor(self.ins_vocab.encode(track_name), dtype=torch.long)

        # ====== get bar-related info => length, index, embeded_ids======
        bar_len = torch.tensor(len(remi.groups), dtype=torch.long)

        bpe_encoded_track_ids = self.bpe.apply_bpe(remi_track)
        # 每个轨道上添加<bos>
        bpe_encoded_track_ids = self.add_bos(bpe_encoded_track_ids)

        seq = []
        bar_ids = []
        bar_embed_ids = []
        for i, bpe_encoded_ids in enumerate(bpe_encoded_track_ids):
            if track_status[i]:
                seq_each = torch.tensor(bpe_encoded_ids, dtype=torch.long)
                seq.append(seq_each)

                bpe_encoded_bytes_track = [self.bpe.bpe_model.id_to_token(id_) for id_ in bpe_encoded_ids]
                bpe_encoded_tokens_track = [self.bpe.bpe_bytes_to_tokens_dict[byte_] for byte_ in bpe_encoded_bytes_track]

                _, bar_ids_track, bar_embed_ids_track = get_bars(bpe_encoded_tokens_track)
                bar_ids.append(bar_ids_track)
                bar_embed_ids.append(bar_embed_ids_track)
        # padding seq, bar_embed_ids, phrase_embed_ids => (max_seq_len_within_on_file, 4)
        seq = pad_sequence(seq, batch_first=False, padding_value=0)
        bar_embed_ids = pad_sequence(bar_embed_ids, batch_first=False, padding_value=0)
        bar_ids = torch.stack(bar_ids, dim=1)

        # ========== extract feature seq, chord seq, vq seq ==========
        remi4desc = REMI_Plus_Raw(music_info=music_info,
                                  chord_enable=True,
                                  velocity_enable=self.velocity_enable,
                                  fix_phrase_bar_num=self.fix_phrase_bar_num,
                                  )
        features_seq_encoded, chords_seq_encoded = self.get_descriptions_track(remi4desc, track_status) # (bar_nums, 4, TRACK_NUMS)
        vq_codes = self.get_vq_codes_track(os.path.basename(self.file_names[item]), track_status)  # (bar_nums, 8, TRACK_NUMS)

        x = {
            'seq': seq,  
            'bar_len': bar_len,  
            'bar_ids': bar_ids,  
            'bar_embed_ids': bar_embed_ids,  

            'track_status': track_status,  # (6, ), 是否空轨, bool
            'track_name': track_name,  # (4, ), encoded instrument

            'chords_seq': padding_prior(chords_seq_encoded, self.max_bars),  
            'feat_seq': padding_prior(features_seq_encoded, self.max_bars),  
            'vq_codes': padding_prior(vq_codes, self.max_bars),              

            'file_name': self.file_names[item],
        }

        return x

    def get_4track_name(self, remi_track, track_status):
        ins_name_list = []
        for i, track_events in enumerate(remi_track):
            if track_status[i]:
                ins_name_list.append(track_events[0])
        return ins_name_list

    def add_bos(self, events):
        # 给每个轨道开头添加<bos>
        events_added = []
        for track in events:
            track = track[:self.max_seq_len]
            track_added = self.bpe.apply_bpe([[BOS_TOKEN]])[0] + track
            events_added.append(track_added)
        return events_added

    def get_descriptions_track(self, remi, track_status):
        # remi_track中提取features_seq, chords_seq
        assert remi.velocity_enable == True
        assert remi.chord_enable == True

        bar_groups = remi.get_new_groups(cover_level_type='bar')
        descriptions = remi.get_description_track(bar_groups)

        track4_index = torch.nonzero(track_status).squeeze(1).tolist()
        assert len(track4_index) == TRACK_NUMS

        drums_features_seq = descriptions[track4_index[0]][1:]
        harmony1_features_seq = descriptions[track4_index[1]][1:]
        harmony2_features_seq = descriptions[track4_index[2]][1:]
        melody_features_seq = descriptions[track4_index[3]][1:]

        chords_seq = descriptions[-1][1:]

        # drums_features_seq每小节特征为
        # ['Bar_XXX', 'Drums_Pitch_Type_XXX', 'Drums_Note_Density_XXX'], 3维
        # 鼓轨之外features_seq每小节特征为
        # ['Bar_XXX', 'Note_Density_XXX', 'Mean_Pitch_XXX', 'Mean_Duration_XXX', 'Mean_Velocity_XXX'], 5维
        # 其中'Bar_XXX'都可忽略
        assert len(drums_features_seq) == len(bar_groups) * 3
        assert len(melody_features_seq) == len(bar_groups) * 5
        assert len(chords_seq) == len(bar_groups) * 5

        # 按小节划分features_seq, chords_seq
        bars_drums_feat = [i * 3 for i in range(len(bar_groups) + 1)]
        bars_others = [i * 5 for i in range(len(bar_groups) + 1)]

        context_drums_feat = list(zip(bars_drums_feat[:-1], bars_drums_feat[1:]))
        context_others = list(zip(bars_others[:-1], bars_others[1:]))

        # 已忽略掉'Bar_XXX', drums_features_seq_grouped 2维, 其余4维
        drums_features_seq_grouped = [drums_features_seq[start + 1:end] for start, end in context_drums_feat]
        harmony1_features_seq_grouped = [harmony1_features_seq[start + 1:end] for start, end in context_others]
        harmony2_features_seq_grouped = [harmony2_features_seq[start + 1:end] for start, end in context_others]
        melody_features_seq_grouped = [melody_features_seq[start + 1:end] for start, end in context_others]

        chords_seq_grouped = [chords_seq[start + 1:end] for start, end in context_others]

        # drums_features_seq_encoded: (bar_nums, 2)
        drums_feat_vocab_list = [self.drums_pt_vocab, self.drums_nd_vocab]
        drums_features_seq_encoded = self.get_encoded_features(drums_features_seq_grouped, drums_feat_vocab_list, 2)
        # padding 0 到4维
        drums_features_seq_encoded = F.pad(drums_features_seq_encoded, (0, 2), 'constant', 0)

        # (bar_nums, 4)
        other_feat_vocab_list = [self.nd_vocab, self.mp_vocab, self.md_vocab, self.mv_vocab]
        harmony1_features_seq_encoded = self.get_encoded_features(harmony1_features_seq_grouped, other_feat_vocab_list,
                                                                  4)
        harmony2_features_seq_encoded = self.get_encoded_features(harmony2_features_seq_grouped, other_feat_vocab_list,
                                                                  4)
        melody_features_seq_encoded = self.get_encoded_features(melody_features_seq_grouped, other_feat_vocab_list, 4)

        # (bar_nums, 4)
        chords_vocab_list = [self.chords_feat_vocab] * 4
        chords_seq_encoded = self.get_encoded_features(chords_seq_grouped, chords_vocab_list, 4)

        features_seq_encodes = torch.stack([drums_features_seq_encoded,
                                            harmony1_features_seq_encoded,
                                            harmony2_features_seq_encoded,
                                            melody_features_seq_encoded,
                                            ], dim=2)
        # features_seq_encodes: (bar_nums, 4, 4), 其中drums_features_seq_encoded后两位为padding 0
        # chords_seq_encoded: (bar_nums, 4)
        return features_seq_encodes, chords_seq_encoded

    def encoding_attr(self, events, vocab, dim_id):
        events_attr = [event[dim_id] for event in events]
        encoded_attr = torch.tensor(vocab.encode(events_attr), dtype=torch.long)
        return encoded_attr

    def get_encoded_features(self, features_groups, features_vocab, D):
        encoded_features = [self.encoding_attr(features_groups, features_vocab[attr_id], attr_id) for attr_id in
                            range(D)]
        encoded_features = torch.stack(encoded_features, dim=1)
        return encoded_features

    def get_vq_codes_track(self, name, track_status):
        latent_cache_path = os.path.join(TRACK_LATENT_CACHE_PATH, str(self.max_bars))
        cache_file = os.path.join(latent_cache_path, name)
        vq_codes = pickle.load(open(cache_file, 'rb'))

        return vq_codes[:, :, track_status]

    @classmethod
    def collate(cls, data):
        batch = {}
        seq = [sample['seq'] for sample in data]
        # (bs, max_seq_len, 4)
        seq = pad_sequence(seq, batch_first=True, padding_value=0)

        tmp = seq[:, :-1]
        target = seq[:, 1:].clone().detach()
        seq = tmp

        seq_len = seq.size(1)

        # input & target seq, (bs, seq_len, 4)
        batch['input_seq'] = seq
        batch['target_seq'] = target
        batch['mask'] = (target != 0).long()

        # bar_len, (bs,)
        batch['bar_len'] = torch.stack([sample['bar_len'] for sample in data], dim=0)
        # bar_ids, (bs, 64, 4)
        batch['bar_ids'] = torch.stack([sample['bar_ids'] for sample in data], dim=0)
        # bar_embed_ids, (bs, max_seq_len, 4)
        bar_embed_ids = [sample['bar_embed_ids'] for sample in data]
        bar_embed_ids = pad_sequence(bar_embed_ids, batch_first=True, padding_value=0)
        batch['bar_embed_ids'] = bar_embed_ids[:, :seq_len]

        # track_status: (bs, 6), track_name: (bs, 4)
        batch['track_status'] = torch.stack([sample['track_status'] for sample in data], dim=0)
        batch['track_name'] = torch.stack([sample['track_name'] for sample in data], dim=0)

        # chords, (bs, max_bar_nums, 4), 不区分轨道
        chords_seq = [sample['chords_seq'] for sample in data]
        chords_seq = pad_sequence(chords_seq, batch_first=True, padding_value=0)
        batch['chords_seq'] = chords_seq

        # feat, (bs, max_bar_nums, 4, 4), drums_feat即feat_seq[:, :, :2, 0]只有前两维有效，后两维为padding 0
        feat_seq = [sample['feat_seq'] for sample in data]
        feat_seq = pad_sequence(feat_seq, batch_first=True, padding_value=0)
        batch['feat_seq'] = feat_seq
        # vq_codes, (bs, max_bar_nums, 8, 4)
        vq_codes = [sample['vq_codes'] for sample in data]
        vq_codes = pad_sequence(vq_codes, batch_first=True, padding_value=0)
        batch['vq_codes'] = vq_codes

        # file_name
        batch['file_name'] = [sample['file_name'] for sample in data]

        return batch
