
import json
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import os
import errno
import pickle

import torch

from miditoolkit import MidiFile

from tokenizers import Tokenizer as TokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from representation_multiple_v2 import REMI_Plus_Raw
from vocab_v2 import RemiVocab


CHR_ID_START = 33
REMI_RAW_STRUCT_TOKENS = ['Bar_Normal', 'Bar_Empty', 'Phrase_Lower', 'Phrase_Upper'] + \
                         [f'BCD_{i}' for i in range(1, 17)] + \
                         [f'Position_{i}' for i in range(48)]
REMI_TRACK_STRUCT_TOKENS = REMI_RAW_STRUCT_TOKENS + ['Instrument_Drums', 'Instrument_Acoustic Grand Piano',
                                                     'Instrument_Acoustic Guitar (nylon)', 'Instrument_Acoustic Bass',
                                                     'Instrument_String Ensemble 1', 'Instrument_Lead 1 (square)']

class Musicbpe:
    def __init__(self,
                 vocab_type='remi_raw',
                 bpe_vocab_size=10000,
                 ):
        # vocab: 字典类型, 如: RemiVocab()
        # bpe_vocab_size
        self.vocab_type = vocab_type

        if vocab_type == 'remi_raw':
            self.vocab = RemiVocab()
            self.custom_specials = REMI_RAW_STRUCT_TOKENS
        if vocab_type == 'remi_track':
            self.vocab = RemiVocab()
            self.custom_specials = REMI_TRACK_STRUCT_TOKENS

        self.base_vocab_len = len(self.vocab)

        self.bpe_vocab_size = bpe_vocab_size

        # event_ids => bytes
        self.base_ids_to_bytes_dict = {i: chr(i + CHR_ID_START) for i in range(self.base_vocab_len)}

        # bytes => base token(str)
        self.base_byte_to_token = {chr(i + CHR_ID_START): self.vocab.to_s(i) for i in range(self.base_vocab_len)}

        self.bpe_model = None

        self.bpe_bytes_to_tokens_dict = {}

        self.bpe_status = False

    def load_tokens(self, file_path):
        with open(file_path) as file:
            return json.load(file)

    def ids_to_bytes(self, ids, as_one_str=False):
        if isinstance(ids[0], list):
            return [self.ids_to_bytes(item, as_one_str) for item in ids]
        bytes_ = [self.base_ids_to_bytes_dict[i] for i in ids]
        return ''.join(bytes_) if as_one_str else bytes_

    def get_ori_ids(self, ori_tokens):
        if self.vocab_type == 'remi_raw':
            return self.vocab.encode(ori_tokens)
        if self.vocab_type == 'remi_track':
            return [self.vocab.encode(token) for token in ori_tokens]


    def learn_bpe(self, tokens_path):
        # tokens_path, original samples' path
        # sample['ids'] => events_ids
        # sample['tokens'] => token in strings
        # sample['bytes'] => unique bytes, defaults=None

        if self.bpe_vocab_size <= self.base_vocab_len:
            print(
                f"vocab_size ({self.bpe_vocab_size}) need to be higher than the size of the current vocabulary "
                f"({self.base_vocab_len}). Skipping BPE training."
            )
            return

        iterator = []
        for file_path in tqdm(tokens_path, desc='Loading all token files'):
            # sample结构
            # [deleted_ids]: event_ids
            # [ori_tokens]: token in strings
            # [bytes]: unique bytes
            sample = self.load_tokens(file_path)
            bytes_ = self.ids_to_bytes(sample['deleted_ids'], as_one_str=True)
            iterator += (
                [[byte_] for byte_ in bytes_]
                if isinstance(sample['deleted_ids'][0], list)
                else [bytes_]
            )

        # initial vocab: bytes => bytes_ids
        voc_start = {chr(i + CHR_ID_START): i for i in range(self.base_vocab_len)}
        self.bpe_model = TokenizerFast(
            BPE(
                vocab=voc_start,
                merges=[],
                dropout=None,
                continuing_subword_prefix='',
                end_of_word_suffix='',
                fuse_unk=False,
            )
        )

        special_tokens_bytes = self.ids_to_bytes(
            self.vocab.encode(self.vocab.specials + self.custom_specials)
            # self.vocab.encode(self.vocab.specials)
        )

        trainer = BpeTrainer(
            vocab_size=self.bpe_vocab_size,
            special_tokens=special_tokens_bytes,
            show_progress=True,
        )

        self.bpe_model.train_from_iterator(iterator, length=sum(1 for _ in iterator), trainer=trainer)

        self.bpe_bytes_to_tokens_dict = {
            k: [self.base_byte_to_token[b] for b in k]
            for k in self.bpe_model.get_vocab()
        }

        self.bpe_status = True

    def apply_bpe(self, ori_tokens):
        # 单个token sample => bpe ids
        # sample['deleted_ids'] => events_ids
        # sample['ori_tokens'] => token in strings
        # sample['bytes'] => unique bytes, defaults=None
        if self.bpe_status:
            ori_ids = self.get_ori_ids(ori_tokens)
            token_bytes  = self.ids_to_bytes(ori_ids, as_one_str=True)

        if isinstance(ori_ids[0], list):
            bpe_encoded_tokens = self.bpe_model.encode_batch([[t] for t in token_bytes], is_pretokenized=True)
            return [bpe_tokens.ids for bpe_tokens in bpe_encoded_tokens]
        else:
            bpe_encoded_tokens = self.bpe_model.encode([token_bytes], is_pretokenized=True)
            return bpe_encoded_tokens.ids

    def decode_bpe(self, bpe_sample_ids):

        if isinstance(bpe_sample_ids[0], list):
            return [self.decode_bpe(bpe_sample_id) for bpe_sample_id in bpe_sample_ids]

        encoded_bytes = [self.bpe_model.id_to_token(id_) for id_ in bpe_sample_ids]
        decoded_tokens = [
            self.bpe_bytes_to_tokens_dict[byte_] for byte_ in encoded_bytes
        ]
        decoded_tokens = [
            item for sublist in decoded_tokens for item in sublist
        ]

        return decoded_tokens

    def save_model(self, model_paths):
        self.bpe_model.save(os.path.join(model_paths, self.vocab_type+'_bpe_model.json'))
        with open(os.path.join(model_paths, self.vocab_type+'_bpe_bytes_to_tokens_dict.json'), 'w') as file:
            json.dump(self.bpe_bytes_to_tokens_dict, file)

    def load_model(self, model_paths):
        self.bpe_model = TokenizerFast.from_file(os.path.join(model_paths, self.vocab_type+'_bpe_model.json'))
        self.bpe_status = True

        with open(os.path.join(model_paths, self.vocab_type+'_bpe_bytes_to_tokens_dict.json')) as file:
            self.bpe_bytes_to_tokens_dict = json.load(file)

def findall_endswith(postfix, root):
    """Traverse `root` recursively and yield all files ending with `postfix`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(postfix):
                yield os.path.join(dirpath, filename)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def change_prefix(path, src, dst):
    return os.path.join(dst, os.path.relpath(path, src))

# =============================================================
# =============================================================

def split_single_token_sequence(ori_tokens, struct_tokens):
    assert isinstance(ori_tokens[0], str)
    struct_id_list = []
    for token_idx, token in enumerate(ori_tokens):
        if token in struct_tokens:
            struct_id_list.append(token_idx)
    struct_id_list.append(len(ori_tokens))

    token_sequence_splited = []
    for start_id, end_id in zip(struct_id_list[:-1], struct_id_list[1:]):
        if end_id - start_id == 1:
            continue
        else:
            assert len(ori_tokens[start_id + 1:end_id]) > 0
            token_sequence_splited.append(ori_tokens[start_id + 1:end_id])
    return token_sequence_splited

# =============================================================
# =============================================================

def delete_struct_tokens(ori_tokens, struct_tokens, representations='remi_raw'):
    if representations == 'remi_raw':
        assert struct_tokens == REMI_RAW_STRUCT_TOKENS
        new_tokens = []
        for tokens in ori_tokens:
            if tokens not in struct_tokens:
                new_tokens.append(tokens)
        return new_tokens

    if representations == 'remi_track':
        assert struct_tokens == REMI_TRACK_STRUCT_TOKENS
        new_tokens = []
        for ori_track in ori_tokens:
            new_track = []
            for tokens in ori_track:
                if tokens not in struct_tokens:
                    new_track.append(tokens)
            # 空轨道不包含，其内部所含的Ins_XXX, Phrase_XXX, Bar_XXX均被删除
            if len(new_track) > 0:
                new_tokens.append(new_track)
        return new_tokens


def create_dpe_learning_dataset(path, ori_dir, out_dir, representations='remi_raw', chord_enable=False, velocity_enable=True):
    # 生成BPE训练的数据，即各表示方式下的原始表示序列
    # 默认无chord, 有velocity
    print(path)

    music_info = pickle.load(open(path, 'rb'))
    sample = {}
    if representations == 'remi_raw':
        vocab = RemiVocab()
        ori_tokens = REMI_Plus_Raw(music_info, chord_enable, velocity_enable).get_final_sequence()
        tokens_splited = split_single_token_sequence(ori_tokens, REMI_RAW_STRUCT_TOKENS)

        assert isinstance(tokens_splited[0][0], str)
        deleted_ids = [vocab.encode(token) for token in tokens_splited]
    elif representations == 'remi_track':
        vocab = RemiVocab()
        ori_tokens = REMI_Plus_Raw(music_info, chord_enable, velocity_enable).get_final_sequence(tracks_mode=True)

        tokens_splited = []
        for ori_tokens_track in ori_tokens:
            tokens_splited_track = split_single_token_sequence(ori_tokens_track, REMI_TRACK_STRUCT_TOKENS)
            tokens_splited += tokens_splited_track

        assert isinstance(tokens_splited[0][0], str)
        deleted_ids = [vocab.encode(token) for token in tokens_splited]
    else:
        raise ValueError('Check representation options.')

    sample['ori_tokens'] = ori_tokens
    sample['deleted_ids'] = deleted_ids
    sample['bytes'] = None

    out_path = change_prefix(os.path.dirname(path), ori_dir, out_dir)
    make_sure_path_exists(out_path)
    out_path = os.path.join(out_path, os.path.basename(path).split('.pkl')[0] + '.json')

    with open(out_path, 'w') as file:
        json.dump(sample, file)

