
# only infer on single-gpu, fix the generated vq-codes
# similar with ../model_v2/get_vq_codes.py

# velocity_enable = True
import os
import pickle
import argparse
import json
from tqdm import tqdm

import torch

from representation_multiple_v2 import REMI_Plus_Raw
from vocab_v2 import RemiVocab, DescriptionVocab_Chord
from VQ_VAE_Model import VQVAE
from constants import (
    PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN,
    )

from utils.dataset_utils import reduce_music_info
import joblib

ADD_TOKENS_DICT = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    BOS_TOKEN: 2,
    EOS_TOKEN: 3,
    MASK_TOKEN: 4,
}

RAW_LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH', os.path.join(os.getenv('SCRATCH', os.getenv('TMPDIR', './tempmv')), 'raw_latent_gpu'))
TRACK_LATENT_CACHE_PATH = os.getenv('LATENT_CACHE_PATH', os.path.join(os.getenv('SCRATCH', os.getenv('TMPDIR', './tempmv')), 'track_latent_gpu'))

SERVER_DATA_DIR = '/home/data/music_gen_cl/Bandformer_new'

def mask_and_delete_tokens(events):
    # mask 'Bar_XXX', delete 'Phrase_XXX', 'BCD_XXX'
    events_post = []
    for event in events:
        if 'Bar_' in event:
            events_post.append('<mask>')
        elif 'Phrase_' in event or 'BCD_' in event: #or 'Velocity_' in event:
            continue
        else:
            events_post.append(event)
    return events_post

def load_model(token_type):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args_dict = vars(args)
    args_file = os.path.join('./ckpt/vqvae_model_mv/', f'{token_type}_bar_level_64_args.json')

    with open(args_file, 'rt') as f:
        args_dict.update(json.load(f))

    vq_vocab = RemiVocab(velocity_enable=True)
    vq_model = VQVAE(args, len(vq_vocab))
    resume_dict = torch.load(os.path.join('./ckpt/vqvae_model_mv/', f'{token_type}_bar_level_64_checkpoint.pt'))[
        'state_dict']
    vq_model.load_state_dict(resume_dict)
    vq_model.to(torch.device('cuda'))
    vq_model.eval()
    return vq_model, vq_vocab


def get_raw_vq_codes(vq_model, vq_vocab, file_name, max_bars=64, n_groups=8, server_mode=True):
    save_path = os.path.join(RAW_LATENT_CACHE_PATH, str(max_bars))
    os.makedirs(save_path, exist_ok=True)


    if not server_mode:
        music_info = pickle.load(open(file_name, 'rb'))
    else:
        each_file_name = SERVER_DATA_DIR + file_name.split('Dataset')[-1]
        music_info = pickle.load(open(each_file_name, 'rb'))

    music_info = reduce_music_info(music_info)
    
    name = os.path.basename(file_name)

    # === processing vq codes-info ===
    remi = REMI_Plus_Raw(music_info)
    remi_raw = remi.get_final_sequence()

    bars = [i for i, event in enumerate(remi_raw) if 'Bar_' in event]
    contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(remi_raw))]

    codes = []
    for start, end in contexts:
        if len(remi_raw[start:end]) > 3:
            remi_raw_slice = mask_and_delete_tokens(remi_raw[start:end])[:255]
            remi_raw_slice = ['<bos>'] + remi_raw_slice
            remi_raw_slice_ids = torch.tensor(vq_vocab.encode(remi_raw_slice), dtype=torch.long).unsqueeze(0).to(
                torch.device('cuda'))

            out = vq_model.encode(remi_raw_slice_ids)
            # 将vq_codes后移5位，给padding等token空出位置
            vq_codes = out['codes'].cpu() + len(ADD_TOKENS_DICT)

            codes.append(vq_codes)
        else:
            assert len(remi_raw[start:end]) == 3
            # 用mask_token(id=4)
            codes.append(torch.ones(1, n_groups).long().cpu() * ADD_TOKENS_DICT[MASK_TOKEN])
    vq_codes = torch.cat(codes)
    pickle.dump(vq_codes, open(os.path.join(save_path, name), 'wb'))

def get_track_vq_codes(vq_model, vq_vocab, file_name, max_bars=64, n_groups=8, server_mode=True):
    save_path = os.path.join(TRACK_LATENT_CACHE_PATH, str(max_bars))
    os.makedirs(save_path, exist_ok=True)

    if not server_mode:
        music_info = pickle.load(open(file_name, 'rb'))
    else:
        each_file_name = SERVER_DATA_DIR + file_name.split('Dataset')[-1]
        music_info = pickle.load(open(each_file_name, 'rb'))

    name = os.path.basename(file_name)

    # === processing vq codes-info ===
    remi = REMI_Plus_Raw(music_info=music_info)
    remi_track = remi.get_final_sequence(tracks_mode=True)

    codes = []
    for idx, each_track in enumerate(remi_track):
        codes_track = []
        bars = [i for i, event in enumerate(each_track) if 'Bar_' in event]
        contexts = list(zip(bars[:-1], bars[1:])) + [(bars[-1], len(each_track))]

        for start, end in contexts:
            if len(each_track[start:end]) > 3:
                each_track_slice = mask_and_delete_tokens(each_track[start:end])[:127]
                each_track_slice = [each_track[0]] + each_track_slice
                each_track_slice_ids = torch.tensor(vq_vocab.encode(each_track_slice),
                                                    dtype=torch.long).unsqueeze(0).to(torch.device('cuda'))
                out = vq_model.encode(each_track_slice_ids)
                # 将vq_codes后移5位，给padding等token空出位置
                vq_codes = out['codes'].cpu() + len(ADD_TOKENS_DICT)
                codes_track.append(vq_codes)
            else:
                assert len(each_track[start:end]) == 3
                codes_track.append(torch.ones(1, n_groups).long().cpu() * ADD_TOKENS_DICT[MASK_TOKEN])
        codes.append(torch.cat(codes_track))
    # (bar_nums, 16, 6)
    vq_codes = torch.stack(codes, dim=2)
    pickle.dump(vq_codes, open(os.path.join(save_path, name), 'wb'))

def run_vq_codes(token_type, max_bars=64, n_groups=8, mode='all', server_mode=True):
    # mode: ['all', 'test']
    if not server_mode:
        data_dir = '../Dataset/Final_Data_Path_1127/'
    else:
        data_dir = os.path.join(SERVER_DATA_DIR, 'Final_Data_Path_1127/')

    paths = os.path.join(data_dir, 'data_{}_path_20231127_new.pkl'.format(max_bars))
    all_dataset_paths = pickle.load(open(paths, 'rb'))

    if mode == 'all':
        all_file_names = all_dataset_paths['training'] + all_dataset_paths['valid'] + all_dataset_paths['testing']
    else:
        all_file_names = all_dataset_paths['testing']

    print(f'total files: {len(all_file_names)}')
    if token_type == 'remi_raw':
        vq_model, vq_vocab = load_model(token_type)
        joblib.Parallel(n_jobs=16, verbose=5)(
            joblib.delayed(get_raw_vq_codes)(vq_model, vq_vocab,
                                             file_name=all_file_name, max_bars=max_bars, n_groups=n_groups, server_mode=server_mode)
            for all_file_name in all_file_names)
    else:
        assert token_type == 'remi_track'
        vq_model, vq_vocab = load_model(token_type)
        joblib.Parallel(n_jobs=16, verbose=5)(
            joblib.delayed(get_track_vq_codes)(vq_model, vq_vocab,
                                               file_name=all_file_name, max_bars=max_bars, n_groups=n_groups, server_mode=server_mode)
            for all_file_name in all_file_names)

parser_global = argparse.ArgumentParser()

parser_global.add_argument('--mode', type=str, default='all', help='test or all')
parser_global.add_argument('--token_type', type=str, default='remi_track', help='remi_raw or remi_track')
parser_global.add_argument('--max_bars', type=int, default=64, help='32 or 64')
parser_global.add_argument('--server_mode', type=bool, default=True, help='True or False')
parser_global.add_argument('--cuda_ids', type=str, default='1', help='gpu id on server')

if __name__ == '__main__':
    args = parser_global.parse_args()
    print_s = ''
    for arg in vars(args):
        s = '{}\t{}\n'.format(arg, getattr(args, arg))
        print_s += s

    if args.server_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    run_vq_codes(token_type=args.token_type, max_bars=args.max_bars, mode=args.mode, server_mode=args.server_mode)
    

