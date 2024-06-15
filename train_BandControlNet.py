

import pickle
import numpy as np
import os
import time
import json
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

from BandControlNet import BandControlNet
from Pop_dataset import Pop_Multi_Dataset

from utils.model_utils import get_lr_multiplier, recover_position_track, fix_seed, count_note_nums
from representation_multiple_v2 import remi_track2midi
from vocab_v2 import MMTVocab_instrument
from constants import Ins_LIST

from argparse import ArgumentParser, ArgumentTypeError

SERVER_DATA_DIR = '/home/data/music_gen_cl/Bandformer_new'

VOCAB_SIZE = {
    'seq': 10000,    # bpe
    'chords': 138,    # 5+133
    'vq_codes': 1024+5,

    'drums_pitch_type': 37,    # 5+32
    'drums_note_density': 55,    # 5+50

    'note_density': 71,    # 5+66
    'mean_pitch': 39,    # 5+34
    'mean_duration': 35,    # 5+30
    'mean_velocity': 39,    # 5+34

    'track': len(MMTVocab_instrument()),    # 5+6
}

TRACK_NUMS = 4
MAX_SAMPLES = 2000

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
# === Training & Generating (common use) ===
parser.add_argument('--debug', type=str2bool, default=True)
parser.add_argument('--train_or_gen', type=str2bool, default=True, help='True for training, False for generating.')
parser.add_argument('--server_mode', type=str2bool, default=False, help='True for server, False for local.')
parser.add_argument('--learning_rate', type=float, default=4e-4, help='')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--valid_epoch', type=int, default=1)

# === Architecture ===
parser.add_argument('--maximum_bar_nums', type=int, default=64)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--position_dropout', type=float, default=0.1)

parser.add_argument('--n_groups', type=int, default=8)
parser.add_argument('--n_codes', type=int, default=1024)
parser.add_argument('--d_latent', type=int, default=512)

# ***** Encoder *******
parser.add_argument('--encoder_transformer_n_layer', type=int, default=4)
parser.add_argument('--encoder_transformer_n_head', type=int, default=8)
parser.add_argument('--encoder_transformer_mlp', type=int, default=1024)
parser.add_argument('--encoder_transformer_dropout', type=float, default=0.1)

# ***** Decoder *******
parser.add_argument('--decoder_transformer_n_layer', type=int, default=6)
parser.add_argument('--decoder_transformer_n_head', type=int, default=8)
parser.add_argument('--decoder_transformer_mlp', type=int, default=1024)
parser.add_argument('--decoder_transformer_dropout', type=float, default=0.1)

parser.add_argument('--track_fusion_n_layer', type=int, default=2)

# ***** Configuration ******
parser.add_argument('--max_seq_len', type=int, default=1024, help='remi_track_64=2048 & remi_track_32=1024')
parser.add_argument('--max_bars', type=int, default=32, help='32 or 64 ')
parser.add_argument('--prior_info_flavor', type=str, default='both', help='both, latent, meta')
parser.add_argument('--masking_type', type=str, default='original', help='original, adaptive')

parser.add_argument('--token_type', type=str, default='remi_track', help='only for remi_track')   # fixed

# === I/O (common use) ===
parser.add_argument('--data_dir', type=str, default='../Dataset/Final_Data_Path_1127/', help='save dir of final processed dataset')
parser.add_argument('--save_dir', type=str, default='./ckpt/BandControlNet/')  # fixed
parser.add_argument('--generated_dir', type=str, default='./generated/BandControlNet')  # fixed
parser.add_argument('--max_gen_nums', type=int, default=500)

parser.add_argument('--bpe_model_path', type=str, default=None, help='BPE_Model Path') 
parser.add_argument('--n_parameters', type=int, default=0, help='net parameters') 

# === Log (common use) ===
parser.add_argument('--verbose', type=str2bool, default=True)
# === Device (common use) ===
parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
parser.add_argument('--cuda_ids', type=str, default='0', help='gpu id on server')

parser.add_argument('--parallel', type=str2bool, default=False, help='multi-gpu training.')

def network_paras(model):
    # only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train():
    model_type = 'BandControlNet_' + args.masking_type + '_' + args.prior_info_flavor + '_' + str(args.max_bars)
    print('-' * 10, 'Loading data and model <{}>'.format(model_type), '-' * 10)
    start = time.time()

    # loading train_path, valid_path
    paths = os.path.join(args.data_dir, 'data_{}_path_20231127_new.pkl'.format(args.max_bars))
    if args.server_mode:
        args.data_dir = os.path.join(SERVER_DATA_DIR, 'Final_Data_Path_1127/')
        paths = os.path.join(args.data_dir, 'data_{}_path_20231127_new.pkl'.format(args.max_bars))

    all_dataset_paths = pickle.load(open(paths, 'rb'))
    if args.debug:
        # using testing dataset as debug data
        training_data_paths = all_dataset_paths['testing']
    else:
        training_data_paths = all_dataset_paths['training'] + all_dataset_paths['valid']

    args.bpe_model_path = '../Dataset/BPE_model_{}/10k_1127'.format(args.max_bars)
    if args.server_mode:
        args.bpe_model_path = SERVER_DATA_DIR + '/BPE_model_{}/10k_1127'.format(args.max_bars)

    training_dataset = Pop_Multi_Dataset(file_names=training_data_paths,
                                         bpe_model_path=args.bpe_model_path,
                                         max_seq_len=args.max_seq_len,
                                         max_bars=args.max_bars,
                                         server_mode=args.server_mode,
                                         )
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=4, pin_memory=True,
                                     collate_fn=Pop_Multi_Dataset.collate,
                                     )
    net = BandControlNet(args, VOCAB_SIZE)
    print('Done. It took {:.6f}s'.format(time.time() - start))
    print()

    # ==========================================================================================
    device = args.device

    if args.parallel:
        net = nn.DataParallel(net)
    # ==========================================================================================

    net.to(torch.device(device))
    n_parameters = network_paras(net)
    args.n_parameters = int(n_parameters)
    print('n_parameters: {:,}'.format(n_parameters))

    # print(net)
    print()

    # optimizers
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            (1 / 10 * args.epochs) * len(training_dataloader),  # epoch 20
            (1 / 2 * args.epochs) * len(training_dataloader),  # epoch 100
            0.1,  # 4e-4 -> 4e-5
        ),
    )

    # save configuration
    args_file = os.path.join(args.save_dir, '{}_args.json'.format(model_type))
    print(args_file)
    with open(args_file, 'wt') as f:
        json.dump(vars(args), f, indent=4)

    # print configuration
    print_s = ''
    for arg in vars(args):
        s = '{}\t{}\n'.format(arg, getattr(args, arg))
        print_s += s
    print('\n' + '-' * 40 + '\n')
    print('[Arguments] \n')
    print(print_s)
    print('[Datasets]\n')
    print(f'training_data_length: {len(training_dataset)}\n')
    print('\n' + '-' * 40 + '\n')

    # Training & Valid
    print('-' * 10, 'Training model -- <{}>'.format(model_type), '-' * 10)

    max_grad_norm = 3

    loss_file = open(os.path.join(args.save_dir, '{}_loss.csv'.format(model_type)), 'w')
    loss_file.write('step, epoch_id, current_lr, stop, train_loss_epoch, loss_time\n')
    step = 0
    stop = 0
    best_loss = np.Inf

    for epoch in range(args.epochs):
        if epoch >= 30:
            break
        acc_loss = 0
        net.train()
        for bidx, batch_items in enumerate(tqdm(training_dataloader)):
            batch_input_seq = batch_items['input_seq'].long().to(torch.device(device))
            batch_target_seq = batch_items['target_seq'].long().to(torch.device(device))
            batch_mask = batch_items['mask'].long().to(torch.device(device))

            track_name = batch_items['track_name'].long().to(torch.device(device))
            chords_seq = batch_items['chords_seq'].long().to(torch.device(device))
            feat_seq = batch_items['feat_seq'].long().to(torch.device(device))
            vq_codes = batch_items['vq_codes'].long().to(torch.device(device))

            bar_embed_ids = batch_items['bar_embed_ids'].long().to(torch.device(device))
            bar_ids = batch_items['bar_ids'].long().to(torch.device(device))
            bar_len = batch_items['bar_len'].long().to(torch.device(device))

            step += 1

            net.zero_grad()

            y_events = net(batch_input_seq,
                           track_name,
                           chords_seq, feat_seq, vq_codes,
                           bar_embed_ids, bar_ids, bar_len,
                           batch_mask)
            if args.parallel:
                loss = net.module.compute_loss(y_events, batch_target_seq, batch_mask)
            else:
                loss = net.compute_loss(y_events, batch_target_seq, batch_mask)

            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            acc_loss += loss.item()
        train_epoch_loss = acc_loss / len(training_dataloader)
        train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        print('\n', '-' * 20, '\n')
        print(f'epoch: {epoch}/{args.epochs} | Train_Loss_Epoch: {train_epoch_loss:.6f} | time: {train_time}')

        # save model
        if train_epoch_loss < best_loss:
            torch.save(
                {
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type))
            )
            print(f'loss decreased ({best_loss:.6f} --> {train_epoch_loss:.6f}).  Saving model ...')
            best_loss = train_epoch_loss
            stop = 0
        else:
            stop += 1
            print(f'loss increased. Stop counter: {stop}')

        current_lr = optimizer.param_groups[0]['lr']
        print('current lr: {} -- epoch: {}'.format(current_lr, epoch))
        print('\n', '-' * 20, '\n')

        loss_file.write(f'{step}, {epoch}, {current_lr}, {stop}, {train_epoch_loss}, {train_time}\n')
    loss_file.close()

def load_model_for_generation():
    model_type = 'BandControlNet_' + args.masking_type + '_' + args.prior_info_flavor + '_' + str(args.max_bars)
    assert args.train_or_gen == False
    assert args.server_mode == False

    paths = os.path.join(args.data_dir, 'data_{}_path_20231127_new.pkl'.format(args.max_bars))
    all_dataset_paths = pickle.load(open(paths, 'rb'))
    testing_data_paths = all_dataset_paths['testing']

    args.bpe_model_path = '../Dataset/BPE_model_{}/10k_1127'.format(args.max_bars)
    testing_dataset = Pop_Multi_Dataset(file_names=testing_data_paths,
                                        bpe_model_path=args.bpe_model_path,
                                        max_seq_len=args.max_seq_len,
                                        max_bars=args.max_bars,
                                        server_mode=args.server_mode,
                                        )
    net = BandControlNet(args, VOCAB_SIZE)
    # loading model
    print('-' * 10, 'Loading saved model -- <{}>'.format(model_type), '-' * 10, '\n')

    checkpoint_epoch = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['epoch']
    checkpoint_lr = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['optimizer']

    print('model -- <{}> -- saved at epoch@{} with lr@{}'.format(model_type, checkpoint_epoch,
                                                                 checkpoint_lr['param_groups'][0]['lr']))

    print('\n', '-' * 40, '\n')

    resume_dict = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['state_dict']
    if args.parallel:
        new_resume_dict = dict()
        for k, v in resume_dict.items():
            if 'module.' in k:
                new_resume_dict[k.replace('module.', '')] = v
        resume_dict = new_resume_dict

    net.load_state_dict(resume_dict)
    device = args.device
    net.to(torch.device(device))
    net.eval()

    return model_type, net, testing_dataset

def generate_barbybar(max_gen_nums=500):
    model_type, net, testing_dataset = load_model_for_generation()

    # create dir for generated sample
    generated_dir = os.path.join(args.generated_dir, f'{model_type}')
    os.makedirs(generated_dir, exist_ok=True)

    refer_info_file = open(os.path.join(generated_dir, '{}_refer_info.csv'.format(model_type)), 'w')
    refer_info_file.write('idx, file_name, bar_nums, ins_1, ins_2, ins_3, ins_4, infer_time, token_nums, note_nums\n')

    random_id_list = random.sample(range(0, len(testing_dataset)), MAX_SAMPLES)

    success_num = 1
    for i in range(MAX_SAMPLES):
        try:
            if success_num > max_gen_nums:
                break
            song_name = '{}-{}'.format(model_type, i)
            with torch.no_grad():
                random_song_id = random_id_list[i]
                bos = testing_dataset[random_song_id]['seq'][:3, :]
                bos = torch.as_tensor(bos, dtype=torch.long, device=args.device).unsqueeze(0)

                # List: [(1, 2)]*4
                x = [bos[:, :, track_i] for track_i in range(TRACK_NUMS)]

                # (1, max_bar_nums, TRACK_NUMS)
                chords_seq = testing_dataset[random_song_id]['chords_seq'].unsqueeze(0).to(torch.device(args.device))
                # (1, max_bar_nums, 4, TRACK_NUMS)
                feat_seq = testing_dataset[random_song_id]['feat_seq'].unsqueeze(0).to(torch.device(args.device))
                # (1, max_bar_nums, 16, TRACK_NUMS)
                vq_codes = testing_dataset[random_song_id]['vq_codes'].unsqueeze(0).to(torch.device(args.device))
                # (1, TRACK_NUMS)
                track_name = testing_dataset[random_song_id]['track_name'].unsqueeze(0).to(torch.device(args.device))

                # (1, TRACK_NUMS, D)
                track_embeded = net.track_embedding(track_name)
                # processing features
                ref_bar_nums = testing_dataset[random_song_id]['bar_len'].unsqueeze(0).to(torch.device(args.device))
                features_len = torch.tensor([ref_bar_nums], dtype=chords_seq.dtype, device=chords_seq.device)

                # (bs, max_bar_nums, TRACK_NUMS, D)
                features = net.processing_features(features_len, chords_seq, feat_seq, vq_codes)

                if args.masking_type == 'adaptive':
                    sim_attns = net.cal_sim_attns(features, features_len)
                else:
                    sim_attns = None

                start_time = time.time()
                while True:
                    bar_count = 0
                    for token_id in x[0].squeeze(0):
                        if token_id in [5, 6]:
                            bar_count += 1
                    x = net.generate_perbar(x, track_embeded, features, features_len, sim_attns)

                    if bar_count >= ref_bar_nums:
                        break
                all_tracks = [testing_dataset.bpe.decode_bpe(x_per.squeeze(0).detach().cpu().numpy()) for x_per in x]

            infer_time = time.time() - start_time
            token_nums = sum([len(track_per) - bos.shape[1] for track_per in all_tracks])

            song_gen = remi_track2midi([recover_position_track(track_per[1:]) for track_per in all_tracks])
            song_gen.dump(os.path.join(generated_dir, f'{song_name}_cover.mid'))

            note_nums = count_note_nums(song_gen)

            ori_file = testing_dataset[random_song_id]['file_name']
            print(f'No. {i} reference songs name: {ori_file}')
            ref_bar_nums = testing_dataset[random_song_id]['bar_len']
            track_name = testing_dataset[random_song_id]['track_name'] - 5
            ins_name = [Ins_LIST[i] for i in track_name.tolist()]
            refer_info_file.write(f'{i}, {ori_file}, {ref_bar_nums.item()}, '
                                  f'{ins_name[0]}, {ins_name[1]}, {ins_name[2]}, {ins_name[3]}, '
                                  f'{infer_time}, {token_nums}, {note_nums}\n')

            ori_seq = testing_dataset[random_song_id]['seq']
            ori_mask = (ori_seq != 0).long()
            ori_length = torch.sum(ori_mask, dim=0)

            ori_tokens = [testing_dataset.bpe.decode_bpe(ori_seq[:ori_length[i], i]) for i in range(TRACK_NUMS)]
            song_ori = remi_track2midi([recover_position_track(track_per[1:]) for track_per in ori_tokens])
            song_ori.dump(os.path.join(generated_dir, f'{song_name}_ori.mid'))

            success_num += 1
        except Exception as e:
            print(e)

    refer_info_file.close()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.server_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids

    # create dir for model saving
    os.makedirs(args.save_dir, exist_ok=True)

    if args.train_or_gen:
        fix_seed(1024)
        train()
    else:
        fix_seed(1024)
        generate_barbybar(max_gen_nums=500)