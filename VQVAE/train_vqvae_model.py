
import numpy as np
import pickle
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

from VQ_VAE_Model import VQVAE, VectorQuantizeEMA

from dataset_bpe_vqvae import VQVAE_Dataset_Raw, VQVAE_Dataset_Track
from vocab_v2 import RemiVocab

from utils.model_utils import EarlyStopping, get_lr_multiplier, fix_seed, recover_position_raw, recover_position_track
from representation_multiple_v2 import remi_raw2midi, remi_track2midi

from argparse import ArgumentParser, ArgumentTypeError

SERVER_DATA_DIR = '/home/data/music_gen_cl/Bandformer_new'

# BPE_VOCAB_SIZE = 10000

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
parser.add_argument('--debug', type=str2bool, default=False)
parser.add_argument('--train_or_gen', type=str2bool, default=True, help='True for training, False for inference.')
parser.add_argument('--server_mode', type=str2bool, default=False, help='True for server, False for local.')
parser.add_argument('--learning_rate', type=float, default=4e-4)
parser.add_argument('--batch_size', type=int, default=320, help='remi_raw: 160; remi_track:320')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--valid_epoch', type=int, default=1)

# === Architecture (for vq-vae model) ===
parser.add_argument('--max_seq_len', type=int, default=128, help='remi_raw: 256; remi_track: 128')
parser.add_argument('--d_model', type=int, default=256)

parser.add_argument('--encoder_layers', type=int, default=4)
parser.add_argument('--decoder_layers', type=int, default=6)
parser.add_argument('--encoder_mlp', type=int, default=1024)
parser.add_argument('--decoder_mlp', type=int, default=1024)
parser.add_argument('--n_head', type=int, default=8)

parser.add_argument('--max_position_embeddings', type=int, default=512)

parser.add_argument('--n_codes', type=int, default=1024)    
parser.add_argument('--n_groups', type=int, default=8)     
parser.add_argument('--d_latent', type=int, default=512)    
parser.add_argument('--commitment_cost', type=float, default=0.25, help='')

parser.add_argument('--sampler_rate', type=int, default=3, help='')
parser.add_argument('--velocity_enable', type=int, default=True, help='')

# === I/O (common use) ===
parser.add_argument('--data_dir', type=str, default='../Dataset/Final_Data_Path_1127/', help='save dir of final processed dataset')
parser.add_argument('--max_bars', type=int, default=64, help='maximum bars of final processed dataset, fixed=64')
parser.add_argument('--token_type', type=str, default='remi_track', help='remi_raw or remi_track')
parser.add_argument('--cover_level_type', type=str, default='bar', help='bar, no phrase_level')
parser.add_argument('--save_dir', type=str, default='./ckpt/vqvae_model_mv/')
parser.add_argument('--generated_dir', type=str, default='./generated/vqvae_model_mv/')

# === Log (common use) ===
parser.add_argument('--verbose', type=str2bool, default=True)

# === Device (common use) ===
parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
parser.add_argument('--autocast', type=str2bool, default=False, help='mixed precision for training.')
parser.add_argument('--cuda_ids', type=str, default='0', help='gpu id on server')

parser.add_argument('--parallel', type=str2bool, default=False, help='multi-gpu training.')

def network_paras(model):
    # only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train():
    REMI_VOCAB_SIZE = len(RemiVocab(velocity_enable=args.velocity_enable))
    # loading data & model
    model_type = args.token_type + '_' + args.cover_level_type + '_level_' + str(args.max_bars)
    print('-' * 10, 'Loading data and model <{}>'.format(model_type), '-' * 10)
    start = time.time()

    # loading train_path, valid_path
    paths = os.path.join(args.data_dir, 'data_{}_path_20240101.pkl'.format(args.max_bars))
    if args.server_mode:
        args.data_dir = os.path.join(SERVER_DATA_DIR, 'Final_Data_Path_1127/')
        paths = os.path.join(args.data_dir, 'data_{}_path_20240101.pkl'.format(args.max_bars))

    all_dataset_paths = pickle.load(open(paths, 'rb'))
    if args.debug:
        # using testing dataset as debug data
        training_data_paths = all_dataset_paths['testing']
    else:
        # 合并training set和testing set
        training_data_paths = all_dataset_paths['training'] + all_dataset_paths['testing']
    valid_data_paths = all_dataset_paths['valid']

    if args.token_type == 'remi_raw':
        training_dataset = VQVAE_Dataset_Raw(file_names=training_data_paths,
                                             max_seq_len=args.max_seq_len,
                                             velocity_enable=args.velocity_enable,
                                             server_mode=args.server_mode,
                                             )

        valid_dataset = VQVAE_Dataset_Raw(file_names=valid_data_paths,
                                          max_seq_len=args.max_seq_len,
                                          velocity_enable=args.velocity_enable,
                                          server_mode=args.server_mode,
                                          )
        training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=4, pin_memory=True,
                                         collate_fn=VQVAE_Dataset_Raw.collate,
                                         )
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      collate_fn=VQVAE_Dataset_Raw.collate)

        net = VQVAE(args, REMI_VOCAB_SIZE)

    else:
        assert args.token_type == 'remi_track'
        training_dataset = VQVAE_Dataset_Track(file_names=training_data_paths,
                                               max_seq_len=args.max_seq_len,
                                               velocity_enable=args.velocity_enable,
                                               server_mode=args.server_mode,
                                               )

        valid_dataset = VQVAE_Dataset_Track(file_names=valid_data_paths,
                                            max_seq_len=args.max_seq_len,
                                            velocity_enable=args.velocity_enable,
                                            server_mode=args.server_mode,
                                            )

        training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=4, pin_memory=True,
                                         collate_fn=VQVAE_Dataset_Track.collate,
                                         )
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=4, pin_memory=True,
                                      collate_fn=VQVAE_Dataset_Track.collate)

        net = VQVAE(args, REMI_VOCAB_SIZE)

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
            (1 / 2 * args.epochs) * len(training_dataloader),   # epoch 100
            0.1,                             # 4e-4 => 4e-5
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
    print(f'valid_data_length: {len(valid_dataset)}\n')
    print('\n' + '-' * 40 + '\n')

    # Training & Valid
    print('-' * 10, 'Training model -- <{}>'.format(model_type), '-' * 10)

    early_stopping = EarlyStopping(save_dir=args.save_dir, save_name=model_type, patience=args.epochs // 20,
                                   verbose=True)
    max_grad_norm = 3
    loss_file = open(os.path.join(args.save_dir, '{}_loss.csv'.format(model_type)), 'w')
    loss_file.write(
        'step, epoch_id, current_lr, early_stopping_counter, '
        'train_loss_epoch, train_rec_loss_epoch, train_vq_loss_epoch, train_ppl,'
        'valid_loss_epoch, valid_rec_loss_epoch, valid_vq_loss_epoch, valid_ppl,'
        'time-valid\n')
    step = 0

    for epoch in range(args.epochs):
        acc_loss = 0
        acc_rec_loss = 0
        acc_vq_loss = 0
        acc_ppl = 0
        # acc_avg_usage = 0

        net.train()
        for bidx, batch_items in enumerate(tqdm(training_dataloader)):
            batch_input_seq = batch_items['input_seq'].long().to(torch.device(device))
            batch_target_seq = batch_items['target_seq'].long().to(torch.device(device))

            step += 1

            net.zero_grad()

            if args.parallel:
                metrics = net.module.compute_loss(batch_input_seq, batch_target_seq)
            else:
                metrics = net.compute_loss(batch_input_seq, batch_target_seq)

            loss = metrics['loss']
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            rec_loss = metrics['rec_loss']
            vq_loss = metrics['vq_loss']
            ppl = metrics['ppl']

            acc_loss += loss.item()
            acc_rec_loss += rec_loss.item()
            acc_vq_loss += vq_loss.item()
            acc_ppl += ppl.item()

        epoch_loss = acc_loss / len(training_dataloader)
        epoch_rec_loss = acc_rec_loss / len(training_dataloader)
        epoch_vq_loss = acc_vq_loss / len(training_dataloader)
        epoch_ppl = acc_ppl / len(training_dataloader)

        train_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

        print('\n', '-' * 20, '\n')
        print('epoch: {}/{} '
              '| Train_Loss_Epoch: {:.6f} '
              '| Train_Rec_Loss_Epoch: {:.6f} '
              '| Train_VQ_Loss_Epoch: {:.6f} '
              '| Train_PPL_Loss_Epoch: {:.6f} '
              '| time: {}'.format(epoch, args.epochs,
                                  epoch_loss, epoch_rec_loss, epoch_vq_loss, epoch_ppl,
                                  train_time,
                                  ))

        # valid step
        if (epoch + 1) % args.valid_epoch == 0:
            net.eval()
            acc_loss = 0
            acc_rec_loss = 0
            acc_vq_loss = 0
            acc_ppl = 0

            with torch.no_grad():
                for bidx_valid, batch_items_valid in enumerate(valid_dataloader):
                    batch_input_seq_valid = batch_items_valid['input_seq'].long().to(torch.device(device))
                    batch_target_seq_valid = batch_items_valid['target_seq'].long().to(torch.device(device))

                    if args.parallel:
                        metrics = net.module.compute_loss(batch_input_seq_valid, batch_target_seq_valid)
                    else:
                        metrics = net.compute_loss(batch_input_seq_valid, batch_target_seq_valid)

                    loss = metrics['loss']
                    rec_loss = metrics['rec_loss']
                    vq_loss = metrics['vq_loss']
                    ppl = metrics['ppl']

                    acc_loss += loss.item()
                    acc_rec_loss += rec_loss.item()
                    acc_vq_loss += vq_loss.item()
                    acc_ppl += ppl.item()

            valid_loss = acc_loss / len(valid_dataloader)
            valid_rec_loss = acc_rec_loss / len(valid_dataloader)
            valid_vq_loss = acc_vq_loss / len(valid_dataloader)
            valid_ppl = acc_ppl / len(valid_dataloader)

            valid_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
            print('epoch: {}/{} '
                  '| Valid_Loss_Epoch: {:.6f} '
                  '| Valid_Rec_Loss_Epoch: {:.6f} '
                  '| Valid_VQ_Loss_Epoch: {:.6f} '
                  '| Valid_PPL_Loss_Epoch: {:.6f} '
                  '| time: {}'.format(epoch, args.epochs,
                                      valid_loss, valid_rec_loss, valid_vq_loss, valid_ppl,
                                      valid_time))
            early_stopping(valid_loss, net, epoch, optimizer)
            if early_stopping.early_stop:
                print('Early stopping at Epoch--{}'.format(epoch))
                break
            current_lr = optimizer.param_groups[0]['lr']
            print('current lr: {} -- epoch: {}'.format(current_lr, epoch))
            print('\n', '-' * 20, '\n')

            loss_file.write(f'{step}, {epoch}, {current_lr}, {early_stopping.counter}, '
                            f'{epoch_loss}, {epoch_rec_loss}, {epoch_vq_loss}, {epoch_ppl}, '
                            f'{valid_loss}, {valid_rec_loss}, {valid_vq_loss}, {valid_ppl}, '
                            f'{valid_time}\n')
    loss_file.close()

def load_model_for_generation():
    model_type = args.token_type + '_' + args.cover_level_type + '_level_' + str(args.max_bars)
    assert args.train_or_gen == False

    paths = os.path.join(args.data_dir, 'data_{}_path_20240101.pkl'.format(args.max_bars))
    all_dataset_paths = pickle.load(open(paths, 'rb'))
    testing_data_paths = all_dataset_paths['testing']

    if args.token_type == 'remi_raw':
        testing_dataset = VQVAE_Dataset_Raw(file_names=testing_data_paths,
                                            max_seq_len=args.max_seq_len,
                                            velocity_enable=False,
                                            )
        net = VQVAE(args, REMI_VOCAB_SIZE)
    else:
        assert args.token_type == 'remi_track'
        testing_dataset = VQVAE_Dataset_Track(file_names=testing_data_paths,
                                               max_seq_len=args.max_seq_len,
                                               velocity_enable=False,
                                               )
        net = VQVAE(args, REMI_VOCAB_SIZE)

    # loading model
    print('-' * 10, 'Loading saved model -- <{}>'.format(model_type), '-' * 10, '\n')
    checkpoint_epoch = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['epoch']
    checkpoint_lr = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['optimizer']
    print('model -- <{}> -- saved at epoch@{} with lr@{}'.format(model_type, checkpoint_epoch,
                                                                 checkpoint_lr['param_groups'][0]['lr']))
    print('\n', '-' * 40, '\n')

    resume_dict = torch.load(os.path.join(args.save_dir, '{}_checkpoint.pt'.format(model_type)))['state_dict']
    net.load_state_dict(resume_dict)
    device = args.device
    net.to(torch.device(device))
    net.eval()

    return model_type, net, testing_dataset


if __name__ == '__main__':
    args = parser.parse_args()
    if args.server_mode:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids
    # create dir for model saving
    os.makedirs(args.save_dir, exist_ok=True)
    # create dir for generated sample
    os.makedirs(args.generated_dir, exist_ok=True)

    if args.train_or_gen:
        fix_seed(1024)
        train()


