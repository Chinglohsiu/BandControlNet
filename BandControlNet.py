
# 使用单序列中提取的feat_seq, chords_seq, vq_codes作为条件
# adaptive or original => 结构增强版本和普通版本
#

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.transformers import TransformerEncoderLayer
from fast_transformers.masking import TriangularCausalMask, LengthMask, FullMask

from custom_layers.AdaptiveTransformerLayer import AdaptiveTransformerDecoderLayer
from custom_layers.AdaptiveAttentionLayer import AdaptiveAttentionLayer
from custom_layers.AdaptiveAttention import AdaptiveAttention
from fast_transformers.attention import AttentionLayer, FullAttention, LinearAttention

from utils.model_utils import Embeddings, PositionalEncoding, sampling_v2
from utils.inference_utils import grammar_control_without_phrase, get_struct_info_track_without_phrase, infer_embed_ids


TRACK_NUMS = 4

class BandControlNet(nn.Module):
    def __init__(self, args, vocab_size):
        # vocab_size is a dict
        # 'seq', 'chords', 'vq_codes'
        # 'drums_pitch_type', 'drums_note_density'
        # 'note_density', 'mean_pitch', 'mean_duration', 'mean_velocity'
        # 'track'

        super(BandControlNet, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.chords_embed_size = 256
        self.vq_codes_embed_size = args.d_latent
        self.feat_embed_size = {
            'drums_pitch_type': 64,
            'drums_note_density': 128,
            'note_density': 128,
            'mean_pitch': 64,
            'mean_duration': 64,
            'mean_velocity': 64,
        }

        self.maximum_bar_nums = args.maximum_bar_nums
        self.max_bars = args.max_bars
        self.n_groups = args.n_groups

        self.D = args.d_model
        self.position_dropout = args.position_dropout

        self.train_or_gen = args.train_or_gen

        self.prior_info_flavor = args.prior_info_flavor
        self.masking_type = args.masking_type
        self.blur = args.blur

        self.encoder_transformer_n_layer = args.encoder_transformer_n_layer
        self.encoder_transformer_n_head = args.encoder_transformer_n_head
        self.encoder_transformer_mlp = args.encoder_transformer_mlp
        self.encoder_transformer_dropout = args.encoder_transformer_dropout

        self.decoder_transformer_n_layer = args.decoder_transformer_n_layer
        self.decoder_transformer_n_head = args.decoder_transformer_n_head
        self.decoder_transformer_mlp = args.decoder_transformer_mlp
        self.decoder_transformer_dropout = args.decoder_transformer_dropout

        self.track_fusion_n_layer = args.track_fusion_n_layer

        self.token_embedding = Embeddings(self.vocab_size['seq'], self.D, padding_idx=0)
        self.bar_embedding = Embeddings(self.maximum_bar_nums + 1, self.D, padding_idx=0)
        # Track embedding
        self.track_embedding = Embeddings(self.vocab_size['track'], self.D, padding_idx=0)

        if self.prior_info_flavor in ['latent', 'both']:
            self.vq_codes_embedding = Embeddings(self.vocab_size['vq_codes'], self.vq_codes_embed_size // self.n_groups)
            self.vq_codes_linear = nn.Linear(self.vq_codes_embed_size, self.D)
        if self.prior_info_flavor in ['meta', 'both']:
            self.chords_embedding = Embeddings(self.vocab_size['chords'], self.chords_embed_size, padding_idx=0)
            self.chords_linear = nn.Linear(self.chords_embed_size * 4, self.D)

            # drums track => drums_feat: [drums_pitch_type, drums_note_density]
            self.drums_feat_embedding = nn.ModuleList(
                [Embeddings(self.vocab_size['drums_pitch_type'], self.feat_embed_size['drums_pitch_type']),
                 Embeddings(self.vocab_size['drums_note_density'], self.feat_embed_size['drums_note_density']),
                 ])
            self.drums_feat_linear = nn.Linear(
                self.feat_embed_size['drums_pitch_type'] + self.feat_embed_size['drums_note_density'], self.D)

            # other track => [piano, guitar, bass, strings, melody] share the same feats:
            #   [note_density, mean_pitch, mean_duration, mean_velocity]
            self.others_feat_embedding = nn.ModuleList(
                [Embeddings(self.vocab_size['note_density'], self.feat_embed_size['note_density']),
                 Embeddings(self.vocab_size['mean_pitch'], self.feat_embed_size['mean_pitch']),
                 Embeddings(self.vocab_size['mean_duration'], self.feat_embed_size['mean_duration']),
                 Embeddings(self.vocab_size['mean_velocity'], self.feat_embed_size['mean_velocity']),
                 ])
            self.others_feat_linear = nn.Linear(
                self.feat_embed_size['note_density'] + self.feat_embed_size['mean_pitch'] + \
                self.feat_embed_size['mean_duration'] + self.feat_embed_size['mean_velocity'], self.D)

        if self.prior_info_flavor == 'both':
            # 合并vq_codes, feat, chords, 并缩小维度为self.D
            self.combined_prior2d = nn.Linear(self.D * 2, self.D)
            self.combined_prior3d = nn.Linear(self.D * 3, self.D)
        if self.prior_info_flavor == 'meta':
            # 合并feat, chords, 并缩小维度为self.D
            self.combined_prior2d = nn.Linear(self.D * 2, self.D)

        self.PosEmb = PositionalEncoding(self.D, dropout=self.position_dropout)

        # === Encoder === normal transformer encoder, attention_type=linear
        # self.track_encoder = TransformerEncoderBuilder.from_kwargs(
        self.time_encoder = TransformerEncoderBuilder.from_kwargs(
            # n_layers=2,
            n_layers=self.encoder_transformer_n_layer,  # 4-layer
            n_heads=self.encoder_transformer_n_head,
            query_dimensions=self.D // self.encoder_transformer_n_head,
            value_dimensions=self.D // self.encoder_transformer_n_head,
            feed_forward_dimensions=self.encoder_transformer_mlp,
            activation='gelu',
            attention_type='linear',  # linear
            dropout=self.encoder_transformer_dropout,
        ).get()

        # self.track_pooler = nn.Sequential(
        #     nn.Linear(self.D, self.D),
        #     nn.Tanh(),
        # )
        # self.time_encoder = copy.deepcopy(self.track_encoder)

        # === decoder ==
        self.bottom_decoder = self.get_decoder()
        self.top_decoder = self.get_decoder()

        # 类似与track_encoder, 推理时注意！！！(获取每轨的bar_ids)
        self.fusion_decoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.track_fusion_n_layer,  # 2-layer
            n_heads=self.decoder_transformer_n_head,
            query_dimensions=self.D // self.decoder_transformer_n_head,
            value_dimensions=self.D // self.decoder_transformer_n_head,
            feed_forward_dimensions=self.decoder_transformer_mlp,
            activation='gelu',
            attention_type='linear',  # linear
            dropout=self.decoder_transformer_dropout,
        ).get()

        if self.masking_type == 'adaptive':
            self.norm_sim = nn.LayerNorm(self.max_bars)
            self.sim_query_linear = nn.Linear(self.D,
                                              self.D // TRACK_NUMS * TRACK_NUMS)
            self.sim_key_linear = nn.Linear(self.D,
                                            self.D // TRACK_NUMS * TRACK_NUMS)

        # === Prediction
        # output_project_layer for each track
        self.project_linear = nn.ModuleList([nn.Linear(self.D, self.vocab_size['seq'])
                                             for _ in range(TRACK_NUMS)])
        self.loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

    def get_decoder(self):
        return nn.ModuleList([
            AdaptiveTransformerDecoderLayer(self_attention=AdaptiveAttentionLayer(AdaptiveAttention(), self.D, self.decoder_transformer_n_head),
                                            cross_attention=AttentionLayer(FullAttention(), self.D, self.decoder_transformer_n_head),
                                            d_model=self.D,
                                            d_ff=self.decoder_transformer_mlp,
                                            dropout=self.decoder_transformer_dropout,
                                            masking_type=self.masking_type,
            )
            for _ in range(self.decoder_transformer_n_layer//2)     # top_decoder&bottom_decoder => 3 layers
        ])

    def forward_decoder(self, x, memory,
                        layers,
                        x_mask, x_length_mask, memory_mask, memory_length_mask,
                        custom_attns,
                        ):
        for layer in layers:
            x = layer(x, memory,
                      x_mask=x_mask, x_length_mask=x_length_mask,
                      memory_mask=memory_mask, memory_length_mask=memory_length_mask,
                      custom_attns=custom_attns,
                      )
        return x

    def cal_sim_attns(self, x_prior_seq, prior_len):
        # x_prior_seq: (bs, max_bar_nums, TRACK_NUMS, D)
        # prior_len为prior中小节长度，训练时为bar_len, 推理时为参考序列的bar_len

        bs = x_prior_seq.shape[0]
        max_bar_nums = x_prior_seq.shape[1]

        scale = self.D ** -0.5
        length_mask = LengthMask(prior_len, max_len=max_bar_nums, device=x_prior_seq.device)

        # prior_query = self.sim_query_linear(x_prior_seq).reshape(bs, max_bar_nums, TRACK_NUMS, -1)
        # prior_key = self.sim_key_linear(x_prior_seq).reshape(bs, max_bar_nums, TRACK_NUMS, -1)
        # prior_QK = torch.einsum('nlhe, nshe->nhls', prior_query, prior_key)
        # prior_QK = prior_QK + length_mask.additive_matrix[:, None, None]

        # multi-head, # 将TRACK_NUMS当做n_head
        prior_query = self.sim_query_linear(x_prior_seq)
        prior_key = self.sim_key_linear(x_prior_seq)
        prior_QK = torch.einsum('nlhe, nshe->nhls', prior_query, prior_key)
        prior_QK = prior_QK + length_mask.additive_matrix[:, None, None]

        # (bs, TRACK_NUMS, max_bar_nums, max_bar_nums)
        sim_attns = torch.softmax(prior_QK * scale, dim=-1)
        # 直接使用Layer Normalization
        sim_attns = self.norm_sim(sim_attns)

        # (bs, TRACK_NUMS, max_bar_nums, max_bar_nums)
        return sim_attns

    def expand_sim_attns_per(self, x_per, sim_attns_per, bar_ids_per, bar_len):
        # 扩充每个轨道的sim_attns
        bs = x_per.shape[0]
        max_seq_len = x_per.shape[1]

        custom_attns = []
        for bs_i in range(bs):
            sim_attns_each = sim_attns_per[bs_i]

            custom_attns_each = torch.ones(max_seq_len, max_seq_len, device=x_per.device)

            if self.train_or_gen:
                seq_length_each = torch.sum(x_per[bs_i] != 0).item()
            else:
                seq_length_each = max_seq_len

            new_bar_ids = [0] + bar_ids_per[bs_i][1:bar_len[bs_i]].tolist() + [seq_length_each]
            bar_contexts = list(zip(new_bar_ids[:-1], new_bar_ids[1:]))
            actual_bar_nums = len(bar_contexts)
            for i in range(actual_bar_nums):
                for j in range(actual_bar_nums):
                    custom_attns_each[bar_contexts[i][0]:bar_contexts[i][1],
                    bar_contexts[j][0]:bar_contexts[j][1]] = sim_attns_each[i, j]
            custom_attns.append(custom_attns_each)

        # (bs, max_seq_len, max_seq_len)
        custom_attns = torch.stack(custom_attns, dim=0)
        return custom_attns

    def processing_features(self, features_len, chords_seq, feat_seq, vq_codes):
        # features_len 训练是等同于bar_len, 生成时为实际小节长度
        # chords_seq: (bs, max_bar_nums, 4), shared; max_bar_nums: batch中最大小节数
        # feat_seq: (bs, max_bar_nums, 4, TRACK_NUMS) drums=>2维 feat_seq[:, :, :2, :]，其余padding，其他轨=>4维
        # vq_codes: (bs, max_bar_nums, 16, TRACK_NUMS), 每个轨16维

        # return features: (bs, max_bar_nums, D, TRACK_NUMS)

        # === Embedding (prior/features) ===
        if self.prior_info_flavor in ['meta', 'both']:
            bs = chords_seq.shape[0]
            max_bar_nums = chords_seq.shape[1]

            x_chords = self.chords_embedding(chords_seq)
            x_chords = self.chords_linear(x_chords.reshape(bs, max_bar_nums, -1))

            x_feat = []
            # (bs, max_bar_nums, 2)
            feat_seq_drums = feat_seq[:, :, :2, 0]
            x_feat_drums = torch.cat([self.drums_feat_embedding[0](feat_seq_drums[..., 0]),
                                      self.drums_feat_embedding[1](feat_seq_drums[..., 1])],
                                     dim=-1)
            # (bs, max_bar_nums, D)
            x_feat_drums = self.drums_feat_linear(x_feat_drums)
            x_feat.append(x_feat_drums)

            for i in range(1, TRACK_NUMS):
                # (bs, max_bar_nums, 4)
                feat_seq_others = feat_seq[:, :, :, i]
                x_feat_others = [self.others_feat_embedding[j](feat_seq_others[..., j]) for j in range(4)]
                x_feat_others = torch.cat(x_feat_others, dim=-1)
                x_feat_others = self.others_feat_linear(x_feat_others)
                x_feat.append(x_feat_others)
            # list：[(bs, max_bar_nums, D)] * TRACK_NUMS
            assert len(x_feat) == TRACK_NUMS

        if self.prior_info_flavor in ['latent', 'both']:
            bs = vq_codes.shape[0]
            max_bar_nums = vq_codes.shape[1]

            # list：[(bs, max_bar_nums, D)] * TRACK_NUMS
            x_vq_codes = [self.vq_codes_linear(self.vq_codes_embedding(vq_codes[..., i]).reshape(bs, max_bar_nums, -1))
                          for i in range(TRACK_NUMS)]

        if self.prior_info_flavor == 'meta':
            # features = [self.combined_prior(torch.cat([x_chords, x_feat[i]], dim=-1)) for i in range(1, TRACK_NUMS)]

            features = [x_feat[0]]
            for i in range(1, TRACK_NUMS):
                features.append(self.combined_prior2d(torch.cat([x_chords, x_feat[i]], dim=-1)))

        if self.prior_info_flavor == 'latent':
            features = x_vq_codes
        if self.prior_info_flavor == 'both':
            # features = [self.combined_prior(torch.cat([x_chords, x_feat[i], x_vq_codes[i]], dim=-1)) for i in range(TRACK_NUMS)]

            features = [self.combined_prior2d(torch.cat([x_feat[0], x_vq_codes[0]], dim=-1))]
            for i in range(1, TRACK_NUMS):
                features.append(self.combined_prior3d(torch.cat([x_chords, x_feat[i], x_vq_codes[i]], dim=-1)))

        # (bs, max_bar_nums, TRACK_NUMS, D)
        features = torch.stack(features, dim=2)

        # add gaussian noise, before encoder
        # features = (1 - self.blur) * features.clone() + \
        #            self.blur * torch.empty(features.shape, device=features.device).normal_(mean=0, std=1)

        # # === prior encoder ===
        # features = self.PosEmb.dropout(features + self.PosEmb.pe[:, :max_bar_nums, :].unsqueeze(2))
        # # (bs, max_bar_nums, TRACK_NUMS, D) => (bs*max_bar_nums, TRACK_NUMS, D)
        # features = features.reshape(-1, TRACK_NUMS, self.D)
        # features = self.track_encoder(features)
        # # Global Average Pooling(Average) all Tracks
        # features = torch.mean(features, dim=1)
        # features = self.track_pooler(features)
        # # (bs, max_bar_nums, D)
        # features = features.reshape(-1, max_bar_nums, self.D)
        # features = self.PosEmb(features)
        # features_length_mask = LengthMask(features_len, max_len=max_bar_nums, device=features_len.device)
        # features = self.time_encoder(features, attn_mask=None, length_mask=features_length_mask)

        # === 更改版 ====
        features = self.PosEmb.dropout(features + self.PosEmb.pe[:, :max_bar_nums, :].unsqueeze(2))
        # (bs, max_bar_nums, TRACK_NUMS, D) => (bs, TRACK_NUMS, max_bar_nums, D) => (bs*TRACK_NUMS, max_bar_nums, D)
        features = features.permute(0, 2, 1, 3).reshape(-1, max_bar_nums, self.D)
        features_length_mask = LengthMask(features_len.repeat_interleave(TRACK_NUMS),
                                          max_len=max_bar_nums,
                                          device=features_len.device)
        features = self.time_encoder(features, attn_mask=None, length_mask=features_length_mask)

        features = features.reshape(-1, TRACK_NUMS, max_bar_nums, self.D).permute(0, 2, 1, 3)

        # # (bs, max_bar_nums, D)
        # return features

        # (bs, max_bar_nums, TRACK_NUMS, D)
        return features

    def forward(self, x,
                track_name,
                chords_seq, feat_seq, vq_codes,
                bar_embed_ids, bar_ids, bar_len,
                batch_mask,
                ):
        # x & batch_mask: (bs, max_seq_len, TRACK_NUMS)
        # track_name: (bs, TRACK_NUMS) LONG

        # bar_embed_ids: (bs, max_seq_len, TRACK_NUMS)
        # bar_ids: (bs, 64, TRACK_NUMS); bar_len: (bs), shared

        # chords_seq: (bs, max_bar_nums, TRACK_NUMS), shared; max_bar_nums: batch中最大小节数
        # feat_seq: (bs, max_bar_nums, 4, TRACK_NUMS) drums=>2维 feat_seq[:, :, :2, :]，其余padding，其他轨=>4维
        # vq_codes: (bs, max_bar_nums, 16, TRACK_NUMS), 每个轨16维

        bs = x.shape[0]
        max_seq_len = x.shape[1]
        max_bar_nums = chords_seq.shape[1]  # 已padding，32/64

        # === encoder ===
        # (bs, max_bar_nums, TRACK_NUMS, D)
        features = self.processing_features(bar_len, chords_seq, feat_seq, vq_codes)

        # (bs, TRACK_NUMS, max_bar_nums, max_bar_nums)
        if self.masking_type == 'adaptive':
            sim_attns = self.cal_sim_attns(features, bar_len)
            custom_attns = []
            for track_i in range(TRACK_NUMS):
                sim_attns_per = sim_attns[:, track_i, :, :]
                custom_attns_per = self.expand_sim_attns_per(x[:, :, track_i], sim_attns_per,
                                                             bar_ids[:, :, track_i], bar_len)
                custom_attns.append(custom_attns_per)

        # # Global Average Pooling(Average) all Tracks
        # features = torch.mean(features, dim=2)
        # # (bs, max_bar_nums, D)
        # features = self.track_pooler(features)

        # === Embedding ===
        # (bs, max_seq_len, TRACK_NUMS, D)
        x_event = self.token_embedding(x)
        x_event += self.bar_embedding(bar_embed_ids)

        # (bs, TRACK_NUMS, D)
        track_embeded = self.track_embedding(track_name)
        x_event[:, 0, :, :] += track_embeded

        # === bottom decoder ===
        seq_triangular_mask = TriangularCausalMask(max_seq_len, device=x.device)
        x_bottom = []
        x_bar_fusion = torch.zeros(bs, max_bar_nums, TRACK_NUMS, self.D, device=x.device)
        for track_i in range(TRACK_NUMS):
            # (bs, max_seq_len, D)
            x_event_per = x_event[:, :, track_i, :]
            # (bs, max_seq_len)
            batch_mask_per = batch_mask[:, :, track_i]

            # (bs, max_bar_nums, D)
            features_per = features[:, :, track_i, :]

            # (bs, 64)
            bar_ids_per = bar_ids[:, :, track_i]

            if self.masking_type == 'original':
                custom_attns_per = None
            elif self.masking_type == 'adaptive':
                custom_attns_per = custom_attns[track_i]
            else:
                raise ValueError('error masking type.')

            x_event_per = self.PosEmb(x_event_per)
            seq_length_mask_per = LengthMask(torch.sum(batch_mask_per, dim=-1), max_len=max_seq_len, device=x.device)
            features_length_mask = LengthMask(bar_len, max_len=max_bar_nums, device=x.device)

            x_bottom_per = self.forward_decoder(x_event_per, features_per,
                                                # x_event_per, features,
                                                self.bottom_decoder,
                                                x_mask=seq_triangular_mask, x_length_mask=seq_length_mask_per,
                                                memory_mask=None, memory_length_mask=features_length_mask,
                                                custom_attns=custom_attns_per,
                                                )
            x_bottom.append(x_bottom_per)

            for bs_i in range(bs):
                x_bar_fusion[bs_i, :bar_len[bs_i], track_i, :] = x_bottom_per[bs_i, bar_ids_per[bs_i, :bar_len[bs_i]], :]

        # (bs, max_seq_len, TRACK_NUMS, D)
        x_bottom = torch.stack(x_bottom, dim=2)

        # === fusion decoder ===
        x_bar_fusion = self.PosEmb.dropout(x_bar_fusion + self.PosEmb.pe[:, :max_bar_nums, :].unsqueeze(2))
        # (bs, max_bar_nums, TRACK_NUMS, D) => (bs*max_bar_nums, TRACK_NUMS, D)
        x_bar_fusion = x_bar_fusion.reshape(bs * max_bar_nums, -1, self.D)
        x_bar_fusion = self.fusion_decoder(x_bar_fusion)

        x_bar_fusion = x_bar_fusion.reshape(bs, max_bar_nums, -1, self.D)

        # === decoder ===
        y_event = []
        for track_i in range(TRACK_NUMS):
            # (bs, max_seq_len, D)
            x_bottom_per = x_bottom[:, :, track_i, :]
            # (bs, max_seq_len)
            batch_mask_per = batch_mask[:, :, track_i]

            # (bs, max_bar_nums, D)
            features_per = features[:, :, track_i, :]

            # (bs, 64)
            bar_ids_per = bar_ids[:, :, track_i]

            if self.masking_type == 'original':
                custom_attns_per = None
            elif self.masking_type == 'adaptive':
                custom_attns_per = custom_attns[track_i]
            else:
                raise ValueError('error masking type.')

            # update x_bottom
            for bs_i in range(bs):
                x_bottom_per[bs_i, bar_ids_per[bs_i, :bar_len[bs_i]], :] = x_bar_fusion[bs_i, :bar_len[bs_i], track_i, :]

            x_bottom_per = self.PosEmb(x_bottom_per)
            seq_length_mask_per = LengthMask(torch.sum(batch_mask_per, dim=-1), max_len=max_seq_len, device=x.device)
            features_length_mask = LengthMask(bar_len, max_len=max_bar_nums, device=x.device)

            x_top_per = self.forward_decoder(x_bottom_per, features_per,
                                             # x_bottom_per, features,
                                             self.top_decoder,
                                             x_mask=seq_triangular_mask, x_length_mask=seq_length_mask_per,
                                             memory_mask=None, memory_length_mask=features_length_mask,
                                             custom_attns=custom_attns_per,
                                             )
            y_event_per = self.project_linear[track_i](x_top_per)
            y_event.append(y_event_per)

        return y_event

    # ============== GENERATION / INFERENCE ==============
    # def generate_perbar(self, x, track_embeded, features, features_len, sim_attns):
    #     # features是已经mean后的
    def generate_perbar(self, x, track_embeded, features, features_len, sim_attns):
        # features: (bs, max_bar_nums, TRACK_NUMS, D)
        # 生成所有轨的新的单个小节

        bs = features.shape[0]
        assert bs == 1
        max_bar_nums = features.shape[1]

        # x 为所有轨道的完整小节list, 所有轨道均以Bar_XXX token结尾
        # 先获取 x_bar_fusion
        x_bar_fusion = torch.zeros(1, max_bar_nums, TRACK_NUMS, self.D, device=x[0].device)
        custom_attns = []
        for track_i in range(TRACK_NUMS):
            x_per = x[track_i]
            track_embeded_per = track_embeded[:, track_i, :]
            curr_bar_embed_ids, curr_bar_ids, curr_bar_len = get_struct_info_track_without_phrase(x_per)
            curr_seq_len = x_per.shape[1]

            features_per = features[:, :, track_i, :]

            if self.masking_type == 'original':
                custom_attns_per = None
            elif self.masking_type == 'adaptive':
                sim_attns_per = sim_attns[:, track_i, :, :]
                custom_attns_per = self.expand_sim_attns_per(x_per, sim_attns_per,
                                                             curr_bar_ids, curr_bar_len,
                                                             )
            else:
                raise ValueError('error masking type.')
            custom_attns.append(custom_attns_per)

            x_event_per = self.token_embedding(x_per)
            x_event_per += self.bar_embedding(curr_bar_embed_ids)
            x_event_per[:, 0, :] += track_embeded_per

            x_event_per = self.PosEmb(x_event_per)
            seq_triangular_mask = TriangularCausalMask(curr_seq_len, device=x_per.device)
            features_length_mask = LengthMask(features_len, max_len=max_bar_nums, device=x_per.device)

            x_bottom_per = self.forward_decoder(x_event_per, features_per,
                                                # x_event_per, features,
                                                self.bottom_decoder,
                                                x_mask=seq_triangular_mask, x_length_mask=None,
                                                memory_mask=None, memory_length_mask=features_length_mask,
                                                custom_attns=custom_attns_per,
                                                )

            x_bar_fusion[0, :curr_bar_len[0], track_i, :] = x_bottom_per[0, curr_bar_ids[0, :curr_bar_len[0]], :]

        # === fusion decoder ===
        x_bar_fusion = self.PosEmb.dropout(x_bar_fusion + self.PosEmb.pe[:, :max_bar_nums, :].unsqueeze(2))
        # (bs, max_bar_nums, TRACK_NUMS, D) => (bs*max_bar_nums, TRACK_NUMS, D)
        x_bar_fusion = x_bar_fusion.reshape(bs * max_bar_nums, -1, self.D)
        x_bar_fusion = self.fusion_decoder(x_bar_fusion)

        x_bar_fusion = x_bar_fusion.reshape(bs, max_bar_nums, -1, self.D)


        # 正式的生成步骤
        for track_i in range(TRACK_NUMS):
            x_per = x[track_i]
            track_embeded_per = track_embeded[:, track_i, :]
            _, curr_bar_ids, curr_bar_len = get_struct_info_track_without_phrase(x_per)
            if self.masking_type == 'original':
                custom_attns_per = None
            elif self.masking_type == 'adaptive':
                custom_attns_per = custom_attns[track_i]
            else:
                raise ValueError('error masking type.')

            features_per = features[:, :, track_i, :]

            step = 0
            while True:
                curr_seq_len = x_per.shape[1]
                curr_bar_embed_ids = infer_embed_ids(curr_bar_ids, curr_seq_len)
                x_event_per = self.token_embedding(x_per)
                x_event_per += self.bar_embedding(curr_bar_embed_ids)
                x_event_per[:, 0, :] += track_embeded_per

                x_event_per = self.PosEmb(x_event_per)
                seq_triangular_mask = TriangularCausalMask(curr_seq_len, device=x_per.device)
                features_length_mask = LengthMask(features_len, max_len=features.shape[1], device=features.device)

                # == bottom decoder ===
                x_bottom_per = self.forward_decoder(x_event_per, features_per,
                                                    # x_event_per, features,
                                                    self.bottom_decoder,
                                                    x_mask=seq_triangular_mask, x_length_mask=None,
                                                    memory_mask=None, memory_length_mask=features_length_mask,
                                                    custom_attns=custom_attns_per,
                                                    )
                # update x_bottom
                x_bottom_per[0, curr_bar_ids[0, :curr_bar_len[0]], :] = x_bar_fusion[0, :curr_bar_len[0], track_i, :]

                x_bottom_per = self.PosEmb(x_bottom_per)

                # === top decoder ===
                x_top_per = self.forward_decoder(x_bottom_per, features_per,
                                                 # x_bottom_per, features,
                                                 self.top_decoder,
                                                 x_mask=seq_triangular_mask, x_length_mask=None,
                                                 memory_mask=None, memory_length_mask=features_length_mask,
                                                 custom_attns=custom_attns_per,
                                                 )
                y_event_per = self.project_linear[track_i](x_top_per)
                y_event_per = grammar_control_without_phrase(x_per, y_event_per)
                y_sampling_per = sampling_v2(y_event_per[:, -1, :], k_p=0.02, mode='topk')
                x_per = torch.cat([x_per, y_sampling_per], dim=1)
                step += 1

                # 单小节长度超过256自动停止，并报错
                if step > 256:
                    raise ValueError('generating wrong bar, without new bar_tokens.')
                if y_sampling_per.detach().cpu().numpy() in [5, 6]:
                    break
            x[track_i] = x_per

        return x


    def compute_loss(self, pred_events, y_gt, batch_mask):
        # (bs, max_seq_len, event_class_num, TRACK_NUMS)
        pred_events = torch.stack(pred_events, dim=-1)
        # (bs, event_class_num, max_seq_len, TRACK_NUMS)
        pred_events = pred_events.permute(0, 2, 1, 3)

        loss = self.loss_func(pred_events, y_gt)
        loss = torch.sum(loss * batch_mask) / torch.sum(batch_mask)

        return loss



