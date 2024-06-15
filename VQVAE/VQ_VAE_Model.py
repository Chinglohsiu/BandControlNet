
# for pretraining
# bar level
# phrase level

# single-sequence : (bs, dim)
# multi_sequence  : (bs*tracks, dim)

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import transformers
from transformers import (
  BertConfig,
  EncoderDecoderConfig,
  EncoderDecoderModel
)

from utils.model_utils import Embeddings, sampling

# refer to FIGARO: https://github.com/dvruette/figaro/blob/main/src/models/vae.py
class VectorQuantizeEMA(nn.Module):
    def __init__(self,
                 d_latent,
                 n_codes,
                 n_groups,           # dimension splitted into n_groups
                 decay=0.995, eps=1e-4, restart_threshold=0.99,
                 ):
        assert d_latent // n_groups == d_latent / n_groups

        super(VectorQuantizeEMA, self).__init__()

        self.d_latent = d_latent
        self.n_groups = n_groups
        self.dim = d_latent // n_groups
        self.n_codes = n_codes

        self.decay = decay
        self.eps = eps
        self.threshold = restart_threshold
        self.init = False

        vq_embedding = torch.randn(self.n_codes, self.dim)
        self.register_buffer('embedding', vq_embedding)
        self.register_buffer('cluster_size', torch.ones(self.n_codes))
        self.register_buffer('cluster_sum', vq_embedding.clone().detach())

    def get_code_indices(self, x_):
        # x_ : (X, dim)

        # L2 distance
        emb_t = self.embedding.t()
        distance = (
            x_.pow(2).sum(1, keepdims=True)
            - 2*x_ @ emb_t
            + emb_t.pow(2).sum(0, keepdims=True)
        )

        _, embed_idx = (-distance).max(1)

        return embed_idx

    def quantize(self, embed_idx):
        return F.embedding(embed_idx, self.embedding)

    def forward(self, x, dist=None):
        assert x.shape[-1] == self.n_groups * self.dim
        x_ = x.reshape(-1, self.dim)

        if self.training and not self.init:
            self._init_embeddings(x_, dist=dist)

        embed_idx = self.get_code_indices(x_)
        embed_onehot = F.one_hot(embed_idx, self.n_codes).type(x_.dtype)

        # (-1, d_latent)
        quantize = self.quantize(embed_idx).view(-1, self.n_groups*self.dim)
        # mse loss
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()
        codes = embed_idx.view(-1, self.n_groups)

        if self.training:
            update_metrics = self._ema_update(x_, embed_onehot, dist=dist)
        else:
            update_metrics = {}

        # computing ppl both for training and valid
        avg_pr = embed_onehot.sum(0)
        avg_pr = avg_pr / avg_pr.sum()
        ppl = -(avg_pr * (avg_pr + 1e-5).log()).sum()

        return dict(
            z=quantize,
            diff=diff,
            codes=codes,
            ppl=ppl,
            **update_metrics,
        )

    def _init_embeddings(self, x, dist=None):
        self.init = True
        rand_centers = self._randomize(x)
        self.cluster_sum.data.copy_(rand_centers)
        self.cluster_size.data.fill_(1)

    def _randomize(self, x):
        n = x.size(0)
        if n < self.n_codes:
            r = (self.n_codes + n - 1) // n
            std = 0.01 / np.sqrt(self.dim)
            x = x.repeat(r, 1)
            x += std * torch.randn_like(x)
        return x[torch.randperm(x.size(0))][:self.n_codes]

    def _ema_update(self, x, cluster_assign, dist=None):
        # cluster_assign => onehot tensor of embed_idx
        with torch.no_grad():
            cluster_size = cluster_assign.sum(0)
            cluster_sum = cluster_assign.t() @ x

            rand_centers = self._randomize(x)

            # EMA update step
            self.cluster_size.data.copy_(self.decay * self.cluster_size + (1 - self.decay) * cluster_size)
            self.cluster_sum.data.copy_(self.decay * self.cluster_sum + (1 - self.decay) * cluster_sum)

            used = (self.cluster_size >= self.threshold).float().unsqueeze(-1)

            n = self.cluster_size.sum()
            # Use additive smoothing to mitigate exploding gradients
            count = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n

            cluster_centers = self.cluster_sum / count.unsqueeze(-1)
            cluster_centers = used * cluster_centers + (1 - used) * rand_centers
            self.embedding.data.copy_(cluster_centers)

            # Compute metrics
            avg_usage = used.mean()
            usage = used.sum()
            # pr = cluster_size / cluster_size.sum()
            #
            # # PPL, log format
            # entropy = -(pr * (pr + 1e-5).log()).sum()

        return {
            'avg_usage': avg_usage,
            'usage': usage,
            # 'entropy': entropy,
        }


# VQ-VAE Main Module
class VQVAE(nn.Module):
    def __init__(self, args, vocab_size):
        super(VQVAE, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.D = args.d_model
        self.n_head = args.n_head

        self.encoder_layers = args.encoder_layers
        self.decoder_layers = args.decoder_layers

        self.encoder_mlp = args.encoder_mlp
        self.decoder_mlp = args.decoder_mlp

        self.max_position_embeddings = args.max_position_embeddings
        self.max_seq_len = args.max_seq_len

        self.n_codes = args.n_codes
        self.n_groups = args.n_groups
        self.d_latent = args.d_latent

        self.beta = args.commitment_cost

        encoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.D,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.n_head,
            intermediate_size=self.encoder_mlp,
            max_position_embeddings=self.max_position_embeddings,
            position_embedding_type='relative_key_query'
        )
        decoder_config = BertConfig(
            vocab_size=1,
            pad_token_id=0,
            hidden_size=self.D,
            num_hidden_layers=self.decoder_layers,
            num_attention_heads=self.n_head,
            intermediate_size=self.decoder_mlp,
            max_position_embeddings=self.max_position_embeddings,
            position_embedding_type='relative_key_query'
        )

        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.transformer = EncoderDecoderModel(config)
        self.transformer.config.decoder.is_decoder = True
        self.transformer.config.decoder.add_cross_attention = True
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder

        self.token_embedding = Embeddings(self.vocab_size, self.D, padding_idx=0)
        self.logits_proj = nn.Linear(self.D, self.vocab_size)

        self.vq_embed = VectorQuantizeEMA(self.d_latent, self.n_codes, self.n_groups)
        self.D2latent = nn.Linear(self.D, self.d_latent, bias=False)
        self.latent2D = nn.Linear(self.d_latent, self.D, bias=False)
        self.attention_project = nn.Linear(self.D, self.D)

        self.loss_func = nn.CrossEntropyLoss(ignore_index=0)

    def encode(self, x):
        # x: bar-level or phrase-level token_id sequences =>  [bos, <seq>] with masked bar-token
        # (bs, X)
        x_embed = self.token_embedding(x)

        out = self.encoder(inputs_embeds=x_embed, output_hidden_states=True)
        # (bs, D)
        encoded_out = out.pooler_output
        # (bs, d_latent)
        encoded_out = self.D2latent(encoded_out)

        latent_info_dict = self.vq_embed(encoded_out, dist=None)

        return latent_info_dict

    def decode(self, x, quantized):
        # decode input == encode input => [bos, <seq>]
        # x: bar-level or phrase-level token_id sequences =>  [bos, <seq>] with masked bar-token
        # gt: [<seq>, eos]

        # quantized: (bs, d_latent)

        x_embed = self.token_embedding(x)
        # max_seq_len among the batch
        max_seq = x_embed.size(1)

        init_state = self.latent2D(quantized)

        # add quantized latent to every step of decode input
        x_embed += init_state.unsqueeze(1).repeat(1, max_seq, 1)

        # quantized latent as the memory for cross-attention
        memory = self.attention_project(init_state.unsqueeze(1).repeat(1, self.max_seq_len, 1))

        # padding the length of decode input to max_seq_len
        padding = torch.zeros_like(memory)
        padding[:, :x_embed.size(1)] = x_embed
        x_embed = padding

        decoder_attention_mask = self.get_attention_mask(x)
        # padding the length of mask to max_seq_len
        padding = torch.zeros((x.size(0), self.max_seq_len, self.max_seq_len), device=x.device, dtype=torch.int)
        padding[:, :decoder_attention_mask.size(1), :decoder_attention_mask.size(2)] = decoder_attention_mask
        decoder_attention_mask = padding

        out = self.decoder(
            inputs_embeds=x_embed,
            encoder_hidden_states=memory,
            attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )

        decode_out = out.hidden_states[-1][:, :max_seq]
        logits = self.logits_proj(decode_out).contiguous()

        return logits

    def get_attention_mask(self, x, mask_size=1, k=1):
        batch_size, seq_len = x.shape[:2]

        # Standard self-attention mask for auto-regressive modelling
        tri_mask = torch.ones((seq_len // mask_size + 1, seq_len // mask_size + 1), device=x.device, dtype=torch.int)
        tri_mask = torch.triu(tri_mask, diagonal=k)
        tri_mask = (~tri_mask.bool()).int()
        # Create windowed self-attention mask, forcing the model to prefict farther into the future
        window_mask = tri_mask.repeat_interleave(mask_size, dim=0).repeat_interleave(mask_size, dim=1)[:seq_len,
                      :seq_len]
        # First token needs to be always visible
        window_mask[:, 0] = 1

        return window_mask.unsqueeze(0).repeat(batch_size, 1, 1)

    def forward(self, x):
        latent_info_dict = self.encode(x)
        quantized = latent_info_dict['z']
        logits = self.decode(x, quantized)
        return {
            'logits': logits,
             **latent_info_dict
        }

    def inference(self, x, vq_codes):
        # x: (1, 1) or (1, X)
        # vq_codes: (1, 16) 单小节的vq_codes
        quantized = self.vq_embed.quantize(vq_codes).view(-1, self.vq_embed.n_groups * self.vq_embed.dim)

        predited_logits = self.decode(x, quantized)
        predited_sampling = torch.as_tensor(sampling(predited_logits[:, -1, :]), device=x.device).unsqueeze(0)
        return predited_sampling.unsqueeze(0)

    def compute_loss(self, x, gt, loss_mask=None):
        # x: (bs, max_seq)
        # gt: (bs, max_seq)
        # loss_mask: (bs, max_seq)

        out = self.forward(x)

        # (bs, max_seq, vocab_size)
        # logits = out['logits']
        # logits = logits.permute(0, -1, 1)

        # rec_loss = self.loss_func(logits, gt)
        # rec_loss = torch.sum(rec_loss * loss_mask) / torch.sum(loss_mask)

        logits = out['logits']
        logits = logits.view(-1, logits.size(-1))
        gt = gt.view(-1)
        rec_loss = self.loss_func(logits, gt)

        vq_loss = out['diff']
        loss = rec_loss + self.beta*vq_loss

        if self.training:
            # ppl = out['entropy']
            avg_usage = out['avg_usage']
        else:
            # ppl = None
            avg_usage = None

        return {
            'logits': out['logits'],
            'loss': loss,
            'rec_loss': rec_loss,
            'vq_loss': vq_loss,
            'ppl': out['ppl'],
            'avg_usage': avg_usage,
        }









