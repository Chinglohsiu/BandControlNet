
import torch
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.masking import FullMask, LengthMask, TriangularCausalMask
from fast_transformers.events import EventDispatcher
import copy

class AdaptiveTransformerDecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model,
                 d_ff=None, dropout=0.1, event_dispatcher='',
                 masking_type='original',
                 ):
        super(AdaptiveTransformerDecoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.masking_type = masking_type

        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, memory, x_mask=None, x_length_mask=None,
                memory_mask=None, memory_length_mask=None,
                custom_attns=None,
                ):
        bs = x.shape[0]
        seq_len = x.shape[1]
        max_bar_nums = memory.shape[1]

        if self.masking_type == 'original':
            assert custom_attns == None
        if self.masking_type == 'adaptive':
            assert custom_attns is not None

        # self-attention
        x_length_mask = x_length_mask or \
                        LengthMask(x.new_full((bs,), seq_len, dtype=torch.long))

        x_ori = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask,
                query_lengths=None,
                key_lengths=x_length_mask,
                custom_attns=None,
        ))
        x_ori = self.norm1(x_ori)

        # cross-attention
        memory_mask = memory_mask or FullMask(seq_len, max_bar_nums, device=x.device)
        memory_length_mask = memory_length_mask or \
                             LengthMask(x.new_full((bs,), max_bar_nums, dtype=torch.int64))
        x = x_ori + self.dropout(self.cross_attention(
            x_ori, memory, memory,
            attn_mask=memory_mask,
            query_lengths=x_length_mask,
            key_lengths=memory_length_mask,
        ))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm3(x + y)






