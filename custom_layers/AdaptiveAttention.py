# add custom_attns for original self-attention

from math import sqrt

import torch
from torch.nn import Dropout, Module
import torch.nn.functional as F

from fast_transformers.attention_registry import AttentionRegistry, Optional, Float, EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent


class AdaptiveAttention(Module):
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=''):
        super(AdaptiveAttention, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths, custom_attns):
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1. / sqrt(E)

        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)    # (N, H, L, S)

        if custom_attns is not None:
            if len(custom_attns.shape) == 4:
                QK = QK * custom_attns
            else:
                QK = QK * custom_attns[:, None]

        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        return V.contiguous()

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "ada", AdaptiveAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)




