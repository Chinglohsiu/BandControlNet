import torch


def infer_embed_ids(cover_ids, seq_len):
    embed_ids = torch.bincount(cover_ids.squeeze(0), minlength=seq_len)
    embed_ids = torch.cumsum(embed_ids, dim=0).unsqueeze(0)
    return embed_ids

def get_struct_info_track_without_phrase(x):
    assert x.size(0) == 1
    seq_len = x.size(1)
    assert seq_len > 2, '至少含有一个Bar_XXX, 使得bar_ids, phrase_ids不为空'

    bar_ids = []

    for i in range(seq_len):
        if x[:, i].detach().item() in [5, 6]:
            bar_ids.append(i)

    bar_len = torch.tensor(len(bar_ids), dtype=torch.long, device=x.device).unsqueeze(0)
    bar_ids = torch.tensor(bar_ids, dtype=torch.long, device=x.device)
    bar_embed_ids = torch.bincount(bar_ids, minlength=seq_len)
    bar_embed_ids = torch.cumsum(bar_embed_ids, dim=0).unsqueeze(0).to(x.device)

    return bar_embed_ids, bar_ids.unsqueeze(0), bar_len


def grammar_control_without_phrase(x, y_event):
    # 无 < 'Phrase_XXX' >, < 'BCD_XXX' >，用<'mask'>代替
    # Bar_XXX => [5, 6]
    # Phrase_XXX => [7, 8], BCD_XXX => [9, 10, 11, ..., 23, 24]
    # Position_XXX => [25, 26, 27, ..., 71, 72]
    if x[:, -1].detach().cpu().numpy() in [5, 6]:
        # [5,6]后接4
        y_event[:, -1, :4] = -float('inf')  # 屏蔽其他specials
        # 只空出了[4] => '<mask>'
        y_event[:, -1, 5:] = -float('inf')
    if x[:, -1].detach().cpu().numpy() in [4]:
        # [4]后接 Bar_XXX 或者 Position_XXX
        y_event[:, -1, :5] = -float('inf')  # 屏蔽 specials
        # 空出了[5:7] => 'Bar_XXX'
        y_event[:, -1, 7:25] = -float('inf')  # 屏蔽 Phrase_XXX, BCD_XXX
        # 空出了[25:73] => Position_XXX
        y_event[:, -1, 73:] = -float('inf')

    return y_event