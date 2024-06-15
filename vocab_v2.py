import pretty_midi
from collections import Counter
import torchtext
from torch import Tensor


from constants import (
  DEFAULT_VELOCITY_BINS,
  DEFAULT_DURATION_BINS,
  DEFAULT_TEMPO_BINS,

  DEFAULT_DRUMS_PITCH_TYPE_BINS,
  DEFAULT_DRUMS_NOTE_DENSITY_BINS,
  # DEFAULT_DRUMS_MEAN_VELOCITY_BINS,

  DEFAULT_NOTE_DENSITY_BINS,
  DEFAULT_MEAN_VELOCITY_BINS,
  DEFAULT_MEAN_PITCH_BINS,
  DEFAULT_MEAN_DURATION_BINS,

  DEFAULT_POS_PER_QUARTER,
  DEFAULT_RESOLUTION,
  MAX_TIME_SHIFT,
  
  Ins_LIST,
  Ins_ID,
  DRUM_PITCH
  
)

from constants import (
  MAX_BAR_LENGTH,
  MAX_N_BARS,

  PAD_TOKEN,
  UNK_TOKEN,
  BOS_TOKEN,
  EOS_TOKEN,
  MASK_TOKEN,
)

class REMI_Tokens:
    def get_instrument_tokens(key='Instrument'):
        # instrument固定为6种：Melody, Piano, Strings, Bass, Guitar, Drums
        tokens = [f'{key}_{ins}' for ins in Ins_LIST]
        return tokens
    # 和弦识别算法更改为Chorder, 其内置和弦类型为11种，故更新之
    def get_chord_tokens(key='Chord', qualities = ['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'dim7', 'hd7', 'sus2', 'sus4']):
        pitch_classes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']

        chords = [f'{root}:{quality}' for root in pitch_classes for quality in qualities]
        chords.append('N:N')

        tokens = [f'{key}_{chord}' for chord in chords]
        return tokens
    
    def get_time_signature_tokens(key='Time Signature'):
        # only consider 4/4
        time_sigs = ['4/4']
        tokens = [f'{key}_{time_sig}' for time_sig in time_sigs]
        return tokens
    
    def get_midi_tokens(
                        instrument_key='Instrument', 
                        #time_signature_key='Time Signature',
                        pitch_key='Pitch',
                        velocity_key='Velocity',
                        duration_key='Duration',
                        #tempo_key='Tempo',
                        bar_key='Bar',
                        phrase_key='Phrase',
                        BCD_key='BCD',
                        position_key='Position',
                        do_include_velocity = True,
                        ):
        instrument_tokens = REMI_Tokens.get_instrument_tokens(instrument_key)
        
        #pitch_durm_音高值31种,在DRUM_PITCH中
        pitch_tokens = [f'{pitch_key}_{i}' for i in range(128)] + [f'{pitch_key}_drum_{i}' for i in DRUM_PITCH]
        velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
        duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
        #tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
        
        # phrase token
        phrase_tokens = [f'{phrase_key}_{i}' for i in ['Lower', 'Upper']]
        
        # V2版本添加，Bar_CountDown => BCD token, 取值为BCD_16 -> BCD_1,共16种
        BCD_tokens = [f'{BCD_key}_{i}' for i in range(1, 17)]

        # bar token改为只有Bar_Normal, Bar_Empty两种
        bar_tokens = [f'{bar_key}_{i}' for i in ['Normal', 'Empty']]
        position_tokens = [f'{position_key}_{i}' for i in range(4*DEFAULT_RESOLUTION)]

        # time_sig_tokens = REMI_Tokens.get_time_signature_tokens(time_signature_key)
        
        # new token type, indicate how many empty bars segment in a track.
        # semiend: non-empty bars behind the marker
        # fullend: none non-empty bars
        # 仅在multiple sequence表示中使用
        #emptybars_marker_token = [f'emptybars_{i}' for i in ['0', '1', '2', '3', 'semiend', 'fullend']]
        
        # new token type, summary_token
        # 暂不使用
        #summary_token = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)] 

        if do_include_velocity:
            return (
                    bar_tokens +
                    phrase_tokens +
                    BCD_tokens + 
                    position_tokens +
                    # emptybars_marker_token +
                    # summary_token
                    # time_sig_tokens +
                    # tempo_tokens +
                    instrument_tokens +
                    pitch_tokens +
                    duration_tokens +
                    velocity_tokens
            )
        else:
            return (
                    bar_tokens +
                    phrase_tokens +
                    BCD_tokens + 
                    position_tokens +
                    # emptybars_marker_token +
                    # summary_token
                    # time_sig_tokens +
                    # tempo_tokens +
                    instrument_tokens +
                    pitch_tokens +
                    duration_tokens
                    # velocity_tokens
            )

class Vocab:
    def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN], unk_token=UNK_TOKEN):
        self.vocab = torchtext.vocab.vocab(counter)

        self.specials = specials
        for i, token in enumerate(self.specials):
            self.vocab.insert_token(token, i)
    
        if unk_token in specials:
            self.vocab.set_default_index(self.vocab.get_stoi()[unk_token])

    def to_i(self, token):
        return self.vocab.get_stoi()[token]
    
    def to_s(self, idx):
        if idx >= len(self.vocab):
            return UNK_TOKEN
        else:
            return self.vocab.get_itos()[idx]

    def __len__(self):
        return len(self.vocab)

    def encode(self, seq):
        return self.vocab(seq)

    def decode(self, seq):
        if isinstance(seq, Tensor):
            seq = seq.numpy()
        return self.vocab.lookup_tokens(seq)

# REMI_like vocab
class RemiVocab(Vocab):
    def __init__(self, chord_enable=False, velocity_enable=True):
        midi_tokens = REMI_Tokens.get_midi_tokens(do_include_velocity=velocity_enable)
        chord_tokens = REMI_Tokens.get_chord_tokens()
    
        if chord_enable:
            self.tokens = midi_tokens + chord_tokens
        else:
            self.tokens = midi_tokens

        counter = Counter(self.tokens)
        super().__init__(counter)

class Description_Tokens():
    
    def get_bar_tokens(key='Bar'):
        return [f'{key}_{i}' for i in ['Normal', 'Empty']]
    
    # Drums features
    def drum_pitch_type_tokens(key='Drums_Pitch_Type'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_DRUMS_PITCH_TYPE_BINS))]
    def drum_note_density_tokens(key='Drums_Note_Density'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_DRUMS_NOTE_DENSITY_BINS))]
    # def drum_mean_velocity_tokens(key='Drums_Mean_Velocity'):
    #     return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_DRUMS_MEAN_VELOCITY_BINS))]
    
    # other track features
    def note_density_tokens(key='Note_Density'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
    def mean_pitch_tokens(key='Mean_Pitch'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
    def mean_duration_tokens(key='Mean_Duration'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))]
    def mean_velocity_tokens(key='Mean_Velocity'):
        return [f'{key}_NaN'] + [f'{key}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]

class DescriptionVocab_Bar(Vocab):
    # 7 5+2
    def __init__(self):
        self.tokens = Description_Tokens.get_bar_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter) 
# ====

class DescriptionVocab_DPT(Vocab):
    # 37 5+32
    def __init__(self):
        self.tokens = Description_Tokens.drum_pitch_type_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)

class DescriptionVocab_DND(Vocab):
    # 55 5+50
    def __init__(self):
        self.tokens = Description_Tokens.drum_note_density_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)

# class DescriptionVocab_DMV(Vocab):
#     # 39 5+34
#     def __init__(self):
#         self.tokens = Description_Tokens.drum_mean_velocity_tokens()
#         counter = Counter(self.tokens)
#         super().__init__(counter)
# =====

class DescriptionVocab_ND(Vocab):
    # 71 5+66
    def __init__(self):
        self.tokens = Description_Tokens.note_density_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)

class DescriptionVocab_MP(Vocab):
    # 39 5+34
    def __init__(self):
        self.tokens = Description_Tokens.mean_pitch_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)

class DescriptionVocab_MD(Vocab):
    # 35 5+30
    def __init__(self):
        self.tokens = Description_Tokens.mean_duration_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)
        
class DescriptionVocab_MV(Vocab):
    # 39 5+34
    def __init__(self):
        self.tokens = Description_Tokens.mean_velocity_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)
# =====

class DescriptionVocab_Chord(Vocab):
    # 138 5+133
    def __init__(self):
        self.tokens = REMI_Tokens.get_chord_tokens()
        counter = Counter(self.tokens)
        super().__init__(counter)