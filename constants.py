

import numpy as np

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

Ins_LIST = ['Drums', 'Acoustic Grand Piano',
            'Acoustic Guitar (nylon)', 'Acoustic Bass',
            'String Ensemble 1', 'Lead 1 (square)']

Ins_ID = [-1, 0, 24, 32, 48, 80]

DRUM_PITCH = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
              46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
              57, 59, 60, 61, 62, 63, 64, 69, 70]

# ==========================================================================
# ==========================================================================


DEFAULT_POS_PER_QUARTER = 12

DEFAULT_DURATION_BINS = np.sort(np.concatenate([
  np.arange(1, 13), # smallest possible units up to 1 quarter
  np.arange(12, 24, 3)[1:], # 16th notes up to 0.5 bar
  np.arange(12, 24, 4)[1:], # triplets up to 0.5 bar
  np.arange(24, 48, 6), # 8th notes up to 1 bars
  np.arange(48, 4*48, 12), # quarter notes up to 4 bars
  #np.arange(4*48, 16*48+1, 24) # half notes up to 16 bars
]))

DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32+1, dtype=np.int16)
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int16)


# 鼓轨音高种类
DEFAULT_DRUMS_PITCH_TYPE_BINS = np.arange(1, len(DRUM_PITCH)+1)  # 第1种到第31种，没有用NaN表示
# 鼓轨音符密度
DEFAULT_DRUMS_NOTE_DENSITY_BINS = np.linspace(0, 1, 48+1)   #没有音符，用NaN表示
# 鼓轨平均力度，same as FIGARO
# DEFAULT_DRUMS_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32+1)   #没有音符，用NaN表示


# 其他轨的音符密度, #没有音符，用NaN表示
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 2.5, 64+1)          
# 其他轨的平均音高，same as FIGARO, #没有音符，用NaN表示
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32+1)           
# 其他轨的平均时值，#没有音符，用NaN表示
DEFAULT_MEAN_DURATION_BINS = np.sort(np.concatenate([
  np.arange(1, 13), # smallest possible units up to 1 quarter
  np.arange(12, 24, 3)[1:], # 16th notes up to 0.5 bar
  np.arange(12, 24, 4)[1:], # triplets up to 0.5 bar
  np.arange(24, 48, 6), # 8th notes up to 1 bars
  np.arange(48, 3*48, 12), # quarter notes up to 3 bars
]))   

# 其他轨的平均力度，same as FIGARO, #没有音符，用NaN表示
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32+1)         

DEFAULT_RESOLUTION = 12  #已调整为12

# 拍号最多为4/4拍 单小节最多为4拍
MAX_BAR_LENGTH = 4

# 歌曲最大小节数，mmt representation, description feature
MAX_N_BARS = 64






