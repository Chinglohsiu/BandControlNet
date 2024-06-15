
from chorder import Dechorder
from chorder import Chord

import muspy
import pretty_midi
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct

import numpy as np
from collections import defaultdict
import random
import copy
from tqdm import tqdm
import pickle

from constants import (
  EOS_TOKEN,
  PAD_TOKEN,
  BOS_TOKEN,
  EOS_TOKEN,

  DEFAULT_POS_PER_QUARTER,    # 12
  DEFAULT_VELOCITY_BINS,
  DEFAULT_DURATION_BINS,
  DEFAULT_TEMPO_BINS,
  
  DEFAULT_DRUMS_PITCH_TYPE_BINS,
  DEFAULT_DRUMS_NOTE_DENSITY_BINS,

  DEFAULT_NOTE_DENSITY_BINS,
  DEFAULT_MEAN_VELOCITY_BINS,
  DEFAULT_MEAN_PITCH_BINS,
  DEFAULT_MEAN_DURATION_BINS,

  DEFAULT_RESOLUTION,        # 12
  MAX_TIME_SHIFT,
  
  Ins_LIST,    # instrument_name
  Ins_ID,      # instrument_program_id
)

class Item(object):
  def __init__(self, name, start, end, velocity=None, pitch=None, instrument=None):
    self.name = name
    self.start = start
    self.end = end
    self.velocity = velocity
    self.pitch = pitch
    self.instrument = instrument

  def __repr__(self):
    return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'.format(
      self.name, self.start, self.end, self.velocity, self.pitch, self.instrument)

class Event(object):
  def __init__(self, name, time, value, text):
    self.name = name
    self.time = time
    self.value = value
    self.text = text

  def __repr__(self):
    return 'Event(name={}, time={}, value={}, text={})'.format(
      self.name, self.time, self.value, self.text)


# 构造固定小节数的乐句
def get_fixed_bar_num_phrase_info(bar_nums, fix_phrase_bar_num, resolution):
    fixed_phrase_num = bar_nums // fix_phrase_bar_num
    new_bar_nums = fixed_phrase_num * fix_phrase_bar_num
    phrase_info = []
    for i in range(fixed_phrase_num):
        phrase_info_dict = {
            'phrase': 'F-{}'.format(fix_phrase_bar_num),
            'from': i * fix_phrase_bar_num * 4*resolution,
            'to': (i+1) * fix_phrase_bar_num * 4*resolution,
            'name': 'F',
            'duration': fix_phrase_bar_num * 4*resolution,
            }
        phrase_info.append(phrase_info_dict)
    return phrase_info, new_bar_nums
    

# file(music_info) -> single sequence  get_remi_raw_events()
# single sequence -> multiple sequence  get_remi_track_events
class REMI_Plus_Raw():

    # remi+ 
    def __init__(self, music_info, chord_enable=False, velocity_enable=True, fix_phrase_bar_num=None):
        # music_info为 final_segment_info_all中的value 
        # self.music_info = pickle.load(open(music_info_path, 'rb'))
        self.music_info = music_info
        
        self.resolution = self.music_info['RESOLUTION']
        self.bar_nums = self.music_info['BAR_NUM']
        
        self.note_items = None
        self.tempo_items = None
        self.chords_items = None
        self.phrase_items = None
        
        self.groups = None

        self.chord_enable = chord_enable
        self.velocity_enable = velocity_enable
        self.fix_phrase_bar_num = fix_phrase_bar_num
    
        self._read_items()
        self._group_items()
    
    def _read_items(self):
        # tempo
        
        # note
        self.note_items = []
        note_info = self.music_info['NOTES_INFO']
        for ins in note_info.keys():
            ins_id = Ins_ID[Ins_LIST.index(ins)]
            for note in note_info[ins]:
                self.note_items.append(Item(
                    name='Note',
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity if self.velocity_enable else 64,
                    pitch=note.pitch,
                    instrument=ins_id,
                    ))
        self.note_items.sort(key=lambda x:(x.start, x.pitch))
        
        # phrase
        self.phrase_items = []
        phrase_info = self.music_info['PHRASE_INFO']
        if self.fix_phrase_bar_num is not None:
            # 构造以fix_phrase_bar_num为固定乐句数的phrase_info
            phrase_info, new_bar_num = get_fixed_bar_num_phrase_info(self.bar_nums, 
                                                                     self.fix_phrase_bar_num,
                                                                     self.resolution,
                                                                     )
            self.bar_nums = new_bar_num
            
        for phrase in phrase_info:
            self.phrase_items.append(Item(
                name='Phrase',
                start=phrase['from'],
                end=phrase['to'],
                velocity=None,
                pitch='Lower' if phrase['name'].islower() else 'Upper',
                ))
        
        
        # chord
        if self.chord_enable:
            self.chord_items = []
            chord_info = self.music_info['CHORDS_INFO']
            for chord in chord_info:
                self.chord_items.append(Item(
                    name='Chord',
                    start=chord['start_beats'],
                    end=chord['end_beats'],
                    velocity=None,
                    pitch=chord['name'],
                    ))
        
        
    def _group_items(self):
        if self.chord_enable:
            items = self.chord_items + self.note_items + self.phrase_items
        else:
            items = self.note_items + self.phrase_items
            
        def _get_key(item):
            type_priority = {
                'Phrase': 0,  
                'Chord': 1,   
                'Note': 2,
                }
            
            return (item.start,
                    type_priority[item.name],
                    item.instrument,
                    item.pitch)
        
        items.sort(key=_get_key)
        
        barlines = [b * self.resolution * 4 for b in range(self.bar_nums+1)]
        self.groups = []
        
        for bar_start_tick, bar_end_tick in zip(barlines[:-1], barlines[1:]):
            inside_bar_items = []
            for item in items:
                if (item.start >= bar_start_tick) and (item.start < bar_end_tick):
                    inside_bar_items.append(item)
            overall = [bar_start_tick] + inside_bar_items + [bar_end_tick]
            self.groups.append(overall)
            
        for idx in [0, -1]:
            while len(self.groups) > 0:
                group = self.groups[idx]
                notes = [item for item in group[1:-1] if item.name == 'Note' or item.name == 'Phrase']
                if len(notes) == 0:
                    self.groups.pop(idx)
                else:
                    break
        
        return self.groups
    
    def tick_to_position(self, tick):
        return round(tick / self.resolution * DEFAULT_POS_PER_QUARTER)
    
    def get_remi_raw_events(self):
        n_downbeat=0
        
        positions_per_bar = DEFAULT_POS_PER_QUARTER * 4
        
        group_events = []
        
        for i in range(len(self.groups)):
            events = []
            current_chord = None
            
            if len(self.groups[i][1:-1]) > 0:
                if self.groups[i][1].name == 'Phrase':
                    current_phrase = self.groups[i][1]
                    bar_num = (current_phrase.end - current_phrase.start) // (self.resolution*4)
            
            bar_st, bar_et = self.groups[i][0], self.groups[i][-1]
            bar_empty_flag = len([item for item in self.groups[i][1:-1] if item.name == 'Note']) > 0
            n_downbeat += 1
            
            # bar类型
            # 字典中只有两类 Bar_Normal, Bar_Empty
            events.append(Event(
                name='Bar',
                time=None,
                value='Normal' if bar_empty_flag else 'Empty',
                text='{}'.format(n_downbeat),
                ))
            
            if current_phrase is not None:
                events.append(Event(
                    name='Phrase',
                    time=current_phrase.start,
                    value=current_phrase.pitch,
                    text='{}'.format(current_phrase.pitch),
                    ))
                events.append(Event(
                    name='BCD',
                    time=current_phrase.start,
                    value=bar_num,
                    text='{}'.format(bar_num)
                    ))
                
                bar_num -= 1
                
                assert bar_num >= 0
            
            # chord类型
            if current_chord is not None:
                events.append(Event(
                    name='Position',
                    time=0,
                    value='{}'.format(0),
                    text='{}/{}'.format(1, positions_per_bar),
                    ))
                events.append(Event(
                    name='Chord',
                    time=current_chord.start,
                    value=current_chord.pitch,
                    text='{}'.format(current_chord.pitch),
                    ))
            
            ticks_per_bar = self.resolution * 4   # 48
            flags = np.linspace(bar_st, bar_st + ticks_per_bar, positions_per_bar, endpoint=False)
            
            for item in self.groups[i][1:-1]:
                # position
                index = np.argmin(abs(flags-item.start))
                pos_event = Event(
                    name='Position',
                    time=item.start,
                    value='{}'.format(index),
                    text='{}/{}'.format(index+1, positions_per_bar),
                    )
                
                    
                if item.name == 'Note':
                    # position - ins - pitch - duration - velocity
                    
                    # position
                    events.append(pos_event)
                    
                    # instrument
                    if item.instrument == -1:
                        name = 'Drums'
                    else:
                        name = pretty_midi.program_to_instrument_name(item.instrument)
                    events.append(Event(
                        name='Instrument',
                        time=item.start,
                        value=name,
                        text='{}'.format(name),
                        ))
                    
                    # pitch
                    events.append(Event(
                        name='Pitch',
                        time=item.start,
                        value='drum_{}'.format(item.pitch) if name == 'Drums' else '{}'.format(item.pitch),
                        text='{}'.format(pretty_midi.note_number_to_drum_name(item.pitch)) if name == 'Drums' else '{}'.format(pretty_midi.note_number_to_name(item.pitch)),
                        ))
                    
                    # duration
                    duration = self.tick_to_position(item.end - item.start)
                    duration_index = 0 if name == 'Drums' else np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                    events.append(Event(
                        name='Duration',
                        time=item.start,
                        value=duration_index,
                        text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[duration_index]),
                        ))
                    
                    # velocity
                    if self.velocity_enable:
                        velocity_index = 16 if name == 'Drums' else np.argmin(abs(DEFAULT_VELOCITY_BINS - item.velocity))
                        events.append(Event(
                            name='Velocity',
                            time=item.start,
                            value=velocity_index,
                            text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index]),
                            ))
                            
                elif item.name == 'Chord':
                    if current_chord is None or item.pitch != current_chord.pitch:
                        events.append(pos_event)
                        events.append(Event(
                            name='Chord',
                            time=item.start,
                            value=item.pitch,
                            text='{}'.format(item.pitch),
                            ))
                        current_chord = item
                        
            each_group_events = [f'{e.name}_{e.value}' for e in events]
            group_events.append(each_group_events)
        return group_events

    def get_remi_track_events(self, group_events_raw):

        # [80, -1, 0, 24, 32, 48]
        # [Melody, Drums, Piano, Guitar, Bass, Strings]
        #  [[Ins_Melody, Bar_XX, ...,]
        #  [Ins_Drums, Bar_XX, ..., ]
        #  [Ins_Piano, Bar_XX, ..., ]
        #  [Ins_Guitar, Bar_XX, ..., ]
        #  [Ins_Bass, Bar_XX, ..., ]
        #  [Ins_Strings, Bar_XX, ..., ]]
        
        # skip chord events if chord events exist

        track_events = []
        for ins in Ins_LIST:
            track_events.append(['Instrument_{}'.format(ins)])

        for bar_id, bar_events in enumerate(group_events_raw):
            # phrase_flag = len([events for events in bar_events if 'Phrase_' in events]) > 0
        
            for ins_id, ins in enumerate(Ins_LIST):
                ins_pos_index = [pos_index for pos_index, events in enumerate(bar_events) if ins in events]
                
                if len(ins_pos_index) > 0:
                    track_events[ins_id].append('Bar_Normal')
                    
                    track_events[ins_id].append(bar_events[1])
                    track_events[ins_id].append(bar_events[2])
                    
                    for pos_index in ins_pos_index:
                        track_events[ins_id].append(bar_events[pos_index-1])
                        track_events[ins_id].append(bar_events[pos_index+1])
                        # 去除drum的duration表示(四十八分音符), velocity表示(固定为64)
                        if ins != 'Drums':
                            if self.velocity_enable:
                                track_events[ins_id].append(bar_events[pos_index+2])
                                track_events[ins_id].append(bar_events[pos_index+3])
                            else:
                                track_events[ins_id].append(bar_events[pos_index+2])
        
                
                else:
                    track_events[ins_id].append('Bar_Empty')
                    
                    track_events[ins_id].append(bar_events[1])
                    track_events[ins_id].append(bar_events[2])
        
        return track_events
                
    def merge_same_position(self, ori_events):    
        bar_index = [i for i, e in enumerate(ori_events) if 'Bar_' in e] + [len(ori_events)]
        merged_events = ori_events[:bar_index[0]]
        
        for bar_st, bar_et in zip(bar_index[:-1], bar_index[1:]):
            bar_events = ori_events[bar_st:bar_et]
            bar_new_events = []
            current_position = None
            
            for event in bar_events:
                if 'Position_' not in event:
                    bar_new_events.append(event)
                else:
                    if event != current_position:
                        bar_new_events.append(event)
                        current_position = event
            merged_events += bar_new_events
            
        return merged_events
        
    def recover_position_raw(self, merged_events):
        # 只对single_sequence
        pos_index = [i for i, e in enumerate(merged_events) if 'Position_' in e] + [len(merged_events)]
        recover_events = merged_events[:pos_index[0]]
        
        for pos_st, pos_et in zip(pos_index[:-1], pos_index[1:]):
            pos_events = merged_events[pos_st:pos_et]
            pos_new_events = []
            for event in pos_events:
                if 'Position_' in event:
                    current_position = event
                if 'Instrument_' not in event:
                    pos_new_events.append(event)
                else:
                    if pos_new_events[-1] != current_position:
                        pos_new_events.append(current_position)
                        pos_new_events.append(event)
                    else:
                        pos_new_events.append(event)
            recover_events += pos_new_events
        
        return recover_events
    
    def recover_position_track(self, merged_events_per_track):
        # 对multiple_sequence的单个轨道而言
        pos_index = [i for i, e in enumerate(merged_events_per_track) if 'Position_' in e] + [len(merged_events_per_track)]
        recover_events_per_track = merged_events_per_track[:pos_index[0]]
        
        for pos_st, pos_et in zip(pos_index[:-1], pos_index[1:]):
            pos_events = merged_events_per_track[pos_st:pos_et]
            pos_new_events = []
            for event in pos_events:
                if 'Position_' in event:
                    current_position = event
                # 与single_sequence相比，只是判断条件从Instrument_变成了Pitch_
                if 'Pitch_' not in event:
                    pos_new_events.append(event)
                else:
                    if pos_new_events[-1] != current_position:
                        pos_new_events.append(current_position)
                        pos_new_events.append(event)
                    else:
                        pos_new_events.append(event)
                        
            recover_events_per_track += pos_new_events
        return recover_events_per_track
    
    def get_sequence_without_phrase(self, ori_sequence):
        sequence = []
        for event in ori_sequence:
            if event.startswith('Phrase'):
                continue
            sequence.append(event)
        return sequence
    
    def get_final_sequence(self, tracks_mode=False):
        # 获取最终的（压缩后的）序列表示
        group_events_raw = self.get_remi_raw_events()
        
    
        if tracks_mode:
            track_events = self.get_remi_track_events(group_events_raw)
            track_events = [self.merge_same_position(track) for track in track_events]
            return track_events
        else:
            raw_events = []
            for events in group_events_raw:
                raw_events += events
            raw_events = self.merge_same_position(raw_events)
            return raw_events
    
    
    # ================================================
    # ================================================
    
    def get_description_raw(self, reorganized_groups):
        assert self.chord_enable == True
        
        positions_per_bar = DEFAULT_POS_PER_QUARTER * 4
        
        n_downbeat = 0
        
        raw_description = []
        raw_description.append([Event(
            name='Features',
            time=None,
            value='Seq',
            text=None,
            )])
        raw_description.append([Event(
            name='Chords',
            time=None,
            value='Seq',
            text=None,
            )])
        
        current_chord = None
        
        for i in range(len(reorganized_groups)):
            n_downbeat += 1
            
            group_st = reorganized_groups[i][0]
            group_ed = reorganized_groups[i][-1]
            num_bar = (group_ed - group_st) // positions_per_bar
            
            cut_groups = [item for item in reorganized_groups[i][1:-1] if item.name == 'Note']
            
            notes_ins_group = self.group_items_by_ins(cut_groups, tracks_mode=False)
            
            cut_chords = [item for item in reorganized_groups[i][1:-1] if item.name == 'Chord']
            
            if len(cut_groups) > 0:
                raw_description[0].append(Event(
                    name='Bar',
                    time=None,
                    value='Normal',
                    text='{}'.format(n_downbeat),
                    ))
            else:
                raw_description[0].append(Event(
                    name='Bar',
                    time=None,
                    value='Empty',
                    text='{}'.format(n_downbeat),
                    ))
            
            drums_notes = notes_ins_group['Drums']
            drums_pitch = [item.pitch for item in drums_notes]
            # drums_velocity = [item.velocity for item in drums_notes]
            
            # 鼓轨平均音符种类
            drums_pitch_type = len(list(set(drums_pitch))) 
            index = np.argmin(abs(DEFAULT_DRUMS_PITCH_TYPE_BINS-drums_pitch_type))
            raw_description[0].append(Event(
                name='Drums_Pitch_Type',
                time=None,
                value=index if len(drums_notes)>0 else 'NaN',
                text=None,
                ))
            
            # 鼓轨平均音符密度
            drums_note_density = len(drums_notes)/(positions_per_bar*num_bar)
            index = np.argmin(abs(DEFAULT_DRUMS_NOTE_DENSITY_BINS-drums_note_density))
            raw_description[0].append(Event(
                name='Drums_Note_Density',
                time=None,
                value=index if len(drums_notes)>0 else 'NaN',
                text=None,
                ))

            normal_notes = notes_ins_group['Normal']
            normal_pitch = [item.pitch for item in normal_notes]
            normal_duration = [item.end-item.start for item in normal_notes]
            normal_velocity = [item.velocity for item in normal_notes]
            
            # 其他轨的音符密度
            normal_note_density = len(normal_notes)/(positions_per_bar*num_bar)
            index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS-normal_note_density))
            raw_description[0].append(Event(
                name='Note_Density',
                time=None,
                value=index if len(normal_notes)>0 else 'NaN',
                text=None,
                ))
            
            # 其他轨的平均音高
            normal_mean_pitch = sum(normal_pitch)/len(normal_pitch) if len(normal_notes)>0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS-normal_mean_pitch))
            raw_description[0].append(Event(
                name='Mean_Pitch',
                time=None,
                value=index if len(normal_notes)>0 else 'NaN',
                text=None,
                ))
            
            # 其他轨的平均时值
            normal_mean_duration = sum(normal_duration)/len(normal_duration) if len(normal_notes)>0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_DURATION_BINS-normal_mean_duration))
            raw_description[0].append(Event(
                name='Mean_Duration',
                time=None,
                value=index if len(normal_notes)>0 else 'NaN',
                text=None,
                ))
            
            # 其他轨的平均力度
            normal_mean_velocity = sum(normal_velocity)/len(normal_velocity) if len(normal_notes)>0 else np.nan
            index = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS-normal_mean_velocity))
            raw_description[0].append(Event(
                name='Mean_Velocity',
                time=None,
                value=index if len(normal_notes)>0 else 'NaN',
                text=None,
                ))
            
            
            raw_description[-1].append(Event(
                name='Bar',
                time=None,
                value='Normal',
                text='{}'.format(n_downbeat),
                ))
            
            # 和弦信息，每个轨都相同
            if len(cut_chords) == 0 and current_chord is not None:
                cut_chords = [current_chord]
            elif len(cut_chords) > 0:
                if cut_chords[0].start > group_st and current_chord is not None:
                    cut_chords.insert(0, current_chord)
                current_chord = cut_chords[-1]
            
            
            # 将chord按beat存入raw_description
            for chord in cut_chords:
                if chord.end > group_ed:
                    temp_chord_ed = group_ed
                else:
                    temp_chord_ed = chord.end
                    
                if chord.start < group_st:
                    temp_chord_st = group_st
                else:
                    temp_chord_st = chord.start
                tmp_factor = (temp_chord_ed - temp_chord_st) // DEFAULT_POS_PER_QUARTER
                for i in range(tmp_factor):
                    raw_description[-1].append(Event(
                                name='Chord',
                                time=None,
                                value=chord.pitch,
                                text='{}'.format(chord.pitch),
                                ))
        
        raw_desc_seq = []
        for events in raw_description:
            raw_desc_seq.append([f'{e.name}_{e.value}' for e in events])
        return raw_desc_seq
        
            
    
    def get_description_track(self, reorganized_groups, tracks_mode=True):
        assert self.chord_enable == True
        assert tracks_mode == True
        
        positions_per_bar = DEFAULT_POS_PER_QUARTER * 4
        
        n_downbeat = 0
        
        track_description = []
        
        current_chord = None
        
        for ins in Ins_LIST:
            track_description.append([Event(
                name='Instrument',
                time=None,
                value=ins,
                text=None,
                )])
        
        # chord序列
        track_description.append([Event(
            name='Chords',
            time=None,
            value='Seq',
            text=None,
            )])
        
        for i in range(len(reorganized_groups)):
            n_downbeat += 1
            
            group_st = reorganized_groups[i][0]
            group_ed = reorganized_groups[i][-1]
            num_bar = (group_ed - group_st) // positions_per_bar
            
            cut_groups = [item for item in reorganized_groups[i][1:-1] if item.name == 'Note']
            
            notes_ins_group = self.group_items_by_ins(cut_groups, tracks_mode)
            
            cut_chords = [item for item in reorganized_groups[i][1:-1] if item.name == 'Chord']
            
            for ins_id, ins in enumerate(Ins_LIST):
                if len(notes_ins_group[ins]) > 0:
                    track_description[ins_id].append(Event(
                        name='Bar',
                        time=None,
                        value='Normal',
                        text='{}'.format(n_downbeat),
                        ))
                else:
                    track_description[ins_id].append(Event(
                        name='Bar',
                        time=None,
                        value='Empty',
                        text='{}'.format(n_downbeat),
                        ))
                    
                
                if ins == 'Drums':
                    drums_notes = notes_ins_group[ins]
                    drums_pitch = [item.pitch for item in drums_notes]
                    # drums_velocity = [item.velocity for item in drums_notes]
                    
                    # 鼓轨平均音符种类
                    drums_pitch_type = len(list(set(drums_pitch))) 
                    index = np.argmin(abs(DEFAULT_DRUMS_PITCH_TYPE_BINS-drums_pitch_type))
                    track_description[ins_id].append(Event(
                        name='Drums_Pitch_Type',
                        time=None,
                        value=index if len(drums_notes)>0 else 'NaN',
                        text=None,
                        ))
                    
                    # 鼓轨平均音符密度
                    drums_note_density = len(drums_notes)/(positions_per_bar*num_bar)
                    index = np.argmin(abs(DEFAULT_DRUMS_NOTE_DENSITY_BINS-drums_note_density))
                    track_description[ins_id].append(Event(
                        name='Drums_Note_Density',
                        time=None,
                        value=index if len(drums_notes)>0 else 'NaN',
                        text=None,
                        ))
                    
                else:
                    normal_notes = notes_ins_group[ins]
                    normal_pitch = [item.pitch for item in normal_notes]
                    normal_duration = [item.end-item.start for item in normal_notes]
                    normal_velocity = [item.velocity for item in normal_notes]
                    
                    # 其他轨的音符密度
                    normal_note_density = len(normal_notes)/(positions_per_bar*num_bar)
                    index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS-normal_note_density))
                    track_description[ins_id].append(Event(
                        name='Note_Density',
                        time=None,
                        value=index if len(normal_notes)>0 else 'NaN',
                        text=None,
                        ))
                    
                    # 其他轨的平均音高
                    normal_mean_pitch = sum(normal_pitch)/len(normal_pitch) if len(normal_notes)>0 else np.nan
                    index = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS-normal_mean_pitch))
                    track_description[ins_id].append(Event(
                        name='Mean_Pitch',
                        time=None,
                        value=index if len(normal_notes)>0 else 'NaN',
                        text=None,
                        ))
                    
                    # 其他轨的平均时值
                    normal_mean_duration = sum(normal_duration)/len(normal_duration) if len(normal_notes)>0 else np.nan
                    index = np.argmin(abs(DEFAULT_MEAN_DURATION_BINS-normal_mean_duration))
                    track_description[ins_id].append(Event(
                        name='Mean_Duration',
                        time=None,
                        value=index if len(normal_notes)>0 else 'NaN',
                        text=None,
                        ))
                    
                    # 其他轨的平均力度
                    normal_mean_velocity = sum(normal_velocity)/len(normal_velocity) if len(normal_notes)>0 else np.nan
                    index = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS-normal_mean_velocity))
                    track_description[ins_id].append(Event(
                        name='Mean_Velocity',
                        time=None,
                        value=index if len(normal_notes)>0 else 'NaN',
                        text=None,
                        ))
            
            track_description[-1].append(Event(
                name='Bar',
                time=None,
                value='Normal',
                text='{}'.format(n_downbeat),
                ))
            
            # 和弦信息，每个轨都相同
            if len(cut_chords) == 0 and current_chord is not None:
                cut_chords = [current_chord]
            elif len(cut_chords) > 0:
                if cut_chords[0].start > group_st and current_chord is not None:
                    cut_chords.insert(0, current_chord)
                current_chord = cut_chords[-1]
            
            # 将chord按beat写入track_description
            for chord in cut_chords:
                if chord.end > group_ed:
                    temp_chord_ed = group_ed
                else:
                    temp_chord_ed = chord.end
                    
                if chord.start < group_st:
                    temp_chord_st = group_st
                else:
                    temp_chord_st = chord.start
                tmp_factor = (temp_chord_ed - temp_chord_st) // DEFAULT_POS_PER_QUARTER
                
                for i in range(tmp_factor):
                    track_description[-1].append(Event(
                                name='Chord',
                                time=None,
                                value=chord.pitch,
                                text='{}'.format(chord.pitch),
                                ))
                        
                        
        # return [f'{e.name}_{e.value}' for e in track_description]
        track_desc_seq = []
        for events in track_description:
            track_desc_seq.append([f'{e.name}_{e.value}' for e in events])
        return track_desc_seq
                
    
    def group_items_by_ins(self, cut_groups, tracks_mode):
        if tracks_mode == True:
            notes_ins_group = {}
            for ins_name in Ins_LIST:
                notes_ins_group[ins_name] = []
        
        
            for item in cut_groups:
                ins_name = 'Drums' if item.instrument == -1 else pretty_midi.program_to_instrument_name(item.instrument)
                notes_ins_group[ins_name].append(item)
        
            return notes_ins_group
        else:
            notes_ins_group = {
                'Normal': [],
                'Drums': [],
                }
            for item in cut_groups:
                if item.instrument == -1:
                    notes_ins_group['Drums'].append(item)
                else:
                    notes_ins_group['Normal'].append(item)
            return notes_ins_group
            
    
    def get_new_groups(self, cover_level_type='bar'):
        positions_per_bar = DEFAULT_POS_PER_QUARTER * 4
        
    
        if cover_level_type == 'bar':
            return self.groups
        else:
            assert cover_level_type == 'phrase'
            groups = []
            phrase_context = []
            for item in self.phrase_items:
                phrase_context.append((
                    item.start//positions_per_bar,
                    item.end//positions_per_bar,
                    ))
            
            for start, end in phrase_context:
                temp_groups = self.groups[start:end]
                temp = []
                for items in temp_groups:
                    temp += items[1:-1]
                temp = [temp_groups[0][0]] + temp + [temp_groups[-1][-1]]
                groups.append(temp)
            return groups
        
        
    # ================================================
    
        
# write midi for single sequence
def remi_raw2midi(raw_events, bpm=120., time_signature=(4,4), polyphony_limit=8, velocity_enable=True):
    midi_obj = mid_parser.MidiFile(ticks_per_beat=DEFAULT_RESOLUTION)
    midi_obj.tempo_changes = [ct.TempoChange(tempo=bpm, time=0)]
    midi_obj.time_signature_changes = [ct.TimeSignature(
                numerator=time_signature[0],
                denominator=time_signature[1],
                time=0,
                )]
    
    if velocity_enable:
        pointer_len = 4
    else:
        pointer_len = 3
        
    instruments = {}
    
    bar = -1
    n_notes = 0
    polyphony_control = {}
    
    for i, event in enumerate(raw_events):
        if event == EOS_TOKEN:
            break
        
        if not bar in polyphony_control:
            polyphony_control[bar]  = {}
        
        if 'Bar_' in raw_events[i]:
            bar += 1
            polyphony_control[bar]  = {}
        
        elif i+pointer_len < len(raw_events) and \
            'Position_' in raw_events[i] and \
            'Instrument_' in raw_events[i+1] and \
            'Pitch_' in raw_events[i+2] and \
            'Duration_' in raw_events[i+3]:
            
            # position
            position = int(raw_events[i].split('_')[-1])
            if not position in polyphony_control[bar]:
                polyphony_control[bar][position] = {}
            
            # instrument
            instrument_name = raw_events[i+1].split('_')[-1]
            if instrument_name not in polyphony_control[bar][position]:
                polyphony_control[bar][position][instrument_name] = 0
            elif polyphony_control[bar][position][instrument_name] > polyphony_limit:
                continue
            
            if instrument_name not in instruments:
                if instrument_name == 'Drums':
                    instrument = ct.Instrument(program=0, is_drum=True, name=instrument_name)
                else:
                    program = pretty_midi.instrument_name_to_program(instrument_name)
                    instrument = ct.Instrument(program=program, is_drum=False, name=instrument_name)
                instruments[instrument_name] = instrument
            
            else:
                instrument = instruments[instrument_name]
            
            # pitch
            pitch = int(raw_events[i+2].split('_')[-1])
            
            # duration
            duration_index = 0 if instrument_name == 'Drums' else int(raw_events[i+3].split('_')[-1])
            duration = DEFAULT_DURATION_BINS[duration_index]
            
            # velocity
            if velocity_enable:
                velocity_index = 16 if instrument_name == 'Drums' else int(raw_events[i+4].split('_')[-1])
                velocity = min(127, DEFAULT_VELOCITY_BINS[velocity_index])
            else:
                velocity = 64
            
            start = bar * DEFAULT_RESOLUTION * 4 + position
            end = start + duration
            
            note = ct.Note(
                pitch=pitch,
                start=start,
                end=end,
                velocity=velocity,
                )
            
            instrument.notes.append(note)
            n_notes += 1
            polyphony_control[bar][position][instrument_name] += 1
    
    instruments = dict(
        sorted(
            instruments.items(),
            key=lambda x: Ins_ID[Ins_LIST.index(x[0])],
            )
        )
    
    for instrument in instruments.values():
        midi_obj.instruments.append(instrument)
    
    return midi_obj

def remi_track2midi(track_events, bpm=120., time_signature=(4,4), polyphony_limit=8, velocity_enable=True):
    midi_obj = mid_parser.MidiFile(ticks_per_beat=DEFAULT_RESOLUTION)
    midi_obj.tempo_changes = [ct.TempoChange(tempo=bpm, time=0)]
    midi_obj.time_signature_changes = [ct.TimeSignature(
                numerator=time_signature[0],
                denominator=time_signature[1],
                time=0,
                )]
    
    if velocity_enable:
        pointer_len = 3
    else:
        pointer_len = 2
    
    
    for track in track_events:
        bar = -1
        n_notes = 0
        polyphony_control = {}
        
        ins_index = [i for i, e in enumerate(track) if 'Instrument_' in e]
        instrument_name = track[ins_index[0]].split('_')[-1]
        # instrument_name = track[0].split('_')[-1]
        if instrument_name == 'Drums':
            instrument = ct.Instrument(program=0, is_drum=True, name=instrument_name)
        else:
            program = pretty_midi.instrument_name_to_program(instrument_name)
            instrument = ct.Instrument(program=program, is_drum=False, name=instrument_name)
        
        
        track = track[ins_index[0]+1:]
        
        if instrument_name == 'Drums':
            # drum轨道固定duration和velocity的值
            duration_index = 0
            velocity_index = 16
            for i, event in enumerate(track):
                if not bar in polyphony_control:
                    polyphony_control[bar] = {}
                if 'Bar_' in track[i]:
                    bar += 1
                    polyphony_control[bar] = {}
                elif i+1 < len(track) and \
                    'Position_' in track[i] and \
                    'Pitch_' in track[i+1]:
                    # position
                    position = int(track[i].split('_')[-1])
                    if not position in polyphony_control[bar]:
                        polyphony_control[bar][position] = 0
                    if polyphony_control[bar][position] > polyphony_limit:
                        continue
                    
                    # pitch
                    pitch = int(track[i+1].split('_')[-1])
                    duration = DEFAULT_DURATION_BINS[duration_index]
                    velocity = DEFAULT_VELOCITY_BINS[velocity_index]
                    
                    start = bar * DEFAULT_RESOLUTION * 4 + position
                    end = start + duration
                    
                    note = ct.Note(
                        pitch=pitch,
                        start=start,
                        end=end,
                        velocity=velocity,
                        )
                    
                    instrument.notes.append(note)
                    n_notes += 1
                    polyphony_control[bar][position] += 1
                    
        else:
            for i, event in enumerate(track):
                if not bar in polyphony_control:
                    polyphony_control[bar] = {}
                if 'Bar_' in track[i]:
                    bar += 1
                    polyphony_control[bar] = {}
                elif i+pointer_len < len(track) and \
                    'Position_' in track[i] and \
                    'Pitch_' in track[i+1] and \
                    'Duration_' in track[i+2]:
                    position = int(track[i].split('_')[-1])
                    if not position in polyphony_control[bar]:
                        polyphony_control[bar][position] = 0
                    if polyphony_control[bar][position] > polyphony_limit:
                        continue
                    
                    # pitch
                    pitch = int(track[i+1].split('_')[-1])
                    # duration
                    duration_index = int(track[i+2].split('_')[-1])
                    duration = DEFAULT_DURATION_BINS[duration_index]
                    # velocity
                    if velocity_enable:
                        velocity_index = int(track[i+3].split('_')[-1])
                        velocity = min(127, DEFAULT_VELOCITY_BINS[velocity_index])
                    else:
                        velocity = 64
                    
                    start = bar * DEFAULT_RESOLUTION * 4 + position
                    end = start + duration
                    
                    note = ct.Note(
                        pitch=pitch,
                        start=start,
                        end=end,
                        velocity=velocity,
                        )
                    
                    instrument.notes.append(note)
                    n_notes += 1
                    polyphony_control[bar][position] += 1
        
        if len(instrument.notes) > 0:
            midi_obj.instruments.append(instrument)
            
    midi_obj.instruments = sorted(midi_obj.instruments, key=lambda x:Ins_ID[Ins_LIST.index(x.name)])
    return midi_obj


        
