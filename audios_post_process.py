import os
import time
from tqdm import tqdm
from pydub import AudioSegment
from pydub.effects import normalize
import pydub.scipy_effects as scipy_effects
from config.config import target_person
from utils import clear_folder, read_json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', default='audio/temp', type=str)
parser.add_argument('--louder', default='audio/louder', type=str)
parser.add_argument('-o', default='audio/finished', type=str)
parser.add_argument('-n', default=f'{target_person}', type=str, choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager', 'namvet'])
args = parser.parse_args()

def adjust_volume(input_path: str, target_dbfs: int = 0, focus_freq=1000) -> AudioSegment:
    """
    修改音频的音量效果
    """
    input_audio: AudioSegment = AudioSegment.from_file(input_path, format="wav")
    gain = target_dbfs - input_audio.max_dBFS + 3
    out = scipy_effects.eq(input_audio, focus_freq=focus_freq, gain_dB=gain, filter_mode="high_shelf", order=2)
    out = normalize(out, headroom=0.05)
    return out

def adjust_length(path: str, target_length: int, return_seg: bool = False):
    """
    调整音频长度(覆盖输入文件)
    """
    target_audio = AudioSegment.from_file(path)
    length = len(target_audio)
    if length == target_length:
        pass
    elif length > target_length:
        target_audio = target_audio[:target_length]
    else:
        silence = AudioSegment.silent(duration=target_length - length + 2)
        target_audio += silence
    if not return_seg:
        target_audio.export(path, format="wav")
    else:
        return target_audio

def adjust_sr(input_path: str, output_path: str, target_sr: int, return_seg: bool = False):
    audio: AudioSegment = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr)
    if not return_seg:
        audio.export(output_path, format='wav')
    else:
        return audio

def change_audios_volume(input_folder: str, output_folder: str, focus_center: int = 150):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)
    for name in tqdm(os.listdir(input_folder), desc='正在修改音量'):
        in_path = os.path.join(input_folder, name)
        out_path = os.path.join(output_folder, name)
        result_audio = adjust_volume(input_path=in_path,
                                     target_dbfs=0,
                                     focus_freq=focus_center)
        result_audio.export(out_path, format='wav')

def change_audios_length(input_folder: str, person: str):
    names_len_sr = read_json(f'jsons/{person}_names_len_sr.json')
    for name in tqdm(os.listdir(input_folder), desc='正在修改音频长度'):
        in_path = os.path.join(input_folder, name)
        adjust_length(in_path, names_len_sr[name]['length'])

def change_audios_sample_rate(input_folder: str, output_folder: str, person: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)
    names_len_sr = read_json(f'jsons/{person}_names_len_sr.json')
    for name in tqdm(os.listdir(input_folder), desc='正在重采样'):
        in_path = os.path.join(input_folder, name)
        out_path = os.path.join(output_folder, name)
        adjust_sr(in_path, out_path, names_len_sr[name]['sr'])
    return f'共有{len(os.listdir(input_folder))}个音频，后处理已完成。'

if __name__ == '__main__':
    change_audios_volume(args.i, args.louder)
    time.sleep(1)
    change_audios_length(args.louder, args.n)
    time.sleep(1)
    change_audios_sample_rate(args.louder, args.o, args.n)
