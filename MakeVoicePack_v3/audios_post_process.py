import os
from tqdm import tqdm
from utils import clear_folder, read_json, \
    audio_phone_like_process, adjust_sr,\
    adjust_length, batch_process_audio, normalize_volume

def batch_make_phone_like(in_folder: str, out_folder: str):
    count = batch_process_audio(
        in_folder,
        out_folder,
        audio_phone_like_process,
        desc='制造电话中...',
        high_center=600
    )
    return f'已经完成对{count}个音频的操作'

def batch_fade(in_folder: str, out_folder: str):
    count = batch_process_audio(
        in_folder,
        out_folder,
        add_cosine_fade,
        desc='fade',
    )
    return f'已经完成对{count}个音频的操作'

def batch_change_volume(
    input_folder: str,
    output_folder: str,
    target_dBFS: float = -10.0,
    pre_gain_db: float = 2.0
):
    count = batch_process_audio(
        input_folder,
        output_folder,
        normalize_volume,
        desc='音量标准化中...',
        pre_gain_db=pre_gain_db,
        target_dBFS=target_dBFS
    )
    return f'已经完成对{count}个音频的操作'

def batch_change_length(input_folder: str, person: str):
    names_len_sr = read_json(f'jsons/{person}_names_len_sr.json')
    for name in tqdm(os.listdir(input_folder), desc=f'正在修改音频长度...人物:{person}'):
        in_path = os.path.join(input_folder, name)
        adjust_length(in_path, names_len_sr[name]['length'])

def batch_change_sr(input_folder: str, output_folder: str, person: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)
    names_len_sr = read_json(f'jsons/{person}_names_len_sr.json')
    for name in tqdm(os.listdir(input_folder), desc=f'正在重采样...人物:{person}'):
        in_path = os.path.join(input_folder, name)
        out_path = os.path.join(output_folder, name)
        adjust_sr(in_path, out_path, names_len_sr[name]['sr'])
    return f'共有{len(os.listdir(input_folder))}个音频，后处理已完成。'
