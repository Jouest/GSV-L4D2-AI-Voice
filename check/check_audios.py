# 检测是否有音频超时或者采样率不对
import os
import time

from config.config import target_person
from utils import get_audio_info, read_json, write_json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=f'{target_person}', type=str,
                    choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager'])
args = parser.parse_args()

def check_audios_info(person: str):
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    finished_dir = os.path.join(main_dir_path, 'audio', 'finished')
    finished_names = os.listdir(finished_dir)
    names_len_sr = read_json(os.path.join(main_dir_path, 'jsons', f'{person}_names_len_sr.json'))
    length_mismatch = {}
    sr_mismatch = {}
    error = 20  # 误差值
    for name in tqdm(finished_names, desc='正在检查音频长度和采样率'):
        channel, sample, width, _, length = get_audio_info(os.path.join(finished_dir, name))
        accurate_sr = names_len_sr[name]['sr']
        accurate_len = names_len_sr[name]['length']
        if length > accurate_len + error or length < accurate_len - error:
            length_mismatch[name] = {'tar_len': accurate_len,
                                     'cur_len': length}
        if sample != accurate_sr or channel != 1 or width != 16:
            sr_mismatch[name] = {'tar': [1, 16, accurate_sr],
                                 'cur': [channel, width, sample]}
    if not sr_mismatch and not length_mismatch:
        return f'{finished_dir}内现存的音频符合标准'
    else:
        formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        write_json(f'{main_dir_path}/log/len-unqualified-{formatted_time}.json', length_mismatch)
        write_json(f'{main_dir_path}/log/sr-unqualified-{formatted_time}.json', sr_mismatch)
        return '存在不合规的音频，不合规音频的信息已保存在log文件夹下'


if __name__ == '__main__':
    print(check_audios_info(person=args.n))
