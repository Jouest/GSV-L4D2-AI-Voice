# 把不在temp文件夹里的音频写入json
import os
from tqdm import tqdm
import argparse
from utils import read_json, write_json, get_audio_info
from config.config import target_person, target_model
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=f'{target_person}', type=str,
                    choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager'])
parser.add_argument('-r', default=f'{target_model}', type=str)
args = parser.parse_args()

def check_left(person: str, ratio):
    unfinished = {}
    ratio = float(ratio)
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    names_contents = read_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'))
    names_len_sr = read_json(os.path.join(main_dir_path, 'jsons', f'{person}_names_len_sr.json'))
    blank_removed_audios = os.listdir(os.path.join(main_dir_path, 'audio', 'blankRemove'))

    max_ratio = ratio
    for name in tqdm(names_contents.keys(), desc='检查中'):
        if name not in os.listdir(os.path.join(main_dir_path, 'audio', 'temp')):
            tar_length = names_len_sr[name]['length']
            if name in blank_removed_audios:
                _, _, _, _, now_length = get_audio_info(os.path.join(main_dir_path, 'audio', 'blankRemove', name))
            else:
                now_length = 0
            unfinished[name] = {'jp': names_contents[name]['jp'],
                                'zh': names_contents[name]['zh'],
                                'tar_length': tar_length * 0.025,
                                'now_length': now_length * 0.025 / max_ratio,
                                'emo': names_contents[name]['emo']}

    write_json(os.path.join(main_dir_path, 'unfinished', f'{person}.json'), unfinished)

    return f'还有{len(unfinished)}个音频需要生成,结果已经储存在unfinished/{person}.json中'

if __name__ == '__main__':
    with open('../config/config.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    out = check_left(person=args.n, ratio=data['ref'][args.r]['ratio'])
    print(out)
