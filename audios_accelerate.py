from utils import accelerate_audio, get_audio_info, clear_folder, write_json, read_json
from tqdm import tqdm
import os
import shutil
from math import ceil, floor
import argparse
from config.config import target_person, target_model
import yaml

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

parser = argparse.ArgumentParser()
parser.add_argument('-i', default='audio/blankRemove', type=str)
parser.add_argument('-o', default='audio/accelerate', type=str)
parser.add_argument('-a', default=data['ref'][target_model]['ratio'], type=float)
parser.add_argument('-n', default=f'{target_person}', type=str, choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager', 'namvet'])
args = parser.parse_args()

def accelerate_audios(input_folder: str, output_folder: str, person: str, max_ratio: float):
    """
    计算当前音频长度与目标音频长度的比值，(若比值未超过最大阈值则)根据比值加速音频，
    超过阈值的台词写入json，交由人工处理
    """
    need_process = {}
    unfinished_count = 0
    total_accelerate_ratio = 0
    accelerate_count = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)
    names_len_sr = read_json(f'jsons/{person}_names_len_sr.json')
    names_contents = read_json(f'jsons/{person}.json')
    pbar = tqdm(os.listdir(input_folder), desc=f'加速!!!!!(ratio:{max_ratio})')
    for index, name in enumerate(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, name)
        output_path = os.path.join(output_folder, name)
        tar_length = names_len_sr[name]['length']
        _, _, _, _, now_length = get_audio_info(input_path)
        # 如果现在的长度小于目标长度， 直接导出即可
        if int(now_length) <= int(tar_length):
            shutil.copy(input_path, output_path)
        else:
            # 计算当前长度/目标长度。向上取整，取2位小数，结果即为需要加速的比例
            accelerate_ratio = ceil((now_length / tar_length) * 100) / 100
            total_accelerate_ratio += accelerate_ratio
            accelerate_count += 1
            pbar.set_postfix_str(f'期望加速比例{(total_accelerate_ratio/accelerate_count):.3f}')
            # 如果需要加速的比例 <= 最大加速比例，加速后导出
            if accelerate_ratio <= max_ratio:
                accelerate_audio(input_path, output_path, ratio=accelerate_ratio)
            # 如果[(目标长度*0.025)->四舍五入] >= [(当前长度*0.025)->加速->向下取整]，可以勉强通过
            elif round(tar_length * 0.025, 0) >= floor(now_length * 0.025 / max_ratio):
                accelerate_audio(input_path, output_path, ratio=max_ratio)
            # 否则交由后续再处理
            else:
                unfinished_count += 1
                if_close = 'close' if (now_length * 0.025 / max_ratio) - tar_length * 0.025 <= 3 else 'far'
                need_process[name] = {'jp': names_contents[name]['jp'],
                                      'zh': names_contents[name]['zh'],
                                      'tar_length': tar_length * 0.025,
                                      'now_length': now_length * 0.025 / max_ratio,
                                      'emo': names_contents[name]['emo'],
                                      'if_close': if_close}
        pbar.update()
    pbar.close()
    print(f'剩余{unfinished_count}个音频不满足要求')

    write_json(f'unfinished/{person}.json', need_process)
    return f'加速任务完成，共有{len(os.listdir(input_folder))}个音频，期望加速比例：{(total_accelerate_ratio/accelerate_count):.3f}\n' \
           f'还剩余{unfinished_count}个音频不满足要求，不满足音频的信息已储存在unfinished/{person}.json中'

if __name__ == '__main__':
    accelerate_audios(input_folder=args.i,
                      output_folder=args.o,
                      person=args.n,
                      max_ratio=args.a)
