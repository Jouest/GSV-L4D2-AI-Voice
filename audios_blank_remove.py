import os
from utils import clear_folder, remove_audio_blank
from tqdm import tqdm
import argparse
import yaml
from config.config import target_model

parser = argparse.ArgumentParser()
parser.add_argument('-i', default='audio/output', type=str)
parser.add_argument('-o', default='audio/blankRemove', type=str)
parser.add_argument('-r', default=f'{target_model}', type=str, help='目标音色模型名称')
args = parser.parse_args()

def remove_audios_blank(input_folder: str, output_folder: str, threshold: int = -40, max_silence: int = 100):
    """
    去除音频首尾的静音段，以及中间部分超过最大静音时长的静音段
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)
    for name in tqdm(os.listdir(input_folder), desc='消除首尾静音中...终端若长时间没有变化，是因为网页被置于后台而无法进行下一个任务，将网页置于前台即可恢复'):
        audio_path = os.path.join(input_folder, name)
        output_path = os.path.join(output_folder, name)
        remove_audio_blank(audio_path, output_path, silence_thresh=threshold, max_silence=max_silence)
    return f'去除音频空白任务完成,共对个{len(os.listdir(input_folder))}音频进行了处理'

if __name__ == '__main__':
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    threshold = data['ref'][args.r]['threshold']
    max_silence = data['ref'][args.r]['max_silence']
    remove_audios_blank(args.i, args.o, threshold, max_silence)
