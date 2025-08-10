
# 把不在temp文件夹里的音频写入unfinished/person.json
import os
from tqdm import tqdm
from utils import read_json, write_json, get_audio_info

def check_left(person: str, rate):
    unfinished = {}
    rate = float(rate)
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    names_contents = read_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'))
    names_len_sr = read_json(os.path.join(main_dir_path, 'jsons', f'{person}_names_len_sr.json'))
    blank_removed_audios = os.listdir(os.path.join(main_dir_path, 'audio', 'blankRemove'))

    max_rate = rate
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
                                'now_length': now_length * 0.025 / max_rate,
                                'emo': names_contents[name]['emo']}

    write_json(os.path.join(main_dir_path, 'unfinished', f'{person}.json'), unfinished)

    return f'还有{len(unfinished)}个音频需要生成,结果已经储存在unfinished/{person}.json中'

# 将unfinished/person.json的信息覆盖jsons/person.json
def unfinished2jsons(person: str):
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    names_contents = read_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'))
    unfinished = read_json(os.path.join(main_dir_path, 'unfinished', f'{person}.json'))
    for name in names_contents.keys():
        if name in unfinished.keys():
            names_contents[name] = {
                'jp': unfinished[name]['jp'],
                'zh': unfinished[name]['zh'],
                'emo': unfinished[name]['emo']
            }

    write_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'), names_contents)
    return f'unfinished/{person}.json--覆盖-->jsons/{person}.json, 操作已完成'
