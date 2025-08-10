import os
from tqdm import tqdm
from utils import read_json, write_json, get_audio_info

# 把在audio/accelerate文件夹里的音频写入unfinished/person_accelerated.json
def check_accelerated(person: str, ratio):
    accelerated = {}
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    names_contents = read_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'))
    names_len_sr = read_json(os.path.join(main_dir_path, 'jsons', f'{person}_names_len_sr.json'))
    accelerated_audios = os.listdir(os.path.join(main_dir_path, 'audio', 'accelerate'))

    for name in tqdm(accelerated_audios, desc='检查中'):
        tar_length = names_len_sr[name]['length']
        _, _, _, _, now_length = get_audio_info(os.path.join(main_dir_path, 'audio', 'accelerate', name))
        accelerated[name] = {'jp': names_contents[name]['jp'],
                             'zh': names_contents[name]['zh'],
                             'tar_length': tar_length * 0.025,
                             'now_length': now_length * 0.025 / float(ratio),
                             'emo': names_contents[name]['emo']}

    write_json(os.path.join(main_dir_path, 'unfinished', f'{person}_accelerated.json'), accelerated)

    return f'accelerate文件夹内有{len(accelerated)}个音频,音频信息已经储存在unfinished/{person}_accelerated.json中'

# 使用unfinished/person_accelerated.json中的信息覆盖jsons/person.json
def accelerated2jsons(person: str):
    main_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    names_contents = read_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'))
    accelerated = read_json(os.path.join(main_dir_path, 'unfinished', f'{person}_accelerated.json'))
    for name in accelerated.keys():
        names_contents[name] = {
                'jp': accelerated[name]['jp'],
                'zh': accelerated[name]['zh'],
                'emo': accelerated[name]['emo']
            }

    write_json(os.path.join(main_dir_path, 'jsons', f'{person}.json'), names_contents)
    return f'unfinished/{person}_accelerated.json--覆盖-->jsons/{person}.json, 操作已完成'
