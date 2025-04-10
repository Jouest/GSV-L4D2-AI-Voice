# Json覆盖
import argparse
from utils import read_json, write_json
from config.config import target_person
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n', default=f'{target_person}', type=str, choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager'])
args = parser.parse_args()

def cover_json(person: str):
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
    return '覆盖操作已完成'

if __name__ == '__main__':
    print(cover_json(person=args.n))
