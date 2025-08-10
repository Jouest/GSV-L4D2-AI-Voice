import os
from utils import get_audio_info, write_json

# 获取NPC音频的采样率与时长

main = r'E:\SteamLibrary\steamapps\common\Left 4 Dead 2\left4dead2\sound\npc'
dlc2 = r'E:\SteamLibrary\steamapps\common\Left 4 Dead 2\left4dead2_dlc2\sound\npc'
dlc3 = r'E:\SteamLibrary\steamapps\common\Left 4 Dead 2\left4dead2_dlc3\sound\npc'

npc_abs = {}  # 格式：npc名称:含该名称的绝对路径
"""
{
npc:['abs1', 'abs2']
}
"""
for dlc in [main, dlc2, dlc3]:
    npc_names = os.listdir(dlc)
    for npc_name in npc_names:
        if npc_name in ['headcrab', 'infected', 'lilpeanut', 'mega_mob',
                        'moustachio', 'pilot', 'soldier1', 'soldier2', 'virgil',
                        'whitaker', 'witch'] or (not os.path.isdir(os.path.join(dlc, npc_name))):
            continue
        if npc_name in npc_abs.keys():
            npc_abs[npc_name].append(os.path.join(dlc, npc_name))
        else:
            npc_abs[npc_name] = [os.path.join(dlc, npc_name)]


for npc in npc_abs.keys():
    abs_paths = npc_abs[npc]
    audio2info = {}  # for every npc
    for abs_path in abs_paths:
        audio_names = os.listdir(abs_path)
        for audio_name in audio_names:
            if audio_name in audio2info.keys():  # 已经存在
                continue
            audio_path = os.path.join(abs_path, audio_name)
            _, sr, _, _, duration = get_audio_info(audio_path)
            audio2info[audio_name] = {'length': duration, 'sr': sr}
    write_json(f'{npc}_names_len_sr.json', audio2info)
