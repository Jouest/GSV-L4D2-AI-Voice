from utils import read_json, write_json, read_txt_by_line, subtitle_to_content

main_jp_lines = read_txt_by_line(path='../subtitles/main/subtitles_japanese.txt', utf='utf-16')
main_zh_lines = read_txt_by_line(path='../subtitles/main/subtitles_schinese.txt', utf='utf-16')
dlc2_jp_lines = read_txt_by_line(path='../subtitles/dlc2/subtitles_japanese.txt', utf='utf-16')
dlc2_zh_lines = read_txt_by_line(path='../subtitles/dlc2/closecaption_schinese.txt', utf='utf-16')
update_jp_lines = read_txt_by_line(path='../subtitles/update/subtitles_japanese.txt', utf='utf-16')
update_zh_lines = read_txt_by_line(path='../subtitles/update/subtitles_schinese.txt', utf='utf-16')

# 按前缀筛选特定行，通过正则表达式筛选音频名称与字幕
prefix = '"npc.churchguy_'

jp = {}
zh = {}
for line in main_jp_lines + dlc2_jp_lines + update_jp_lines:
    if line.strip().startswith(prefix):
        name, content = subtitle_to_content(line)
        jp[name] = content

for line in main_zh_lines + dlc2_zh_lines + update_zh_lines:
    if line.strip().startswith(prefix):
        name, content = subtitle_to_content(line)
        zh[name] = content

soldier = {}
soldier_ = read_json('../jsons/churchguy_names_len_sr.json').keys()

for key in jp.keys():
    if key not in soldier_:
        continue

    if key not in zh.keys():
        print(key)

    else:
        soldier[key] = {
            'jp': jp[key],
            'zh': zh[key],
            'emo': 'w'
        }

write_json('../jsons/churchguy.json', soldier)
