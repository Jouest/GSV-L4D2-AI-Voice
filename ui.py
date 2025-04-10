import re
import time
import gradio as gr
from pydub import AudioSegment
import numpy as np
from utils import write_json, read_json, clear_folder, get_audio_info, get_audio_db
import shutil
import os
import socket
from gsv_v3 import produce_v3
from gsv_v2 import produce_v2
from gradio_client import Client, handle_file
from audios_blank_remove import remove_audios_blank
from audios_accelerate import accelerate_audios
from audios_post_process import change_audios_volume, change_audios_length, change_audios_sample_rate
from check.check_audios_left import check_left
from check.check_json import cover_json
from check.check_audios import check_audios_info
import tqdm
import ast

model_setting_folder = 'model_setting'  # 存放已添加模型信息的文件夹
gradio_port = 6657

gsv_webpage = 'http://localhost:9872/'  # GPT-SoVITS推理界面的网页号
generate_sign = True  # 生成音频时的标识符


def is_port_open(port: int, ip_address: str = '127.0.0.1'):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex((ip_address, port))
        # 如果返回值为 0，表示端口是开放的
        return result == 0
    except socket.error as e:
        print(f"发生异常：{e}")
        return False
    finally:
        sock.close()


def auto_choose_model(choose_model: str, model_kind: str, version: str):
    if choose_model is None:
        return None
    setting_dir = os.path.join(model_setting_folder, version, choose_model)
    json_dict = read_json(f'{setting_dir}/{choose_model}.json')
    model = json_dict[model_kind]
    return model


def change_single_audio_info(audio_path: str, tar_sr: str, tar_len: str):
    if audio_path is None:
        raise gr.Error(message='未上传音频')
    if not audio_path.split('\\')[-1].endswith('.wav'):
        raise gr.Error(message='上传的音频不是波形音频(wav)')
    if tar_sr == '' and tar_len == '':
        raise gr.Error(message='未输入任何待修改数据的值')
    if tar_sr != '' and not bool(re.match(r'^[1-9]\d*$', tar_sr)):
        raise gr.Error(message='输入的目标采样率非法')
    if tar_len != '' and not bool(re.match(r'^[1-9]\d*$', tar_len)):
        raise gr.Error(message='输入的目标长度非法')
    if tar_sr != '' and not 8000 < int(tar_sr) < 192000:
        raise gr.Error(message='采样率过低或过高')
    audio = AudioSegment.from_file(audio_path)
    cur_sr = int(audio.frame_rate)
    cur_len = len(audio)
    tar_sr = cur_sr if tar_sr == '' else int(tar_sr)
    tar_len = cur_len if tar_len == '' else int(tar_len)
    # 重采样
    audio_modified = audio.set_frame_rate(tar_sr)
    # 调整长度
    cur_len = len(audio_modified)
    if cur_len == tar_len:
        pass
    elif cur_len > tar_len:
        audio_modified = audio_modified[:tar_len]
    else:
        silence = AudioSegment.silent(duration=tar_len - cur_len + 2)
        audio_modified += silence
    samples = np.array(audio_modified.get_array_of_samples())
    if audio_modified.channels == 1:
        samples = samples.reshape(-1, 1)
    else:
        samples = samples.reshape(-1, audio_modified.channels)
    return audio_modified.frame_rate, samples

def get_wav_info(audio_dir: str, audio_path: str):
    if audio_dir == '' and audio_path is None:
        raise gr.Error('文件夹路径和音频均未上传')
    info = []
    # 采样率、位深、比特率、声道数, 音量、音频长度
    if audio_dir != '':
        if not os.path.isdir(audio_dir):
            raise gr.Error('文件夹路径无效')
        # 使用文件夹路径
        wav_files = [wav_file for wav_file in os.listdir(audio_dir) if wav_file.endswith('.wav')]
        for wav_file in tqdm.tqdm(wav_files, desc='查询中...'):
            n_channel, sample_rate, bit_depth, bit_rate, duration = get_audio_info(os.path.join(audio_dir, wav_file))
            volume = get_audio_db(os.path.join(audio_dir, wav_file))
            info.append([wav_file, sample_rate, bit_depth, bit_rate, n_channel, volume, duration])
        return info
    else:
        if (not os.path.isfile(audio_path)) or (not audio_path.endswith('.wav')):
            raise gr.Error('音频路径无效，或者当前音频不是波形音频')
        # 使用音频路径
        n_channel, sample_rate, bit_depth, bit_rate, duration = get_audio_info(audio_path)
        volume = get_audio_db(audio_path)
        return [audio_path.split('\\')[-1], sample_rate, bit_depth, bit_rate, n_channel, volume, duration]


def write_info_gotten_in_txt(text: str):
    if text.strip() == '':
        raise gr.Error('文本框为空，无法写入')
    str_to_list = ast.literal_eval(text)
    formatted_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    with open(f'log/wav-info-{formatted_time}.txt', 'w', encoding='utf-8') as file:
        for sub_list in str_to_list:
            if type(sub_list) == list:
                for value in sub_list:
                    file.write(f'{value}\t')
                file.write('\n')
            else:
                file.write(f'{sub_list}\t')
        file.write('\n')
    return f'成功写入log/wav-info-{formatted_time}.txt'


def generate_audios(version, person: str, model: str,
                    finished_dir: str, output_dir: str,
                    language: str, web_port: str,
                    emotion_list: list):
    global generate_sign
    generate_sign = True
    if not emotion_list:
        raise gr.Error(message='你没有选择任何情感种类')
    emotion_list = [s[0] for s in emotion_list]

    if model is None:
        raise gr.Error(message='你没有选择任何音色模型')

    if version is None:
        raise gr.Error(message='版本号异常，请刷新页面重试')

    setting_dir = os.path.join(model_setting_folder, version, model)

    model_json = read_json(f'{setting_dir}/{model}.json')
    pth = model_json['pth']
    ckpt = model_json['ckpt']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        clear_folder(output_dir)

    if os.path.isdir(finished_dir):
        finished_names = os.listdir(finished_dir)
    else:
        raise gr.Error(message='已完成音频的文件夹的路径异常')

    if person not in ['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager', 'namvet']:
        raise gr.Error(message='游戏角色的名称异常')
    names_contents = read_json(f'jsons/{person}.json')

    if language not in ['日文', '中文']:
        raise gr.Error(message='语言选择异常')

    if web_port == '':
        raise gr.Error('端口号为空')
    if not is_port_open(port=int(''.join(re.findall(r'[1234567890]', web_port)))):
        raise gr.Error('端口号错误，或者当前尚未开启GSV推理网页')

    client = Client(web_port)
    if version == 'v3':
        client.predict(sovits_path=pth,
                       prompt_language=language, text_language=language,
                       api_name="/change_sovits_weights")
        client.predict(gpt_path=ckpt,
                       api_name="/change_gpt_weights")
    elif version == 'v2':
        client.predict(sovits_path=pth,
                       prompt_language=language, text_language=language,
                       api_name="/change_sovits_weights")
        client.predict(weights_path=ckpt,
                       api_name="/init_t2s_weights")
    else:
        raise gr.Error('版本号错误')

    pbar = tqdm.trange(len(names_contents.keys()) - len(finished_names), desc=f'版本:{version},情感:{emotion_list}')
    prompt_lang = model_json['lang']
    text_lang = language
    for name, value in names_contents.items():
        if not generate_sign:
            return '生成进程被人为停止'
        if name in finished_names:
            continue
        emotion = value['emo']
        if emotion not in emotion_list:
            pbar.update()
            continue
        text = value['jp'] if language == '日文' else value['zh']
        ref_text = model_json[emotion]['text']
        ref_audio_path = os.path.join(setting_dir, f'{emotion}.wav')

        if version == 'v3':
            if_use_sr = model_json[emotion]['if_sr']
            produce_v3(client=client,
                       output_dir=output_dir,
                       audio_name=name,
                       text=text,
                       ref_audio_path=ref_audio_path,
                       ref_text=ref_text,
                       if_sr=if_use_sr,
                       prompt_lang=prompt_lang,
                       text_lang=text_lang)
        elif version == 'v2':
            aux_ref_audio_paths = []
            if model_json[emotion]['if_aux']:
                # 说明有辅助参考音频
                for file in os.listdir(setting_dir):
                    if file.startswith(emotion) and file.endswith('aux.wav'):
                        aux_ref_audio_paths.append(os.path.join(setting_dir, file))
            produce_v2(client=client,
                       output_folder=output_dir,
                       audio_name=name,
                       text=text,
                       ref_audio_path=ref_audio_path,
                       ref_text=ref_text,
                       aux_ref_audio_paths=aux_ref_audio_paths,
                       prompt_lang=prompt_lang,
                       text_lang=text_lang
                       )
        pbar.set_postfix_str(f'最新已生成音频:{name}')
        pbar.update()
    pbar.close()
    generate_sign = False
    return '音频已经生成完成'


def stop_generate_audios():
    global generate_sign
    if not generate_sign:
        raise gr.Warning(message='生成进程已经处于停止状态')
    else:
        generate_sign = False


def load_model_setting(version: str):
    """读取外部文件获取三个列表"""
    setting_dirs = [item for item in os.listdir(f'{model_setting_folder}/{version}')
                    if os.path.isdir(os.path.join(model_setting_folder, version, item))]
    tones, pths, ckpts = [], [], []
    for model in setting_dirs:
        model_json = read_json(os.path.join(model_setting_folder, version, model, f'{model}.json'))
        pth = model_json['pth']
        ckpt = model_json['ckpt']
        tones.append(model)
        pths.append(pth)
        ckpts.append(ckpt)
    return [tones, pths, ckpts]


def refresh_model_path(version):
    """
    在生成音频界面的刷新模型按钮
    """
    if version is None:
        return [
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[]),
            gr.Dropdown(choices=[])
        ]
    new_lists = load_model_setting(version)
    return [
        gr.Dropdown(choices=new_lists[0], value=new_lists[0][0] if new_lists[0] else None),
        gr.Dropdown(choices=new_lists[1], value=new_lists[1][0] if new_lists[1] else None),
        gr.Dropdown(choices=new_lists[2], value=new_lists[2][0] if new_lists[2] else None)
    ]


def return_self(item):
    return item


def return_str(item):
    return str(item)


def check_audios(name: str, finished_dir: str):
    if name is None:
        raise gr.Error(message='未选择目标人物')
    if finished_dir == '':
        raise gr.Error(message='音频文件夹为空')
    if not os.path.isdir(finished_dir):
        raise gr.Error(message='音频文件夹路径不存在')
    return check_audios_info(person=name)


def v2_model_write_in(name: str, language: str, pth_folder: str, pth: str, ckpt_folder: str, ckpt: str,
                      w_main_audio_path: str, w_main_audio_text: str, w_aux_audio_path: str,
                      d_main_audio_path: str, d_main_audio_text: str, d_aux_audio_path: str,
                      q_main_audio_path: str, q_main_audio_text: str, q_aux_audio_path: str,
                      h_main_audio_path: str, h_main_audio_text: str, h_aux_audio_path: str,
                      a_main_audio_path: str, a_main_audio_text: str, a_aux_audio_path: str):
    if name == '':
        raise gr.Error(message='模型名称为空')
    if language is None:
        raise gr.Error('语言选择出现错误')

    name = name.strip()
    setting_dir = os.path.join(model_setting_folder, 'v2', name)

    if os.path.exists(setting_dir):
        raise gr.Error(message='当前模型已存在同名模型，请更换')

    if pth_folder == '':
        raise gr.Error(message='pth文件夹为空')
    if ckpt_folder == '':
        raise gr.Error(message='ckpt文件夹为空')
    if pth is None:
        raise gr.Error(message='pth模型未选择')
    if ckpt is None:
        raise gr.Error(message='ckpt模型未选择')

    pth = os.path.join(pth_folder, pth)
    ckpt = os.path.join(ckpt_folder, ckpt)

    if not os.path.exists(pth):
        raise gr.Error(message='GPT/pth文件路径不存在')
    if not os.path.exists(ckpt):
        raise gr.Error(message='SoVITS/ckpt文件路径不存在')

    if w_main_audio_path is None or w_main_audio_text == '':
        raise gr.Error(message='警告语气w的主参考音频或参考文本未上传')
    if d_main_audio_path is None or d_main_audio_text == '':
        raise gr.Error(message='低沉语气d的主参考音频或参考文本未上传')
    if q_main_audio_path is None or q_main_audio_text == '':
        raise gr.Error(message='疑问语气q的主参考音频或参考文本未上传')
    if h_main_audio_path is None or h_main_audio_text == '':
        raise gr.Error(message='快乐语气h的主参考音频或参考文本未上传')
    if a_main_audio_path is None or a_main_audio_text == '':
        raise gr.Error(message='生气语气a的主参考音频或参考文本未上传')

    for audio_path in [w_main_audio_path, d_main_audio_path, q_main_audio_path, h_main_audio_path, a_main_audio_path]:
        if not os.path.exists(audio_path):
            audio_path = audio_path.split('\\')[-1]
            raise gr.Error(message=f'主参考音频{audio_path}不存在或过期，请重新上传')

    for audio_path in [w_aux_audio_path, d_aux_audio_path, q_aux_audio_path, h_aux_audio_path, a_aux_audio_path]:
        if audio_path is not None and (not os.path.exists(audio_path)):
            audio_path = audio_path.split('\\')[-1]
            raise gr.Error(message=f'副参考音频{audio_path}不存在或过期，请重新上传')

    os.makedirs(setting_dir)
    shutil.copy(w_main_audio_path, f'{setting_dir}/w.wav')
    shutil.copy(d_main_audio_path, f'{setting_dir}/d.wav')
    shutil.copy(q_main_audio_path, f'{setting_dir}/q.wav')
    shutil.copy(h_main_audio_path, f'{setting_dir}/h.wav')
    shutil.copy(a_main_audio_path, f'{setting_dir}/a.wav')

    model_dict = {
        'lang': language,
        'pth': pth.strip(),
        'ckpt': ckpt.strip(),
        'w': {'text': w_main_audio_text.strip(),
              'if_aux': False},
        'd': {'text': d_main_audio_text.strip(),
              'if_aux': False},
        'q': {'text': q_main_audio_text.strip(),
              'if_aux': False},
        'h': {'text': h_main_audio_text.strip(),
              'if_aux': False},
        'a': {'text': a_main_audio_text.strip(),
              'if_aux': False}}

    if w_aux_audio_path is not None:
        model_dict['w']['if_aux'] = True
        shutil.copy(w_aux_audio_path, f'{setting_dir}/w_aux.wav')
    if d_aux_audio_path is not None:
        model_dict['d']['if_aux'] = True
        shutil.copy(d_aux_audio_path, f'{setting_dir}/d_aux.wav')
    if q_aux_audio_path is not None:
        model_dict['q']['if_aux'] = True
        shutil.copy(q_aux_audio_path, f'{setting_dir}/q_aux.wav')
    if h_aux_audio_path is not None:
        model_dict['h']['if_aux'] = True
        shutil.copy(h_aux_audio_path, f'{setting_dir}/h_aux.wav')
    if a_aux_audio_path is not None:
        model_dict['a']['if_aux'] = True
        shutil.copy(a_aux_audio_path, f'{setting_dir}/a_aux.wav')

    write_json(f'{setting_dir}/{name}.json', model_dict)
    return f'v2模型{name}提交完成'


def v3_model_write_in(name: str, language: str, pth_folder: str, pth: str, ckpt_folder: str, ckpt: str,
                      w_path: str, w_text: str, w_if_sr: bool,
                      d_path: str, d_text: str, d_if_sr: bool,
                      q_path: str, q_text: str, q_if_sr: bool,
                      h_path: str, h_text: str, h_if_sr: bool,
                      a_path: str, a_text: str, a_if_sr: bool):
    if name == '':
        raise gr.Error(message='模型名称为空')
    if language is None:
        raise gr.Error('语言选择出现错误')

    name = name.strip()
    setting_dir = os.path.join(model_setting_folder, 'v3', name)

    if os.path.exists(setting_dir):
        raise gr.Error(message='当前模型已存在同名模型，请更换')

    if pth_folder == '':
        raise gr.Error(message='pth文件夹为空')
    if ckpt_folder == '':
        raise gr.Error(message='ckpt文件夹为空')
    if pth is None:
        raise gr.Error(message='pth模型未选择')
    if ckpt is None:
        raise gr.Error(message='ckpt模型未选择')

    pth = os.path.join(pth_folder, pth)
    ckpt = os.path.join(ckpt_folder, ckpt)

    if not os.path.exists(pth):
        raise gr.Error(message='GPT/pth文件路径不存在')
    if not os.path.exists(ckpt):
        raise gr.Error(message='SoVITS/ckpt文件路径不存在')

    if w_path is None or w_text == '':
        raise gr.Error(message='警告语气w的主参考音频或参考文本未上传')
    if d_path is None or d_text == '':
        raise gr.Error(message='低沉语气d的主参考音频或参考文本未上传')
    if q_path is None or q_text == '':
        raise gr.Error(message='疑问语气q的主参考音频或参考文本未上传')
    if h_path is None or h_text == '':
        raise gr.Error(message='快乐语气h的主参考音频或参考文本未上传')
    if a_path is None or a_text == '':
        raise gr.Error(message='生气语气a的主参考音频或参考文本未上传')

    for audio_path in [w_path, d_path, q_path, h_path, a_path]:
        if not os.path.exists(audio_path):
            audio_path = audio_path.split('\\')[-1]
            raise gr.Error(message=f'音频{audio_path}不存在或过期，请重新上传')

    os.makedirs(setting_dir)
    shutil.copy(w_path, f'{setting_dir}/w.wav')
    shutil.copy(d_path, f'{setting_dir}/d.wav')
    shutil.copy(q_path, f'{setting_dir}/q.wav')
    shutil.copy(h_path, f'{setting_dir}/h.wav')
    shutil.copy(a_path, f'{setting_dir}/a.wav')

    model_dict = {
        'lang': language,
        'pth': pth,
        'ckpt': ckpt,
        'w': {'text': w_text,
              'if_sr': w_if_sr},
        'd': {'text': d_text,
              'if_sr': d_if_sr},
        'q': {'text': q_text,
              'if_sr': q_if_sr},
        'h': {'text': h_text.strip(),
              'if_sr': h_if_sr},
        'a': {'text': a_text.strip(),
              'if_sr': a_if_sr}}

    write_json(f'{setting_dir}/{name}.json', model_dict)
    return f'v3模型{name}提交完成'


def refresh_model_from_path(path: str, postfix: str):
    """
    输入模型文件夹的地址时，下拉框自动刷新可选的模型
    """
    choices = []
    postfix = '.' + postfix
    if not os.path.isdir(path):
        return gr.Dropdown(choices=choices)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        if file.endswith(postfix):
            choices.append(file)
    return gr.Dropdown(choices=choices)


with gr.Blocks() as demo:
    demo.title = 'L4D2 Voice Package Production Tool'
    gr.Markdown('# 语音生成整合包')
    with gr.Tabs():
        with gr.TabItem(label='添加v2模型'):
            gr.Markdown('## 将你的v2模型添加进整合包中')
            with gr.Column():
                with gr.Row(equal_height=True):
                    with gr.Column():
                        v2_model_name = gr.Textbox(label='添加v2模型的名称(注意是v2不是v3)', value='xxx',
                                                   interactive=True,
                                                   show_copy_button=True)
                        v2_model_lang = gr.Dropdown(label='v2模型的语言', value='日文', choices=['中文', '日文'], interactive=True)
                    with gr.Column(scale=2):
                        v2_pth_model_folder = gr.Textbox(label='存放v2 SoVITS/pth模型的文件夹',
                                                         placeholder='存放v2 SoVITS/pth模型的文件夹', interactive=True,
                                                         show_copy_button=True)
                        v2_pth_name = gr.Dropdown(label='SoVITS/pth模型', interactive=True)
                    with gr.Column(scale=2):
                        v2_ckpt_model_folder = gr.Textbox(label='存放v2 GPT/ckpt模型的文件夹',
                                                          placeholder='存放v2 GPT/ckpt模型的文件夹', interactive=True,
                                                          show_copy_button=True)
                        v2_ckpt_name = gr.Dropdown(label='GPT/ckpt模型', interactive=True)
                        ckpt_postfix = gr.State(value='ckpt')
                        pth_postfix = gr.State(value='pth')
                    v2_pth_model_folder.change(fn=refresh_model_from_path, inputs=[v2_pth_model_folder, pth_postfix],
                                               outputs=v2_pth_name)
                    v2_ckpt_model_folder.change(fn=refresh_model_from_path, inputs=[v2_ckpt_model_folder, ckpt_postfix],
                                                outputs=v2_ckpt_name)

                with gr.Column():
                    with gr.Row(equal_height=True):
                        w_main_path = gr.Audio(label='警告、命令、生气情感的主参考音频(也是最泛用的音频)',
                                               type='filepath',
                                               sources=['upload'], scale=2)
                        w_aux_path = gr.Audio(label='辅参考音频 (没有的话就不填)', type='filepath',
                                              sources=['upload'])
                        w_main_text = gr.TextArea(label='主参考音频参考文本(辅参考音频不需要参考文本)',
                                                  placeholder='警告、命令、生气情感的主参考文本',
                                                  interactive=True,
                                                  show_copy_button=True, scale=2)
                    with gr.Row(equal_height=True):
                        d_main_path = gr.Audio(label='低沉、沮丧语气的主参考音频',
                                               type='filepath',
                                               sources=['upload'], scale=2)
                        d_aux_path = gr.Audio(label='辅参考音频 (没有的话就不填)', type='filepath',
                                              sources=['upload'])
                        d_main_text = gr.TextArea(label='主参考音频参考文本(辅参考音频不需要参考文本)',
                                                  placeholder='低沉、沮丧语气的主参考文本',
                                                  interactive=True,
                                                  show_copy_button=True, scale=2)
                    with gr.Row(equal_height=True):
                        q_main_path = gr.Audio(label='疑问、询问语气的的主参考音频',
                                               type='filepath',
                                               sources=['upload'], scale=2)
                        q_aux_path = gr.Audio(label='辅参考音频 (没有的话就不填)', type='filepath',
                                              sources=['upload'])
                        q_main_text = gr.TextArea(label='主参考音频参考文本(辅参考音频不需要参考文本)',
                                                  placeholder='疑问、询问语气的的主参考文本',
                                                  interactive=True,
                                                  show_copy_button=True, scale=2)
                    with gr.Row(equal_height=True):
                        h_main_path = gr.Audio(label='快乐情感的主参考音频',
                                               type='filepath',
                                               sources=['upload'], scale=2)
                        h_aux_path = gr.Audio(label='辅参考音频 (没有的话就不填)', type='filepath',
                                              sources=['upload'])
                        h_main_text = gr.TextArea(label='主参考音频参考文本(辅参考音频不需要参考文本)',
                                                  placeholder='快乐情感的主参考文本',
                                                  interactive=True,
                                                  show_copy_button=True, scale=2)
                    with gr.Row(equal_height=True):
                        a_main_path = gr.Audio(label='生气、愤怒语气的主参考音频',
                                               type='filepath',
                                               sources=['upload'], scale=2)
                        a_aux_path = gr.Audio(label='辅参考音频 (没有的话就不填)', type='filepath',
                                              sources=['upload'])
                        a_main_text = gr.TextArea(label='主参考音频参考文本(辅参考音频不需要参考文本)',
                                                  placeholder='生气、愤怒语气的主参考文本',
                                                  interactive=True,
                                                  show_copy_button=True, scale=2)
                with gr.Row(equal_height=True, min_height=200):
                    upload_v2_model_btn = gr.Button(value='提交v2模型', variant='primary')
                    v2_if_upload = gr.TextArea(label='提交信息', interactive=False)
                    upload_v2_model_btn.click(fn=v2_model_write_in,
                                              inputs=[v2_model_name, v2_model_lang,
                                                      v2_pth_model_folder, v2_pth_name,
                                                      v2_ckpt_model_folder, v2_ckpt_name,
                                                      w_main_path, w_main_text, w_aux_path,
                                                      d_main_path, d_main_text, d_aux_path,
                                                      q_main_path, q_main_text, q_aux_path,
                                                      h_main_path, h_main_text, h_aux_path,
                                                      a_main_path, a_main_text, a_aux_path],
                                              outputs=v2_if_upload)

        with gr.TabItem(label='添加v3模型'):
            gr.Markdown('## 将你的v3模型添加进整合包中')
            with gr.Column():
                with gr.Row(equal_height=True):
                    with gr.Column():
                        v3_model_name = gr.Textbox(label='添加v3模型的名称(注意是v3不是v2)', value='xxx',
                                                   interactive=True, show_copy_button=True)
                        v3_model_lang = gr.Dropdown(label='v3模型的语言', value='日文', choices=['中文', '日文'],
                                                    interactive=True)
                    with gr.Column(scale=2):
                        v3_pth_model_folder = gr.Textbox(label='存放v3 SoVITS/pth模型的文件夹',
                                                         placeholder='存放v3 SoVITS/pth模型的文件夹', interactive=True,
                                                         show_copy_button=True)
                        v3_pth_name = gr.Dropdown(label='SoVITS/pth模型', interactive=True)
                    with gr.Column(scale=2):
                        v3_ckpt_model_folder = gr.Textbox(label='存放v3 GPT/ckpt模型的文件夹',
                                                          placeholder='存放v3 GPT/ckpt模型的文件夹', interactive=True,
                                                          show_copy_button=True)
                        v3_ckpt_name = gr.Dropdown(label='GPT/ckpt模型', interactive=True)
                    v3_pth_model_folder.change(fn=refresh_model_from_path, inputs=[v3_pth_model_folder, pth_postfix],
                                               outputs=v3_pth_name)
                    v3_ckpt_model_folder.change(fn=refresh_model_from_path, inputs=[v3_ckpt_model_folder, ckpt_postfix],
                                                outputs=v3_ckpt_name)

                with gr.Column():
                    with gr.Row(equal_height=True):
                        warn_path = gr.Audio(label='警告、命令、生气情感的参考音频(也是最泛用的音频)', type='filepath',
                                             sources=['upload'])
                        with gr.Column(scale=3):
                            warn_text = gr.Textbox(label='参考文本', placeholder='警告、命令、生气情感的参考文本',
                                                   interactive=True,
                                                   show_copy_button=True, lines=5)
                            warn_if_sr = gr.Radio(value=False, label='是否启用音频超分', choices=[True, False],
                                                  interactive=True)
                    with gr.Row(equal_height=True):
                        deep_path = gr.Audio(label='低沉、沮丧语气的参考音频', type='filepath', sources=['upload'])
                        with gr.Column(scale=3):
                            deep_text = gr.TextArea(label='参考文本', placeholder='低沉、沮丧语气的参考文本',
                                                    interactive=True,
                                                    show_copy_button=True, lines=5)
                            deep_if_sr = gr.Radio(value=False, label='是否启用音频超分', choices=[True, False],
                                                  interactive=True)
                    with gr.Row(equal_height=True):
                        question_path = gr.Audio(label='疑问、询问语气的参考音频', type='filepath', sources=['upload'])
                        with gr.Column(scale=3):
                            question_text = gr.TextArea(label='参考文本', placeholder='疑问、询问语气的参考文本',
                                                        interactive=True,

                                                        show_copy_button=True, lines=5)
                            question_if_sr = gr.Radio(value=False, label='是否启用音频超分', choices=[True, False],
                                                      interactive=True)
                    with gr.Row(equal_height=True):
                        happy_path = gr.Audio(label='快乐情感的参考音频', type='filepath', sources=['upload'])
                        with gr.Column(scale=3):
                            happy_text = gr.TextArea(label='参考文本', placeholder='快乐情感的参考文本',
                                                     interactive=True,
                                                     show_copy_button=True, lines=5)
                            happy_if_sr = gr.Radio(value=False, label='是否启用音频超分', choices=[True, False],
                                                   interactive=True)
                    with gr.Row(equal_height=True):
                        angry_path = gr.Audio(label='生气、愤怒语气的参考音频', type='filepath', sources=['upload'])
                        with gr.Column(scale=3):
                            angry_text = gr.TextArea(label='参考文本', placeholder='生气、愤怒语气的参考文本',
                                                     interactive=True,
                                                     show_copy_button=True, lines=5)
                            angry_if_sr = gr.Radio(value=False, label='是否启用音频超分', choices=[True, False],
                                                   interactive=True)
                with gr.Row(equal_height=True, min_height=200):
                    upload_v3_model_btn = gr.Button(value='提交v3模型', variant='primary')
                    v3_if_upload = gr.TextArea(label='提交信息', interactive=False)
                    upload_v3_model_btn.click(fn=v3_model_write_in,
                                              inputs=[v3_model_name, v3_model_lang,
                                                      v3_pth_model_folder, v3_pth_name,
                                                      v3_ckpt_model_folder, v3_ckpt_name,
                                                      warn_path, warn_text, warn_if_sr,
                                                      deep_path, deep_text, deep_if_sr,
                                                      question_path, question_text, question_if_sr,
                                                      happy_path, happy_text, happy_if_sr,
                                                      angry_path, angry_text, angry_if_sr],
                                              outputs=v3_if_upload)

        with gr.TabItem(label='生成音频'):
            with gr.Row(equal_height=True):
                with gr.Column():
                    model_version_to_generate = gr.Radio(choices=['v2', 'v3'], value='v3', interactive=True,
                                                         label='使用模型的版本号', visible=True)
                with gr.Column(scale=4):
                    target_model = gr.Dropdown(label='选择的音色模型', interactive=True)
                with gr.Column(scale=4):
                    target_person = gr.Dropdown(
                        choices=['coach', 'gambler', 'mechanic', 'producer', 'biker', 'teengirl', 'manager', 'namvet'],
                        label='目标游戏人物', interactive=True)
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    target_pth = gr.Dropdown(
                        label='SoVITS/pth模型  (实际上，生成音频不需要你选择这2个模型，它取决于音色模型选项)',
                        interactive=True)
                    target_ckpt = gr.Dropdown(
                        label='GPT/ckpt模型  (这只是为了方便检查是否已经正确添加模型，如果正确，应该会自动添加上去)',
                        interactive=True)
                demo.load(
                    fn=refresh_model_path,
                    inputs=[model_version_to_generate],
                    outputs=[target_model, target_pth, target_ckpt]
                )
                model_version_to_generate.change(fn=refresh_model_path, inputs=[model_version_to_generate],
                                                 outputs=[target_model, target_pth, target_ckpt])
                refresh_info_btn = gr.Button(value='刷新模型路径', variant='primary')
                refresh_info_btn.click(fn=refresh_model_path, inputs=[model_version_to_generate],
                                       outputs=[target_model, target_pth, target_ckpt])
                target_model.change(fn=auto_choose_model, inputs=[target_model, pth_postfix, model_version_to_generate],
                                    outputs=target_pth).then(
                    fn=auto_choose_model, inputs=[target_model, ckpt_postfix, model_version_to_generate],
                    outputs=target_ckpt
                )

            with gr.Column():
                with gr.Row():
                    finished_folder = gr.Textbox(lines=1, placeholder='已完成音频文件夹(audio/temp)',
                                                 label='已完成音频的文件夹 (请自行将已经生成并且合格的音频放入其中，程序将默认不会生成其中的音频)',
                                                 value='audio/temp', interactive=True)
                with gr.Row():
                    output_folder = gr.Textbox(lines=1, placeholder='输出音频文件夹(audio/output)',
                                               label='输出音频的文件夹', value='audio/output', interactive=True)
                with gr.Row():
                    lang = gr.Dropdown(choices=['日文', '中文'], label='求生之路语音包的语言', value='日文', interactive=True)
                with gr.Row():
                    gsv_port = gr.Textbox(value=gsv_webpage,
                                          label='连接TTS WebUI的端口号，如果你没有改动GSV的代码，默认不需要修改',
                                          interactive=True)
                with gr.Row():
                    emotion_want = gr.Checkboxgroup(
                        choices=['warn(警告、命令)', 'deep(低沉)', 'question(疑问)', 'happy(快乐)', 'angry(生气、愤怒)'],
                        value=['warn(警告、命令)', 'deep(低沉)', 'question(疑问)', 'happy(快乐)', 'angry(生气、愤怒)'],
                        interactive=True, label='需要生成的的情感种类的音频')
                gr.Markdown('## 开始生成音频将会覆盖输出文件夹内的文件! 如果输出文件夹内有重要的音频，请提前转移')
                with gr.Row(equal_height=True):
                    start_generate_btn = gr.Button(value='开始生成音频',
                                                   variant='primary', scale=3)
                    stop_generate_btn = gr.Button(value='终止生成(在当前音频生成完成后停止)', variant='primary', scale=2)
                    stop_generate_btn.click(fn=stop_generate_audios)
                    generate_info = gr.TextArea(placeholder='音频生成信息', interactive=False, show_label=False,
                                                scale=3)
                    start_generate_btn.click(generate_audios,
                                             inputs=[model_version_to_generate, target_person, target_model,
                                                     finished_folder, output_folder,
                                                     lang, gsv_port,
                                                     emotion_want],
                                             outputs=generate_info)

        with gr.TabItem(label='移除音频空白、加速') as tab_blank:
            gr.Markdown('## 为生成音频移除首部、尾部、及中间过长的空白，然后加速音频，缩短音频时长')
            person_need_remove_blank = gr.Dropdown(label='目标游戏人物', value=target_person.value,
                                                   choices=['coach', 'gambler', 'mechanic', 'producer', 'biker',
                                                            'teengirl', 'manager', 'namvet'], interactive=True)
            tab_blank.select(fn=return_self, inputs=target_person, outputs=person_need_remove_blank)
            with gr.Column():
                gr.Markdown('#### 如果你的音色模型声线偏弱，静音阈值建议不高于-50dB，如果声音偏强，建议不高于-40dB')
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        need_remove_blank_folder = gr.Textbox(label='需要移除空白的音频文件夹', value='audio/output',
                                                              interactive=True)
                        blank_removed_folder = gr.Textbox(label='移除空白后导出音频所在的文件夹',
                                                          value='audio/blankRemove',
                                                          interactive=True)
                    with gr.Column(scale=3):
                        silence_thresh = gr.Slider(label='静音检测的阈值(dB)', minimum=-100, maximum=0, step=1,
                                                   value=-45,
                                                   interactive=True)
                        max_silence = gr.Slider(label='在音频中部允许的最大静音时长(ms)', minimum=0, maximum=500,
                                                step=10,
                                                value=300, interactive=True)
                    start_remove_blank_btn = gr.Button(value='移除音频空白部分', interactive=True, variant='primary',
                                                       elem_classes="custom-button")
                gr.Markdown(
                    '#### 建议加速比例默认设置1.15，如果提升0.05的比例对音频听感影响不大，则可以继续提升 (最佳加速比例因不同的模型而异)')
                with gr.Row(equal_height=True):
                    accelerate_out_folder = gr.Textbox(label='加速音频后导出音频所在的文件夹', value='audio/accelerate',
                                                       interactive=True, scale=2)
                    accelerate_ratio = gr.Slider(label='最大加速比例(建议1.1~1.3)', minimum=1, maximum=1.8, value=1.15,
                                                 step=0.01, interactive=True, scale=3)
                    start_accelerate_btn = gr.Button(value='为音频进行合适的加速', interactive=True, variant='primary')
                with gr.Row(equal_height=True):
                    remove_blank_accelerate_btn = gr.Button(variant='primary', value='两种任务，一键完成',
                                                            interactive=True)
                    remove_blank_accelerate_text = gr.TextArea(label='处理信息', interactive=False)
                start_remove_blank_btn.click(fn=remove_audios_blank,
                                             inputs=[need_remove_blank_folder, blank_removed_folder,
                                                     silence_thresh, max_silence],
                                             outputs=remove_blank_accelerate_text)
                start_accelerate_btn.click(fn=accelerate_audios,
                                           inputs=[blank_removed_folder, accelerate_out_folder,
                                                   person_need_remove_blank,
                                                   accelerate_ratio],
                                           outputs=remove_blank_accelerate_text)
                remove_blank_accelerate_btn.click(fn=remove_audios_blank,
                                                  inputs=[need_remove_blank_folder, blank_removed_folder,
                                                          silence_thresh, max_silence],
                                                  outputs=remove_blank_accelerate_text).then(fn=accelerate_audios,
                                                                                             inputs=[
                                                                                                 blank_removed_folder,
                                                                                                 accelerate_out_folder,
                                                                                                 person_need_remove_blank,
                                                                                                 accelerate_ratio],
                                                                                             outputs=remove_blank_accelerate_text)

        with gr.TabItem(label='辅助工具') as tab_check:
            gr.Markdown('')
            with gr.Column():
                person_in_check_step = gr.Dropdown(label='目标游戏人物', value=target_person.value,
                                                   choices=['coach', 'gambler', 'mechanic', 'producer', 'biker',
                                                            'teengirl', 'manager', 'namvet'], interactive=True)
                gr.Markdown('## 也许你想知道还有多少音频需要生成')
                with gr.Row(equal_height=True):
                    with gr.Column():
                        finished_folder_in_check = gr.Textbox(label='已完成音频的文件夹', value='audio/temp',
                                                              interactive=True)
                        accelerate_ratio_in_check = gr.Textbox(label='在加速音频的时的加速比例',
                                                               value=accelerate_ratio.value, interactive=True)
                    tab_check.select(fn=return_self, inputs=target_person, outputs=person_in_check_step).then(
                        fn=return_str, inputs=accelerate_ratio, outputs=accelerate_ratio_in_check)

                    check_left_btn = gr.Button(value='检查剩余没有生成的音频', interactive=True, variant='primary')
                    check_left_text = gr.TextArea(label='检查结果', interactive=False, scale=2)
                    check_left_btn.click(fn=check_left, inputs=[person_in_check_step, accelerate_ratio_in_check],
                                         outputs=check_left_text)
                gr.Markdown(
                    '## 也许你在unfinished/xxx.json中修改了内容，想同步到jsons/xxx.json中(因为生成音频的文本依赖于这里)')
                with gr.Row(equal_height=True):
                    cover_json_btn = gr.Button(value='执行覆盖操作 (请谨慎操作)', interactive=True, variant='primary')
                    cover_json_text = gr.TextArea(label='覆盖信息', interactive=False, scale=2)
                    cover_json_btn.click(cover_json, inputs=person_in_check_step, outputs=cover_json_text)
                gr.Markdown('## 查询文件夹内所有或单个波形音频(wav)(文件夹优先)的信息')
                gr.Markdown('## 包括采样率(Hz)、位深(位)、比特率(bps)、声道数, 音量(dB)、音频长度(ms)')
                with gr.Row(equal_height=True):
                    with gr.Column():
                        audio_folder_need_get_info = gr.Textbox(label='音频文件夹(优先)', interactive=True)
                        audio_need_get_info = gr.Audio(label='上传待查询音频', sources=['upload'], type='filepath')
                    get_info_btn = gr.Button(interactive=True, variant='primary', value='查询信息')
                    info_gotten = gr.TextArea(interactive=False,
                                              label='查询结果(名称、采样率、位深、比特率、声道数, 音量、音频长度)',
                                              show_copy_button=True, scale=2, lines=15)
                    with gr.Column():
                        write_in_txt_btn = gr.Button(interactive=True, value='把结果写入文本文档', variant='primary')
                        write_in_txt_text = gr.TextArea(interactive=False, label='结果')
                    get_info_btn.click(fn=get_wav_info, inputs=[audio_folder_need_get_info, audio_need_get_info],
                                       outputs=info_gotten)
                    write_in_txt_btn.click(fn=write_info_gotten_in_txt, inputs=info_gotten, outputs=write_in_txt_text)
                gr.Markdown('## 修改单个波形音频的长度和采样率')
                with gr.Row(equal_height=True):
                    single_audio_to_modify = gr.Audio(sources=['upload'], type='filepath', label='上传待修改音频')
                    with gr.Column():
                        single_audio_tar_sr = gr.Textbox(label='目标采样率(Hz)不填视为保持原始数据', interactive=True,
                                                         placeholder='hz')
                        single_audio_tar_len = gr.Textbox(label='目标长度(ms)不填视为保持原始数据', interactive=True,
                                                          placeholder='ms')
                    modify_single_audio_btn = gr.Button(value='修改', interactive=True, variant='primary')
                with gr.Row(equal_height=True):
                    single_audio_modified = gr.Audio(label='音频修改结果')
                    modify_single_audio_btn.click(fn=change_single_audio_info,
                                                  inputs=[single_audio_to_modify, single_audio_tar_sr,
                                                          single_audio_tar_len], outputs=single_audio_modified)

        with gr.TabItem(label='音频后处理') as tab_process:
            gr.Markdown('## 如果你确定完成了全部的音频，并且储存进了audio/temp文件夹中\n'
                        '## 那么恭喜你，可以开始最后的音量增益、长度修正以及重采样了')
            with gr.Column():
                person_need_post_process = gr.Dropdown(label='目标游戏人物', value=target_person.value,
                                                       choices=['coach', 'gambler', 'mechanic', 'producer', 'biker',
                                                                'teengirl', 'manager', 'namvet'], interactive=True)
                tab_process.select(fn=return_self, inputs=target_person, outputs=person_need_post_process)
                need_process_folder = gr.Textbox(label='待处理音频文件夹', value='audio/temp', interactive=True)
                audio_gain_process_out = gr.Textbox(label='音量增益后的输出文件夹', value='audio/louder', visible=False)
                finial_process_out = gr.Textbox(label='输出为最终成品的文件夹', value='audio/finished',
                                                interactive=True)
                with gr.Row(equal_height=True):
                    start_process_btn = gr.Button(value='后处理', interactive=True, variant='primary')
                    start_process_text = gr.TextArea(label='后处理信息', interactive=False, scale=2)
                    start_process_btn.click(fn=change_audios_volume,
                                            inputs=[need_process_folder,
                                                    audio_gain_process_out],
                                            outputs=start_process_text).then(
                        fn=change_audios_length,
                        inputs=[audio_gain_process_out,
                                person_need_post_process],
                        outputs=start_process_text).then(
                        fn=change_audios_sample_rate,
                        inputs=[audio_gain_process_out,
                                finial_process_out,
                                person_need_post_process],
                        outputs=start_process_text)
                with gr.Column():
                    gr.Markdown('## 检查最终处理后的音频是否合格(其实并没有必要,不过这取决于你)')
                    audio_folder_need_check = gr.Textbox(label='已完成音频的文件夹', value='audio/finished',
                                                         interactive=True)
                    finial_process_out.change(fn=return_self, inputs=finial_process_out,
                                              outputs=audio_folder_need_check)
                    with gr.Row(equal_height=True):
                        check_audios_btn = gr.Button(value='检查', variant='primary', interactive=True)
                        check_audios_text = gr.TextArea(label='检查结果', interactive=False, show_copy_button=True,
                                                        scale=2)
                        check_audios_btn.click(fn=check_audios_info, inputs=person_need_post_process,
                                               outputs=check_audios_text)

demo.launch(server_port=gradio_port, show_error=True, inbrowser=True, server_name='localhost')
