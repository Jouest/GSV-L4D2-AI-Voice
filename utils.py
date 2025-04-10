# 工具
import re
import shutil
from pydub.effects import normalize
import pydub.scipy_effects as scipy_effects
import subprocess
from pydub import AudioSegment, silence
import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm
import json

def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def write_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_audio_db(file_path: str) -> float:
    sample_rate, data = wavfile.read(file_path)
    if len(data) == 0:
        return -50  # 空白音频

    # 计算均方根振幅
    rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
    # 计算分贝值
    if rms == 0:
        return -np.inf  # 静音
    db = 20 * np.log10(rms / 32767)
    return float(db)

def accelerate_audio(input_path: str, output_path: str, ratio: float):
    if input_path.endswith(".wav"):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", input_path,
            "-filter:a", f"atempo={ratio}",
            "-vn",
            output_path
        ]
        subprocess.run(cmd, check=True)

def remove_audio_blank(input_path, output_path, silence_thresh: int = -45, max_silence: int = 300):
    audio = AudioSegment.from_wav(input_path)
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=50,
        silence_thresh=silence_thresh
    )
    # 全静音文件不处理
    total_duration = len(audio)
    silent_duration = sum([end - start for start, end in silent_ranges])
    if silent_duration >= total_duration * 0.99:
        shutil.copy(input_path, output_path)
        return

    # 首尾静音修剪
    start_trim = 0
    end_trim = len(audio)
    if silent_ranges:
        if silent_ranges[0][0] == 0:
            start_trim = silent_ranges[0][1]
        if silent_ranges[-1][1] == len(audio):
            end_trim = silent_ranges[-1][0]

    # 提取有效段落
    trimmed = audio[start_trim:end_trim]

    # 中间静音处理
    segments = []
    prev_end = 0
    for start, end in silence.detect_silence(trimmed, silence_thresh=silence_thresh):
        # 添加非静音段
        segments.append(trimmed[prev_end:start])

        # 处理静音段
        silence_duration = end - start
        if silence_duration > max_silence:
            segments.append(AudioSegment.silent(duration=max_silence))
        else:
            segments.append(trimmed[start:end])

        prev_end = end

    # 添加最后一段
    segments.append(trimmed[prev_end:])

    # 合并并导出
    processed = sum(segments)
    processed.export(output_path, format="wav")

def get_audio_info(file_path: str):
    """
    查看信息
    :param file_path: 输入地址
    :return: 通道数、采样率(Hz)、位深、比特率(bps)、时长(ms)
    """
    audio = AudioSegment.from_file(file_path)
    # 获取通道数
    n_channels = int(audio.channels)
    # 获取采样率
    sample_rate = int(audio.frame_rate)
    # 获取位深（例如16位）
    bit_depth = int(audio.sample_width * 8)
    # 获取比特率
    bitrate = int(audio.frame_rate * bit_depth * n_channels)
    # 获取音频长度（毫秒）
    duration_ms = int(len(audio))

    return n_channels, sample_rate, bit_depth, bitrate, duration_ms

def clear_folder(folder_path: str):
    """
    清空文件夹内的全部文件
    :param folder_path: 文件夹地址
    :return: None
    """
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return

    for filename in tqdm(os.listdir(folder_path), desc=f'正在清空文件夹:{folder_path}'):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")

def audio_phone_like_process(input_audio_path, output_audio_path, low_center=1600, high_center=300) -> AudioSegment:
    wav = AudioSegment.from_file(input_audio_path)
    # audio = scipy_effects.low_pass_filter(wav, low_center)
    audio = scipy_effects.high_pass_filter(wav, high_center)
    audio = normalize(audio)
    audio.export(output_audio_path, format="wav")
    return audio

def subtitle_to_content(text: str):
    if '_' in text:
        name = re.findall(r'_([^"]+)"', text)[0]
    else:
        name = re.findall(r'.([^"]+)"', text)[0]
    if text.split('"')[-2].strip() == '':
        content = ''
    else:
        if '：' in text:
            content = re.findall(r'：(.*?)"', text)[-1]
        else:
            content = re.findall(r':([^:]*)"', text)[-1]
        if '>' in content:
            content = content.split('>')[-1]
    return name.lower() + '.wav', content.strip()
