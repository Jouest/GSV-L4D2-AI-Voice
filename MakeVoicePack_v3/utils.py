import re
import shutil
from pydub.effects import normalize
from pydub import effects
import pydub.scipy_effects as scipy_effects
import subprocess
from pydub import AudioSegment, silence
import numpy as np
from scipy.io import wavfile
import os
from tqdm import tqdm
import json


def batch_process_audio(
        in_folder: str,
        out_folder: str,
        process_func: callable,
        desc: str = "处理中...",
        **process_kwargs
) -> int:
    """
    批量处理音频文件的公共函数
    参数:
        in_folder: 输入文件夹路径
        out_folder: 输出文件夹路径
        process_func: 单个音频处理函数
        desc: 进度条描述
        **process_kwargs: 传给处理函数的参数

    返回:
        处理的文件数量
    """
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        clear_folder(out_folder)

    filenames = os.listdir(in_folder)
    for filename in tqdm(filenames, desc=desc):
        if filename.lower().endswith('.wav'):
            in_path = os.path.join(in_folder, filename)
            out_path = os.path.join(out_folder, filename)
            if filename == 'blank.wav':
                shutil.copy(in_path, out_path)
                continue
            process_func(in_path, out_path, **process_kwargs)
    return len(filenames)

def normalize_volume(
    src_path: str,
    dst_path: str,
    target_dBFS: float = -10.0,
    pre_gain_db: float = 5.0,
    compress: bool = True,
    comp_threshold: float = -20.0,
    comp_ratio: float = 4.0
):
    """
    1) 先对音频做预增益 (pre_gain_db)
    2) 可选地做动态范围压缩，避免过大过小极端
    3) 根据当前 dBFS 计算增益，使输出达到 target_dBFS
    """
    # 读取
    audio = AudioSegment.from_file(src_path)
    # 预增益
    audio = audio.apply_gain(pre_gain_db)

    # 压缩（阈值 comp_threshold，压缩比 comp_ratio）
    if compress:
        audio = effects.compress_dynamic_range(
            audio,
            threshold=comp_threshold,
            ratio=comp_ratio
        )

    # 计算需要再加/减多少 dB
    change_dB = target_dBFS - audio.dBFS
    audio = audio.apply_gain(change_dB)

    # 导出
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    audio.export(dst_path, format="wav")

def adjust_length(path: str, target_length: int, return_seg: bool = False):
    """
    调整音频长度(覆盖输入文件)
    """
    target_audio = AudioSegment.from_file(path)
    length = len(target_audio)
    if length + 2 == target_length:  # 98 100
        pass
    elif length + 2 > target_length:  # 110 100 -> [0:102] -> 102 100
        target_audio = target_audio[:target_length + 2]
    else:  # 100 110 -> 100 + (110 - 100 + 2) = 100 + 12 = 112 -> 112 110
        silence = AudioSegment.silent(duration=target_length - length + 2)
        target_audio += silence
    if not return_seg:
        target_audio.export(path, format="wav")
    else:
        return target_audio


def adjust_sr(input_path: str, output_path: str, target_sr: int, return_seg: bool = False):
    audio: AudioSegment = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr)
    if not return_seg:
        audio.export(output_path, format='wav')
    else:
        return audio


def read_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(path: str, data: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_txt_by_line(path: str, utf: str = 'utf-8') -> list:
    result = []
    with open(path, 'r', encoding=utf) as f:
        for line in f.readlines():
            result.append(line.strip())
    return result


def get_audio_db(file_path: str) -> float:
    sample_rate, data = wavfile.read(file_path)
    if len(data) == 0:
        return -50  # 空白音频

    rms = np.sqrt(np.mean(data.astype(np.float32) ** 2))
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

def remove_audio_blank(
        input_path: str,
        output_path: str,
        silence_thresh: int = -50,  # dBFS，判断静音的阈值
        min_silence_len: int = 150,  # ms，最小静音长度，用来检测静音段
        max_silence_keep: int = 200  # ms，中间静音片段最多保留的长度
):
    """
    对一个单声道 16-bit WAV：
    1. 去除首尾静音。
    2. 对中间静音区间，如果时长 > max_silence_keep，则只保留 max_silence_keep，剪掉多余部分。
    3. 导出为单声道 16-bit WAV。
    """

    audio = AudioSegment.from_wav(input_path)
    # 确保是单声道，16 位
    audio = audio.set_channels(1).set_sample_width(2)

    # 检测所有静音区间（ms，返回[[start_ms, end_ms], ...]）
    silent_ranges = silence.detect_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    # detect_silence 返回的是在整个音频（0..len(audio)）中的静音区间

    # 整段静音直接原样输出
    if not silent_ranges:
        audio.export(output_path, format="wav", parameters=["-ac", "1", "-sample_fmt", "s16"])
        return

    # 找到首尾非静音位置，然后裁剪掉首尾静音
    start_trim = 0
    end_trim = len(audio)

    # 检查头部静音：如果第一个静音区间从 0 开始，就把它当作头部静音
    first_silence = silent_ranges[0]
    if first_silence[0] == 0:
        start_trim = first_silence[1]

    # 检查尾部静音：如果最后一个静音区间占到结尾
    last_silence = silent_ranges[-1]
    if last_silence[1] == len(audio):
        end_trim = last_silence[0]  # 最后一个静音区间的起点是最后一个非静音的结束

    # 裁剪掉首尾静音：
    trimmed_audio = audio[start_trim:end_trim]

    # 在 trimmed_audio 上重新检测中间静音段
    middle_silences = silence.detect_silence(
        trimmed_audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    # 这里得到的是相对于 trimmed_audio（起点为 0）的区间

    # 如果中间没有静音，直接导出 trimmed_audio 即可
    if not middle_silences:
        trimmed_audio.export(output_path, format="wav", parameters=["-ac", "1", "-sample_fmt", "s16"])
        return

    # 遍历 middle_silences，将每段静音压缩到 max_silence_keep
    segments = []
    prev_end = 0  # 上一个区段结束的时间戳ms，从 0 开始

    for (sil_start, sil_end) in middle_silences:
        # 非静音片段：从 prev_end 到 sil_start
        if sil_start > prev_end:
            segments.append(trimmed_audio[prev_end:sil_start])

        # 静音片段：从 sil_start 到 sil_end
        cur_sil_len = sil_end - sil_start
        if cur_sil_len <= max_silence_keep:
            # 如果静音时长本来就 <= 阈值，直接完整保留
            segments.append(trimmed_audio[sil_start:sil_end])
        else:
            # 如果静音时长 > 阈值，只保留前 max_silence_keep 毫秒的静音
            segments.append(trimmed_audio[sil_start: sil_start + max_silence_keep])

        prev_end = sil_end  # 下一次从当前静音结束的地方开始

    # 处理最后一个静音之后的尾部非静音片段
    if prev_end < len(trimmed_audio):
        segments.append(trimmed_audio[prev_end:])

    # 将所有片段拼接起来
    processed = segments[0]
    for seg in segments[1:]:
        processed += seg

    # 导出为单声道 16-bit WAV
    processed.export(
        output_path,
        format="wav",
        parameters=[
            "-ac", "1",  # 单声道
            "-sample_fmt", "s16"  # 16-bit PCM
        ]
    )

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


def audio_phone_like_process(input_audio_path, output_audio_path, low_center=1900, high_center=800) -> AudioSegment:
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
