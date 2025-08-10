from utils import remove_audio_blank, batch_process_audio

def remove_audios_blank(
        input_folder: str,
        output_folder: str,
        silence_thresh: int = -40,  # dBFS，判断静音的阈值
        min_silence_len: int = 200,  # ms，最小静音长度，用来检测静音段
        max_silence_keep: int = 500  # ms，中间静音片段最多保留的长度
) -> str:
    count = batch_process_audio(
        input_folder,
        output_folder,
        remove_audio_blank,
        desc="消除首尾静音中...",
        silence_thresh=silence_thresh,
        min_silence_len=min_silence_len,
        max_silence_keep=max_silence_keep
    )
    return f"去除音频空白任务完成，共对{count}个音频进行了处理"
