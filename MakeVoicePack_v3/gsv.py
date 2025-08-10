import shutil
from gradio import Warning, Error
from gradio_client import handle_file
import os

def produce_parallel(client, audio_name: str, text: str, output_folder: str,
                     ref_audio_path: str, ref_text: str,
                     aux_ref_audio_paths: list[str], prompt_lang: str, text_lang: str,
                     speed: float):
    aux_ref_audio_files = []
    for aux in aux_ref_audio_paths:
        aux_ref_audio_files.append(handle_file(aux))
    try:
        result = client.predict(
            text=text,
            text_lang=text_lang,
            ref_audio_path=handle_file(ref_audio_path),
            aux_ref_audio_paths=aux_ref_audio_files,
            prompt_text=ref_text,
            prompt_lang=prompt_lang,
            top_k=5,
            top_p=1,
            temperature=1,
            text_split_method="凑四句一切",
            batch_size=20,
            speed_factor=speed,
            ref_text_free=False,
            split_bucket=True,
            fragment_interval=0.2,
            seed=-1,
            keep_random=True,
            parallel_infer=True,
            repetition_penalty=1.35,
            sample_steps="32",
            super_sampling=False,
            api_name="/inference"
        )
        shutil.move(result[0], os.path.join(output_folder, audio_name))
        return True
    except ValueError as e:
        if "Cannot find a function with `api_name`" in str(e):
            Warning('无法连接到名称为"/inference"的api，请检查是否开启并行推理版本。尝试使用非并行推理版本中...', duration=20)
            try:
                result_2 = client.predict(
                    ref_wav_path=handle_file(ref_audio_path),
                    prompt_text=ref_text,
                    prompt_language=prompt_lang,
                    text=text,
                    text_language=text_lang,
                    how_to_cut="凑四句一切",
                    top_k=15,
                    top_p=1,
                    temperature=1,
                    ref_free=False,
                    speed=speed,
                    if_freeze=False,
                    inp_refs=None,
                    sample_steps="8",
                    if_sr=False,
                    pause_second=0.2,
                    api_name="/get_tts_wav"
                )
                shutil.move(result_2, os.path.join(output_folder, audio_name))
                Warning('可使用非并行推理的api:"/get_tts_wav"。请在GSV中选择并行版本的推理页面。', duration=20)
                return False
            except ValueError as e2:
                if "Cannot find a function with `api_name`" in str(e2):
                    Error('无法解决的错误:仍然无法连接名称为"/get_tts_wav"的api')
                    return False
