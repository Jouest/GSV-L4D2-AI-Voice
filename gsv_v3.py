from gradio_client import handle_file
import os
import shutil

def produce_v3(client, output_dir: str, audio_name: str, text: str,
               ref_audio_path: str, ref_text: str,
               if_sr: bool, prompt_lang: str, text_lang: str):
    result = client.predict(
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
        speed=1.0,
        if_freeze=False,
        inp_refs=None,
        sample_steps=32,
        if_sr=if_sr,
        pause_second=0.2,
        api_name="/get_tts_wav"
    )
    shutil.move(result, os.path.join(output_dir, audio_name))
