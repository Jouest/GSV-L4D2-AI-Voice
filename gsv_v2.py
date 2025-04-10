import shutil
from gradio_client import handle_file
import os

def produce_v2(client, audio_name: str, text: str, output_folder: str,
               ref_audio_path: str, ref_text: str,
               aux_ref_audio_paths: list[str], prompt_lang: str, text_lang: str):
    auxs = []
    for aux in aux_ref_audio_paths:
        auxs.append(handle_file(aux))
    result = client.predict(
        text=text,
        text_lang=text_lang,
        ref_audio_path=handle_file(ref_audio_path),
        aux_ref_audio_paths=auxs,
        prompt_text=ref_text,
        prompt_lang=prompt_lang,
        top_k=5,
        top_p=1,
        temperature=1,
        text_split_method="凑四句一切",
        batch_size=20,
        speed_factor=1,
        ref_text_free=False,
        split_bucket=True,
        fragment_interval=0.2,
        seed=-1,
        keep_random=True,
        parallel_infer=True,
        repetition_penalty=1.35,
        api_name="/inference"
    )
    shutil.move(result[0], os.path.join(output_folder, audio_name))
