from utils import remove_audio_blank, accelerate_audio
import os
import gradio as gr

desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
port = 1145

def accelerate(rate: float):
    dir1 = os.path.join(desktop_path, 'audio.wav')
    dir2 = os.path.join(desktop_path, 'remove_blank.wav')
    dir3 = os.path.join(desktop_path, 'speed_up.wav')
    if os.path.exists(dir2):
        os.remove(dir2)
    if os.path.exists(dir3):
        os.remove(dir3)
    if os.path.exists(dir1):
        remove_audio_blank(dir1, dir2)
        accelerate_audio(dir2, dir3, ratio=rate)
        os.remove(dir1)
    return dir1, dir2, dir3

with gr.Blocks(title="快速对音频进行加速处理") as demo:
    gr.Markdown("## 移除空白，加速音频")
    with gr.Row():
        generate_btn = gr.Button("开始处理", variant="primary")
        process = gr.Progress(True)
    with gr.Row():
        audio_output1 = gr.Audio(label="初始音频")
        audio_output2 = gr.Audio(label="空白移除音频")
        audio_output3 = gr.Audio(label="空白移除并加速音频")

    generate_btn.click(
        fn=accelerate,
        inputs=
        [gr.Slider(minimum=1.0, maximum=1.8, show_reset_button=True, step=0.05, label='speed rate',
                   value=1.2)],
        outputs=[audio_output1, audio_output2, audio_output3]
    )

demo.launch(server_port=port, show_error=True, allowed_paths=[desktop_path])
