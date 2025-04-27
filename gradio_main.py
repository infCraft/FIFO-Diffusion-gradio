import gradio as gr
import random
import videocrafter_main as vm
import sample_fifo

def on_generate_click(mode, prompt, seed, video_length):
    if mode == "VideoCrafter2 - Single Prompt":
        video_path, error = vm.craft(prompt, seed, video_length)
    elif mode == "VideoCrafter2 - Multi-Prompt":
        prompts = []
        lengths = []
        for line in prompt.split('\n'):
            if line.strip():
                parts = line.split('.')
                if len(parts) >= 2:
                    pro = "".join(parts[:-1]).strip()
                    prompts.append(pro)
                    lengths.append(int(parts[-1].strip()))
        video_path, error = vm.craft2(seed, prompts, lengths)
    else:
        version = "65x512x512" if mode == "OpenSora - 65x512x512" else "221x512x512"
        
        print(version)
        video_path, error = sample_fifo.craft(prompt, seed, video_length, version)
    
    if error:
        gr.Warning(error, duration=5)
        return None
    else:
        return video_path

if __name__ == "__main__":
    with gr.Blocks(title="FIFO-Diffusion") as demo:
        gr.HTML("""
            <h1 style='text-align: center;'>FIFO-Diffusion Model Demo - by 黄祯 2022302121425</h1>
            <h2 style='text-align: center; color: magenta;'>使用说明</h2>
            <p style='text-align: center;'>1. 选择你所需要的模型</p>
            <p style='text-align: center;'>2. 根据格式输入prompt</p>
            <p style='text-align: center;'>3. 调整seed和视频长度</p>
            <p style='text-align: center;'>4. 点击Generate即可开始生成视频（时间可能会比较长）</p>
        """)
        
        mode_selector = gr.Radio(["VideoCrafter2 - Single Prompt", "VideoCrafter2 - Multi-Prompt", "OpenSora - 65x512x512", "OpenSora - 221x512x512"],
                                 label="Choose Mode", value="VideoCrafter2 - Single Prompt")
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(label="Text Prompt",
                                          placeholder="Please type your prompt here...",
                                          lines=5)
                seed_input = gr.Slider(minimum=0, maximum=2**32 - 1, step=1, label="Seed",
                                       value=random.randint(0, 2**32 - 1))
                video_length_input = gr.Slider(minimum=1, maximum=100, step=1,
                                               label="Video Length (only for Single Prompt Mode)",
                                               value=10)
                generate_button = gr.Button("Generate")
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video", format="mp4")

        def update_placeholder(mode):
            if mode != "VideoCrafter2 - Multi-Prompt":
                return gr.update(placeholder="Please type your prompt here...")
            else:
                return gr.update(placeholder="Please type your prompt here...\nFormat: [prompt]. [video length (in seconds)]\nFor example:\nA man walking in a forest, 4K. 5\nHe saw a mysterious castle, 4K. 3")

        mode_selector.change(update_placeholder, inputs=[mode_selector], outputs=[prompt_input])
        generate_button.click(on_generate_click, 
                              inputs=[mode_selector, prompt_input, seed_input, video_length_input],
                              outputs=[output_video])

        # 添加 Examples（示例）
        gr.Markdown("## Examples")
        
        example_data = [
            ["VideoCrafter2 - Single Prompt", "A panoramic view of the Milky Way, ultra HD.", "results/samples/video1.mp4"],
            ["VideoCrafter2 - Multi-Prompt", "A teddy bear walking on the street, 4K, high resolution. 10\nA teddy bear standing on the street, 4K, high resolution. 10\nA teddy bear dancing on the street, 4K, high resolution. 10", "results/samples/video2.mp4"],
            ["OpenSora - 65x512x512", "A stunning young woman gracefully walking through a serene winter forest, where towering trees are blanketed in pristine white snow. The soft glow of ambient light filters through the branches, casting a magical atmosphere. Ultra-high-definition, 4K resolution, with crisp details and cinematic quality.", "results/samples/video3.mp4"],
            ["OpenSora - 221x512x512", "A breathtaking underwater world adorned with vibrant, multicolored coral reefs. The crystal-clear water shimmers as sunlight filters through, casting a mesmerizing glow on the marine life.", "results/samples/video4.mp4"]
        ]

        def load_example(example):
            return example[0], example[1], example[2]

        examples = gr.Examples(
            examples=example_data,
            inputs=[mode_selector, prompt_input, output_video],
            label="Select an Example to Autofill"
        )


    demo.launch(server_port=7860)

