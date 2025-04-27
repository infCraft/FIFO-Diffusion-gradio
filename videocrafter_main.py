from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import torch

try:
    import torch_npu

    npu_is_available = True
    from torch_npu.contrib import transfer_to_npu
except:
    npu_is_available = False

import numpy as np
from PIL import Image
import imageio

from pytorch_lightning import seed_everything

from scripts.evaluation.funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_gif
from scripts.evaluation.funcs import base_ddim_sampling, fifo_ddim_sampling, fifo_ddim_sampling_multiprompts
from utils.utils import instantiate_from_config
from lvdm.models.samplers.ddim import DDIMSampler


def set_directory(args, prompt):
    if args.output_dir is None:
        output_dir = f"results/videocraft_v2_fifo/random_noise/{prompt[:100]}"
        if args.eta != 1.0:
            output_dir += f"/eta{args.eta}"

        if args.new_video_length != 100:
            output_dir += f"/{args.new_video_length}frames"
        if not args.lookahead_denoising:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/no_lookahead_denoising")
        if args.num_partitions != 4:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/n={args.num_partitions}")
        if args.video_length != 16:
            output_dir = output_dir.replace(f"{prompt[:100]}", f"{prompt[:100]}/f={args.video_length}")

    else:
        output_dir = args.output_dir

    latents_dir = f"results/videocraft_v2_fifo/latents/{args.num_inference_steps}steps/{prompt[:100]}/eta{args.eta}"

    print("The results will be saved in", output_dir)
    print("The latents will be saved in", latents_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)

    return output_dir, latents_dir


def main(args):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    # data_config = config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    if not os.path.exists(args.ckpt_path):
        return None, f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    if not ((args.height % 16 == 0) and (args.width % 16 == 0)):
        return None, "Error: image size [h,w] should be multiples of 16!"

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = args.video_length
    channels = model.channels

    ## step 2: load data
    ## -----------------------------------------------------------------
    if not args.prompt:
        return None, "Error: prompt is empty! "
    prompt = args.prompt

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    output_dir, latents_dir = set_directory(args, prompt)

    batch_size = 1
    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

    prompts = [prompt]
    text_emb = model.get_learned_conditioning(prompts)

    cond = {"c_crossattn": [text_emb], "fps": fps}

    ## inference
    is_run_base = not (os.path.exists(latents_dir + f"/{args.num_inference_steps}.pt") and os.path.exists(
        latents_dir + f"/0.pt"))
    if not is_run_base:
        ddim_sampler = DDIMSampler(model)
        ddim_sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.eta, verbose=False)
    else:
        base_tensor, ddim_sampler, _ = base_ddim_sampling(model, cond, noise_shape, \
                                                          args.num_inference_steps, args.eta,
                                                          args.unconditional_guidance_scale, \
                                                          latents_dir=latents_dir)
        save_gif(base_tensor, output_dir, "origin")

    # 根据参数选择使用哪个采样函数
    # if args.use_memory:
    #     # 使用记忆增强版采样
    #     from scripts.evaluation.funcs import memory_enhanced_fifo_ddim_sampling
    #     video_frames = memory_enhanced_fifo_ddim_sampling(
    #         args, model, cond, noise_shape, ddim_sampler, args.unconditional_guidance_scale, 
    #         output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
    #     )
    # 使用原始采样
    video_frames = fifo_ddim_sampling(
        args, model, cond, noise_shape, ddim_sampler, args.unconditional_guidance_scale, 
        output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
    )

    output_path = output_dir + "/video"

    imageio.mimsave(output_path + ".mp4", video_frames[-args.new_video_length:], fps=args.output_fps)
    # else:
    #     imageio.mimsave(output_path+".gif", video_frames[-args.new_video_length:], duration=int(1000/args.output_fps))

    return output_path + ".mp4", None


def craft(prompt, seed, video_length):
    if not prompt:
        return None, "Error: prompt is empty! "
    if video_length <= 0:
        return None, "Error: video_length should be greater than 0! "
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='videocrafter_models/base_512_v2/model.ckpt',
                        help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--seed", type=int, default=seed)
    # parser.add_argument("--num_inference_steps", type=int, default=16,
    #                     help="number of inference steps, it will be f * n forcedly")

    # FIFO-Diffusion的视频参数设置
    parser.add_argument("--prompt", "-p", type=str, default=prompt, help="path to the prompt file")
    parser.add_argument("--new_video_length", "-l", type=int, default=10*video_length,
                        help="N in paper; desired length of the output video")
    parser.add_argument("--output_fps", type=int, default=10, help="fps of the output video")


    parser.add_argument("--num_processes", type=int, default=1,
                        help="number of processes if you want to run only the subset of the prompts")
    parser.add_argument("--rank", type=int, default=0, help="rank of the process(0~num_processes-1)")
    parser.add_argument("--height", type=int, default=320, help="height of the output video")
    parser.add_argument("--width", type=int, default=512, help="width of the output video")
    parser.add_argument("--save_frames", action="store_true", default=False, help="save generated frames for each step")

    # VideoCrafter2自带的参数设置
    parser.add_argument("--video_length", type=int, default=16, help="f in paper")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="n in paper")
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--lookahead_denoising", "-ld", action="store_false", default=True)
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")
    # parser.add_argument("--use_mp4", action="store_true", default=False, help="use mp4 format for the output video")


    args = parser.parse_args()

    args.num_inference_steps = args.video_length * args.num_partitions

    seed_everything(args.seed)

    video, msg = main(args)
    return video, msg


def craft2(seed, multiprompts, prompts_length):
    """
    使用多个提示生成视频，每个提示控制一段视频内容
    
    参数:
    prompt: 基础提示（用于设置输出路径）
    seed: 随机种子
    multiprompts: 提示列表，如 ["森林场景", "城堡场景", "洞穴场景"]
    prompts_length: 每个提示对应的帧数列表，如 [30, 40, 50]
    
    返回:
    video: 生成的视频路径
    msg: 错误信息（如果有）
    """
    if len(multiprompts) != len(prompts_length) or len(multiprompts) == 0:
        return None, "Error: prompt or multiprompts is empty or their length is not fit!"
    
    # 将prompts_length转换为字符串格式
    prompts_length_str = ','.join([str(length*10) for length in prompts_length])
    
    # 创建完整的multiprompts列表（包含提示和长度信息）
    full_multiprompts = multiprompts + [prompts_length_str]
    
    print(f"full_multiprompts: {full_multiprompts}") 

    # 计算总视频长度
    total_length = sum(prompts_length)
    # print(f"total_length: {total_length}")
    
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='videocrafter_models/base_512_v2/model.ckpt',
                        help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config (yaml) path")
    parser.add_argument("--seed", type=int, default=seed)

    # FIFO-Diffusion的视频参数设置
    parser.add_argument("--prompt", "-p", type=str, default=multiprompts[0], help="base prompt for output directory")
    parser.add_argument("--new_video_length", "-l", type=int, default=total_length*10,
                        help="N in paper; desired length of the output video")
    parser.add_argument("--output_fps", type=int, default=10, help="fps of the output video")

    parser.add_argument("--num_processes", type=int, default=1,
                        help="number of processes if you want to run only the subset of the prompts")
    parser.add_argument("--rank", type=int, default=0, help="rank of the process(0~num_processes-1)")
    parser.add_argument("--height", type=int, default=320, help="height of the output video")
    parser.add_argument("--width", type=int, default=512, help="width of the output video")
    parser.add_argument("--save_frames", action="store_true", default=False, help="save generated frames for each step")

    # VideoCrafter2自带的参数设置
    parser.add_argument("--video_length", type=int, default=16, help="f in paper")
    parser.add_argument("--num_partitions", "-n", type=int, default=4, help="n in paper")
    parser.add_argument("--fps", type=int, default=8)

    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--lookahead_denoising", "-ld", action="store_false", default=True)
    parser.add_argument("--eta", "-e", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default=None, help="custom output directory")

    
    # 添加记忆增强选项
    parser.add_argument("--use_memory", action="store_true", default=True,
                      help="使用记忆增强机制保持视频一致性")

    args = parser.parse_args()

    args.num_inference_steps = args.video_length * args.num_partitions

    seed_everything(args.seed)

    video, msg = main_multiprompts(args, full_multiprompts)
    return video, msg

def main_multiprompts(args, multiprompts):
    ## step 1: model config
    ## -----------------------------------------------------------------
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    if not os.path.exists(args.ckpt_path):
        return None, f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## sample shape
    if not ((args.height % 16 == 0) and (args.width % 16 == 0)):
        return None, "Error: image size [h,w] should be multiples of 16!"

    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    frames = args.video_length
    channels = model.channels

    ## step 2: load data
    ## -----------------------------------------------------------------
    if not args.prompt:
        return None, "Error: prompt is empty! "
    prompt = args.prompt

    ## step 3: run over samples
    ## -----------------------------------------------------------------
    output_dir, latents_dir = set_directory(args, prompt)

    batch_size = 1
    noise_shape = [batch_size, channels, frames, h, w]
    fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

    # 使用第一个提示作为基础提示进行初始化
    base_prompt = multiprompts[0]
    prompts = [base_prompt]
    text_emb = model.get_learned_conditioning(prompts)

    cond = {"c_crossattn": [text_emb], "fps": fps}

    ## inference
    is_run_base = not (os.path.exists(latents_dir + f"/{args.num_inference_steps}.pt") and os.path.exists(
        latents_dir + f"/0.pt"))
    if not is_run_base:
        ddim_sampler = DDIMSampler(model)
        ddim_sampler.make_schedule(ddim_num_steps=args.num_inference_steps, ddim_eta=args.eta, verbose=False)
    else:
        base_tensor, ddim_sampler, _ = base_ddim_sampling(model, cond, noise_shape, \
                                                          args.num_inference_steps, args.eta,
                                                          args.unconditional_guidance_scale, \
                                                          latents_dir=latents_dir)
        save_gif(base_tensor, output_dir, "origin")

    if args.use_memory:
        # 使用记忆增强版采样
        from scripts.evaluation.funcs import memory_enhanced_fifo_ddim_sampling_multiprompts
        video_frames = memory_enhanced_fifo_ddim_sampling_multiprompts(
            args, model, cond, noise_shape, ddim_sampler, multiprompts, args.unconditional_guidance_scale, 
            output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
        )
    else:
    # 使用多提示版本的FIFO采样
        video_frames = fifo_ddim_sampling_multiprompts(
            args, model, cond, noise_shape, ddim_sampler, multiprompts, args.unconditional_guidance_scale, 
            output_dir=output_dir, latents_dir=latents_dir, save_frames=args.save_frames
        )
    output_path = output_dir + "/video"

    imageio.mimsave(output_path + ".mp4", video_frames[-args.new_video_length:], fps=args.output_fps)

    return output_path + ".mp4", None

if __name__ == "__main__":
    # print(torch_npu)  # 检查是否包含 'npu'
    craft("A beautiful Milky Way, ultra HD.", 123, 10)

