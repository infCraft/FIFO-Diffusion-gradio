# FIFO-Diffusion: Generating Infinite Videos from Text without Training (NeurIPS 2024)

## News📢
- v1.0.1 引入MemoryBank记忆机制，结合高斯低通滤波器实现增强模型全局注意力

## Cuda部署FIFO-Diffusion on VideoCrafter2 & Open-Sora v1.1.0

### 1. 克隆本仓库
```sh
git clone https://github.com/infCraft/FIFO-Diffusion-gradio.git
cd FIFO-Diffusion_public
```

然后克隆Open-Sora v1.1.0的分支
```sh
git clone --branch v1.1.0 https://github.com/PKU-YuanGroup/Open-Sora-Plan.git 
```


### 2. 环境配置
这一步可以在conda当中或者python的venv中进行。
```sh
cd Open-Sora-Plan
pip install -e .
pip install deepspeed
pip install gradio==4.44.1
pip install huggingface_hub==0.24.0
pip install kornia
pip install open_clip_torch==2.24.0
```

### 3. 下载模型权重
#### 3.1 VideoCrafter2
|Model|Resolution|Checkpoint
|:----|:---------|:---------
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)

将权重文件按如下格式放置：
```
FIFO-Diffusion_public
    └── videocrafter_models
        └── base_512_v2
            └── model.ckpt      # VideoCrafter2 checkpoint
```

#### 3.2 Open-Sora v1.1.0
```sh
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

# In FIFO-Diffusion_public
git clone https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0
```

### 4. 开始推理
#### 4.1 gradio前端
脚本已写好，直接运行即可
```sh
./start.sh
```

#### 4.2 单独运行Open-Sora模型推理

```sh
# Run with Open-Sora Plan v1.1.0, 65x512x512 model
# Requires about 40GB VRAM with A6000. It uses n=8 by default.
sh scripts/opensora_fifo_65.sh

# Run with Open-Sora Plan v1.1.0, 221x512x512 model
# Requires about 40GB VRAM with A6000. It uses n=4 by default.
sh scripts/opensora_fifo_221.sh
```


