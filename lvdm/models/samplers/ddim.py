import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               latents_dir=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    latents_dir=latents_dir,
                                                    **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None, latents_dir=None,
                      **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device) # [1,4,16,40,64]
        else:
            img = x_T
        
        if timesteps is None: # True
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        if verbose:
            iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        else:
            iterator = time_range

        init_x0 = False

        for i, step in enumerate(iterator):
            if i == 0 and latents_dir is not None:
                torch.save(img, f"{latents_dir}/{i}.pt")
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long) # [1]
            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      x0=x0,
                                      **kwargs)
            
            img, pred_x0 = outs
        if latents_dir is not None:
            torch.save(img, f"{latents_dir}/{total_steps}.pt")

        return img, intermediates

    @torch.no_grad()
    def fifo_onestep(self, cond, shape, latents=None, timesteps=None, indices=None,
                unconditional_guidance_scale=1., unconditional_conditioning=None,
                memory_context=None, **kwargs):
        # device = self.model.betas.device        
        # b, _, f, _, _ = shape

        # ts = torch.Tensor(timesteps.copy()).to(device=device, dtype=torch.long) # [16]
        # noise_pred = self.unet(latents, cond, ts,
        #                         unconditional_guidance_scale=unconditional_guidance_scale,
        #                         unconditional_conditioning=unconditional_conditioning,
        #                         **kwargs)
        
        # latents, pred_x0 = self.ddim_step(latents, noise_pred, indices)

        # return latents, pred_x0

        """
        Modified fifo_onestep to incorporate memory context for improved temporal consistency
        """
        b, _, f, _, _ = shape
        device = self.model.betas.device
        
        if latents is None:
            img = torch.randn(shape, device=self.model.device)
        else:
            img = latents

        ts = torch.Tensor(timesteps.copy()).to(device=device, dtype=torch.long) # [16]
        
        # 获取当前的噪声预测
        noise_pred = self.unet(img, cond, ts, unconditional_guidance_scale, unconditional_conditioning, **kwargs)
        
        # 如果提供了记忆上下文，使用它来增强噪声预测
        if memory_context is not None:
            # 记忆上下文影响噪声预测的强度
            memory_weight = 0.1
            
            # 将记忆上下文调整为与噪声预测相同的形状
            # 假设memory_context的形状为 [batch_size, feature_dim]
            _, c, f, h, w = noise_pred.shape
            
            # 只有当记忆已初始化时才应用
            if torch.sum(torch.abs(memory_context)) > 1e-6:
                # 将记忆特征投影到噪声空间
                # 这需要额外的层，简单实现可以使用线性投影
                if not hasattr(self, 'memory_projector'):
                    # 创建投影层，将记忆特征投影到噪声空间
                    feature_dim = memory_context.shape[1]
                    self.memory_projector = torch.nn.Linear(feature_dim, c).to(self.model.device)
                    # 初始化为小权重，以便轻微影响
                    torch.nn.init.normal_(self.memory_projector.weight, mean=0.0, std=0.01)
                    
                # 投影记忆特征
                memory_noise = self.memory_projector(memory_context)
                
                # 调整形状以匹配噪声预测
                memory_noise = memory_noise.view(b, c, 1, 1, 1).expand(b, c, f, h, w)
                
                # 融合噪声预测和记忆特征
                noise_pred = (1 - memory_weight) * noise_pred + memory_weight * memory_noise
        
        # 应用DDIM步骤
        output, pred_x0 = self.ddim_step(img, noise_pred, indices)
        
        return output, pred_x0

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError
            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)

        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            pred_x0 /= scale_t 
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
    
    @torch.no_grad()
    def unet(self, x, c, t, unconditional_guidance_scale=1.,
             unconditional_conditioning=None, **kwargs):
        
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            e_t = self.model.apply_model(x, t, c, **kwargs)
            e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            
            # text cfg
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        return e_t

    @torch.no_grad()
    def ddim_step(self, sample, noise_pred, indices):
        b, _, f, *_, device = *sample.shape, sample.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        size = (b, 1, 1, 1, 1)
        
        x_prevs = []
        pred_x0s = []

        for i, index in enumerate(indices):
            x = sample[:, :, [i]]
            e_t = noise_pred[:, :, [i]]
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)
            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            
            noise = sigma_t * noise_like(x.shape, device)

            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            x_prevs.append(x_prev)
            pred_x0s.append(pred_x0)

        x_prev = torch.cat(x_prevs, dim=2)
        pred_x0 = torch.cat(pred_x0s, dim=2)

        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

