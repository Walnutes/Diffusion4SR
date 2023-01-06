import math
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torch.nn.functional as F
from torch import device, einsum, nn
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    """
    Emphasis: Warm UP!
    Return: list [linear_start, .... ,linear_end, linear_end,linear_end ]
            warmup_frac < 1 ,经过 warmup_time 时间/迭代
            从start到end值,此后保持end值不变
    """
    betas = linear_end * np.ones(n_timestep, dtype=np.float64) # n_timestep*n_timestep 
                                                               # value=linear_end
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    # Notice: warmup_frac < 1
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """
    Emphasis: Schedule
    Return: 不同的 β schedule的方法
    """
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  
        # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /n_timestep + cosine_s
        )
        # 此处range初值为0！因为在下方通过求取变化率而对其维数=n_timestamp
        
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1] 
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        
        # betas = [linear_start, .... ,linear_end, linear_end,linear_end ]
        # tensor转换为numpy，下处求累积
        
        alphas = 1. - betas
        # 重要等式
        
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # cumprod：累积α；
        # 然后从1开始，舍去最后一累积值构成 cumprod_prev
        # cumprod_prev：计算角度真正需要的累积序列
        
        # TODO:从推理公式可以看到反向推导阶段确实从 累积α (t-1)时刻开始代入
        # TODO:目前不理解为什么需要append (1.),是仅仅为了对齐吗？ (1.)这一数值有无特殊含义，
        # TODO:目前来看 因为第一个值十分接近 1.0（β 的第一个值非常接近 0.0）
        
        self.sqrt_alphas_cumprod_prev = np.sqrt( np.append(1., alphas_cumprod) )
        # TODO:根号α 是从t时刻就需要的，因此需要补齐
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # BUG：或许上式中的alpha_cumprod_tm1应该是alphas_cumprod_prev
        # TODO：目前求得的posterior_variance似乎是演草纸的倒数 ？
        
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        """ 
        从Xt直接推到X0
        """
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
            
    def q_posterior(self, x_start, x_t, t):
        """
        反向推理: 已知X0和Xt推出Xt-1的均值和方差,利用贝叶斯公式。
        """
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None): 
        """
        用p(x_{t-1}|x_t)拟合目标q(x_{t-1}|x_t),已知的是x_t,和每一步的噪声即 α 和 β 值
        因此先从x_t,预测x_0(即下面的x_recon),进而可以通过q_posterior的反向推理求的下一步
        x_{t-1}的均值和方差！
        """
        batch_size = x.shape[0] # N*C*W*H
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        
        # 求出此t时刻的噪声系数(level)，由于是sqrt_alphas_cumprod_prev，故索引需要+1
        # TODO：不清楚denoise_fn是什么函数？似乎是config中的一个参数，不明白这个函数的名字含义
        # 接着根据当前时刻t的Xt直接推导X0即下面的X_recon，相当于恢复的过程，实质便是拟合
        
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
                # dim=1,通道数增加
            
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        # 有了恢复的X_recon便可以作为x0再次代入反向推理的过程计算Xt-1的均值和方差
        # TODO：此处不明白的地方是函数的名字为什么是P_mean_variance
        
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        反向的一步采样: 从上一步的噪声图像x_t恢复x_{t-1}
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        """
        输入x_in即x_t,初始化正态噪声图像img
        对噪声图像进行num_timesteps次的采样:首先得到x_recon,再一步步得到x_{t-1},直到最终的结果:预测的x_0
        """
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            # BUG:在unconditional的情况，上面一行maybe：x_in.shape
            # Sorry，下面函数直接传递元组数据(batch_size, channels, image_size, image_size)
            # 不对啊
            
            img = torch.randn(shape, device=device)
            ret_img = img
            
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            # Now parameter 'continous'=false, so ...
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)
    
    # BUG:没有明白这里在conditional的情况下，元组(batch_size, channels, image_size, image_size)怎么取shape
    # 我猜测这里应该没有使用这个函数，下面的函数和它实现的功能基本相同，下面的输入x_in应该就是一张正儿八经的图像了
    # 返回一个超像素处理的图像
    
    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        正向过程: 无情对初始图像x_start加噪声
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        # 随机整数
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        """
        均匀分布: 分出b个值分别用于batchsize张图片
        """
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)
            
        # 这里恢复的是噪声图像,如果可以精准的生成噪声图像?
        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
