import torch
import torch.nn as nn


class DDIMPipeline:
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    @torch.no_grad()
    def __call__(self, cond_feature, eta: float = 0.0, num_inference_steps: int = 50):
        # random noise
        image_with_noise = torch.randn(cond_feature.shape, device=cond_feature.device, dtype=cond_feature.dtype)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        res = []
        for t in self.scheduler.timesteps:
            # predict noise
            pred_noise = self.model(sample=image_with_noise,
                                    timesteps=t.to(cond_feature.device),
                                    condition=cond_feature)
            # do x_t -> x_t-1
            image_with_noise = self.scheduler.step(pred_noise,
                                                   t,
                                                   image_with_noise,
                                                   eta=eta,
                                                   use_clipped_model_output=True)['prev_sample']

            res.append(image_with_noise)

        return res


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Stereo2DModel(nn.Module):
    def __init__(self, channels_in, num_train_timesteps):
        super().__init__()
        self.time_proj = nn.Embedding(num_train_timesteps, channels_in)
        self.time_embedding = TimestepEmbedding(channels_in, channels_in)

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_in),
            nn.SiLU(),
            nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_in),
            nn.SiLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_in),
            nn.SiLU(),
            nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_in),
            nn.SiLU()
        )

    def forward(self, sample, timesteps, condition):
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)  # [bz,]
        t_emb = self.time_proj(timesteps)  # [bz, 48]
        emb = self.time_embedding(t_emb)[..., None, None]  # [bz, 48, 1, 1]
        emb = condition + emb

        sample = self.conv1(sample) + emb
        sample = self.conv2(sample)

        return sample