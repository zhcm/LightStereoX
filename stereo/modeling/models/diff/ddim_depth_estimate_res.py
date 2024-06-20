from typing import Union, Dict, Tuple, Optional
import torch
from stereo.models.diff.scheduling_ddim import DDIMScheduler
from torch import nn


class DDIMDepthEstimateRes(nn.Module):
    def __init__(self, inference_steps=20, num_train_timesteps=1000, fpn_dim=96):
        super(DDIMDepthEstimateRes, self).__init__()

        channels_in = fpn_dim
        self.model = ScheduledCNNRefine(channels_in=channels_in, channels_noise=channels_in)
        self.diffusion_inference_steps = inference_steps
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        self.pipeline = CNNDDIMPipiline(self.model, self.scheduler)

    def forward(self, fp, features_left, disp_gt_prob):
        """
        :param fp:
        :param features_left:
        :param disp_gt_prob:
        :return:
        """
        refined_feat = self.pipeline(
            device=fp.device,
            dtype=fp.dtype,
            shape=fp.shape,
            input_args=(fp, features_left),
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False)['images']

        noise = torch.randn(refined_feat.shape).to(refined_feat.device)
        bs = refined_feat.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=refined_feat.device).long()
        noisy_images = self.scheduler.add_noise(disp_gt_prob, noise, timesteps)

        refine_module_inputs = (fp, features_left)
        noise_pred = self.model(noisy_images, timesteps, *refine_module_inputs)

        output = {'refined_feat': refined_feat,
                  'noise_pred': noise_pred,
                  'noise': noise}

        return output


class CNNDDIMPipiline:
    def __init__(self, model, scheduler):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def __call__(self, device, dtype, shape, input_args,
            generator: Optional[torch.Generator] = None,
            eta: float = 0.0, num_inference_steps: int = 50,
            return_dict: bool = True, **kwargs) -> Union[Dict, Tuple]:

        image = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            model_output = self.model(image, t.to(device), *input_args)
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=True,
                generator=generator)['prev_sample']

        return {'images': image}

class ScheduledCNNRefine(nn.Module):
    def __init__(self, channels_in, channels_noise, **kwargs):
        super().__init__()
        self.noise_embedding = nn.Sequential(
            nn.Conv2d(channels_noise, channels_noise, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_noise),
            # 不能用batch norm，会统计输入方差，方差会不停的变
            nn.ReLU(True),
            nn.Conv2d(channels_noise, channels_in, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_in)
        )

        self.time_embedding = nn.Embedding(1280, channels_in)

        self.pred = nn.Sequential(
            nn.Conv2d(channels_in, channels_noise, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_noise),
            nn.ReLU(True),
            nn.Conv2d(channels_noise, channels_noise, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(4, channels_noise)
        )

    def forward(self, noisy_image, t, *args):
        feat, features_left = args
        # print('debug: feat shape {}'.format(feat.shape))
        # diff = (noisy_image - blur_depth).abs()
        if t.numel() == 1:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]
            # feat = feat + self.time_embedding(t)[None, :, None, None]
            # t 如果本身是一个值，需要扩充第一个bs维度 (这个暂时不适用)
        else:
            # print(t)
            feat = feat + self.time_embedding(t)[..., None, None]

        ret = self.pred(feat)

        return ret
