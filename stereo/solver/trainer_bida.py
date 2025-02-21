# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import time
import torch

from stereo.modeling.models.bidastereo.eval_utils import eval_batch, aggregate_eval_results
from tabulate import tabulate
from .trainer import Trainer


class BIDATrainer(Trainer):
    def __init__(self, args, cfg, logger, tb_writer):
        for each in cfg.train_loader.all_dataset:
            each.logger = logger
        for each in cfg.val_loader.all_dataset:
            each.logger = logger
            each.augmentations = cfg.augmentations.val

        super().__init__(args, cfg, logger, tb_writer)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):
        evaluate_result = []
        for i, data in enumerate(self.val_set):
            for k, v in data.items():
                if k in ['img', 'disp', 'valid_disp', 'mask']:
                    data[k] = torch.from_numpy(v)
                # data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v

            batch_dict = {
                "stereo_video": data["img"],  # [num_frames, 2(l&r), 3(c), h, w]
                "disparity": data["disp"][:, 0].abs(),  # 取左图disp绝对值 [num_frames, 1(c), h, w]
                "disparity_mask": data["valid_disp"][:, 0].unsqueeze(1),  # 取左图valid_disp [num_frames, 1, h, w]
                "fg_mask": data["valid_disp"][:, 0].unsqueeze(1)
            }
            if "mask" in data:
                batch_dict["fg_mask"] = data["mask"][:, 0].unsqueeze(1)

            disp_preds = []
            video = batch_dict["stereo_video"]
            num_ims = len(video)
            kernel_size = self.cfg.runtime_params.bida_ksize
            if kernel_size >= num_ims:
                with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
                    infer_start = time.time()
                    predictions = self.model({'img': video.to(self.local_rank).unsqueeze(0)})  # [num_frames, bz, channel, h, w]
                    infer_time = time.time() - infer_start
                predictions = predictions.cpu()
            else:
                stride = kernel_size // 2
                for img_i in range(0, num_ims, stride):
                    imgs = video[img_i: min(img_i + kernel_size, num_ims)]
                    with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
                        infer_start = time.time()
                        predictions = self.model({'img': imgs.to(self.local_rank).unsqueeze(0)})  # [num_frames, bz, channel, h, w]
                        infer_time = time.time() - infer_start

                    predictions = predictions.cpu()
                    if len(disp_preds) > 0 and len(predictions) >= stride:
                        if len(predictions) < kernel_size:
                            disp_preds.append(predictions[stride // 2:])
                        else:
                            disp_preds.append(predictions[stride // 2: -stride // 2])
                    elif len(disp_preds) == 0:
                        disp_preds.append(predictions[: -stride // 2])
                predictions = torch.cat(disp_preds)

            if 'pad' in data:
                pad = data['pad']
                pad_top, pad_right, pad_bottom, pad_left = pad[0], pad[1], pad[2], pad[3]
                ht, wd = predictions.shape[-2:]
                predictions = predictions[..., pad_top: ht-pad_bottom, pad_left: wd-pad_right]

            predictions = predictions[:, 0].abs()
            predictions = predictions * batch_dict["disparity_mask"].round()

            batch_eval_result, seq_length = eval_batch(batch_dict, {'disparity': predictions})
            # metrics = sorted(list(batch_eval_result.keys()), key=lambda x: x.metric)
            # self.logger.info(tabulate([[metric, batch_eval_result[metric]] for metric in metrics]))
            evaluate_result.append((batch_eval_result, seq_length))

            if i % self.cfg.runtime_params.log_period == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.val_loader), infer_time * 1000)
                self.logger.info(message)

        results = aggregate_eval_results(evaluate_result)
        metrics = sorted(list(results.keys()), key=lambda x: x.metric)

        self.logger.info(f"Epoch {current_epoch} metrics: ")
        self.logger.info(tabulate([[metric, results[metric]] for metric in metrics]))
