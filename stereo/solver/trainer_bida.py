# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import time
import torch
import torch.distributed as dist

from stereo.modeling.models.bidastereo.eval_utils import eval_batch, aggregate_eval_results
from collections import defaultdict
from stereo.utils import common_utils
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
        all_indexes = []

        evaluate_result = []
        for i, data in enumerate(self.val_loader):
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v

            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = data["img"][0]
            batch_dict["disparity"] = data["disp"][0][:, 0].abs()
            batch_dict["disparity_mask"] = data["valid_disp"][0][:, :1]
            if "mask" in data:
                batch_dict["fg_mask"] = data["mask"][0][:, :1]
            else:
                batch_dict["fg_mask"] = torch.ones_like(batch_dict["disparity_mask"])

            with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
                infer_start = time.time()
                predictions = self.model(data)
                infer_time = time.time() - infer_start

            if 'pad' in data:
                pad = data['pad'].squeeze()
                pad_top, pad_right, pad_bottom, pad_left = pad[0], pad[1], pad[2], pad[3]
                ht, wd = predictions.shape[-2:]
                predictions = predictions[..., pad_top: ht-pad_bottom, pad_left: wd-pad_right]

            predictions = predictions.squeeze(1).abs()[:, :1]
            predictions = predictions * batch_dict["disparity_mask"].round()

            batch_eval_result, seq_length = eval_batch(batch_dict, {'disparity': predictions})
            evaluate_result.append((batch_eval_result, seq_length))

        result = aggregate_eval_results(evaluate_result,)
        print(1)

        #     all_indexes.extend(data['index'].tolist())
        #
        #     if i % self.cfg.runtime_params.log_period == 0:
        #         message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
        #                    ).format(current_epoch, i, len(self.val_loader), infer_time * 1000)
        #         self.logger.info(message)
        #
        # # gather from all gpus
        # if self.args.dist_mode:
        #     dist.barrier()
        #     self.logger.info("Start reduce metrics.")
        #     # gather index
        #     indexes = torch.tensor(all_indexes).to(self.local_rank)
        #     gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
        #     dist.all_gather(gathered_indexes, indexes)
        #     indexes = torch.cat(gathered_indexes)
        #     # gather metrics
        #     for each in metric_names:
        #         metric = torch.tensor(all_metrics[each]).to(self.local_rank)
        #         gathered_metric = [torch.zeros_like(metric) for _ in range(dist.get_world_size())]
        #         dist.all_gather(gathered_metric, metric)
        #         metric = torch.cat(gathered_metric)
        #
        #         unique_dict = {}
        #         for key, value in zip(indexes.tolist(), metric.tolist()):
        #             if key not in unique_dict:
        #                 unique_dict[key] = value
        #         all_metrics[each] = list(unique_dict.values())
        #
        # results = {}
        # for each in metric_names:
        #     results[each] = round(torch.tensor(all_metrics[each]).mean().item(), 2)
        #
        # if self.global_rank == 0 and self.tb_writer is not None:
        #     tb_info = {}
        #     for k, v in results.items():
        #         tb_info[f'scalar/val/{k}'] = v
        #
        #     common_utils.write_tensorboard(self.tb_writer, tb_info, current_epoch)
        #
        # self.logger.info(f"Epoch {current_epoch} metrics: {results}")
