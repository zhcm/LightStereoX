# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import time
import torch
import torch.distributed as dist

from stereo.utils import common_utils
from .trainer import Trainer


class RBHMTrainer(Trainer):
    def __init__(self, args, cfg, logger, tb_writer):
        super().__init__(args, cfg, logger, tb_writer)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):
        all_indexes = []
        metric_names = ['abs_h']
        all_metrics = {'abs_h': []}

        for i, data in enumerate(self.val_loader):
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v

            with torch.cuda.amp.autocast(enabled=self.cfg.runtime_params.mixed_precision):
                infer_start = time.time()
                model_pred = self.model(data)
                infer_time = time.time() - infer_start

            import numpy as np
            height_map = model_pred['pred_height']
            res = height_map.cpu().numpy() * data['bump_mask'].cpu().numpy()
            res = np.max(res, axis=1)
            for batch_index in range(res.shape[0]):
                temp = res[batch_index]
                temp = temp[temp > 0]
                if len(temp) > 0:
                    pred_h = temp.mean()
                else:
                    pred_h = 0
                all_metrics['abs_h'].append(abs(pred_h - data['height'][batch_index]))

            all_indexes.extend(data['index'].tolist())

            if i % self.cfg.runtime_params.log_period == 0:
                message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                           ).format(current_epoch, i, len(self.val_loader), infer_time * 1000)
                self.logger.info(message)

        # gather from all gpus
        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            # gather index
            indexes = torch.tensor(all_indexes).to(self.local_rank)
            gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_indexes, indexes)
            indexes = torch.cat(gathered_indexes)
            # gather metrics
            for each in metric_names:
                metric = torch.tensor(all_metrics[each]).to(self.local_rank)
                gathered_metric = [torch.zeros_like(metric) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_metric, metric)
                metric = torch.cat(gathered_metric)

                unique_dict = {}
                for key, value in zip(indexes.tolist(), metric.tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                all_metrics[each] = list(unique_dict.values())

        results = {}
        for each in metric_names:
            results[each] = round(torch.tensor(all_metrics[each]).mean().item(), 2)

        if self.global_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for k, v in results.items():
                tb_info[f'scalar/val/{k}'] = v

            common_utils.write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")
